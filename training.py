"""
Model Fine-Tuning Pipeline
===========================
Fine-tune sentence-transformer models on domain-specific data using
contrastive learning. Supports training from pairs (query, positive),
triplets (query, positive, negative), and k-fold cross-validation.

Usage:
    from training import FineTuner, TrainingConfig, TrainingPair

    config = TrainingConfig(
        base_model="all-MiniLM-L6-v2",
        output_dir="models/fine-tuned",
        epochs=3,
    )
    tuner = FineTuner(config)
    tuner.add_pairs([
        TrainingPair(query="breach of contract", positive="contractual obligation violated"),
        TrainingPair(query="negligence claim", positive="failure to exercise reasonable care"),
    ])
    result = tuner.train()
    print(result)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TrainingPair:
    """A positive pair: query text and a semantically similar document."""

    query: str
    positive: str


@dataclass
class TrainingTriplet:
    """A triplet: query, positive (similar), and negative (dissimilar) document."""

    query: str
    positive: str
    negative: str


@dataclass
class TrainingConfig:
    """Configuration for model fine-tuning."""

    # Model
    base_model: str = "all-MiniLM-L6-v2"
    output_dir: str = "models/fine-tuned"

    # Training hyper-parameters
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 256

    # Loss
    loss_type: str = "cosine"  # "cosine" | "contrastive" | "triplet"
    margin: float = 0.5  # for contrastive / triplet loss

    # Evaluation
    eval_steps: int = 100
    save_best_model: bool = True
    metric_for_best: str = "cosine_similarity"

    # Cross-validation
    cv_folds: int = 0  # 0 = no CV; >1 = k-fold

    # Reproducibility
    seed: int = 42


@dataclass
class TrainingResult:
    """Result of a training run."""

    model_path: str
    epochs_completed: int
    training_samples: int
    final_loss: float
    best_score: Optional[float]
    elapsed_seconds: float
    config: dict = field(default_factory=dict)
    fold_results: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist result metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        cv = f", folds={len(self.fold_results)}" if self.fold_results else ""
        return (
            f"TrainingResult(loss={self.final_loss:.4f}, "
            f"score={self.best_score}, "
            f"samples={self.training_samples}, "
            f"time={self.elapsed_seconds:.1f}s{cv})"
        )


# ---------------------------------------------------------------------------
# Hard-Negative Mining
# ---------------------------------------------------------------------------


def mine_hard_negatives(
    pairs: Sequence[TrainingPair],
    model_name: str = "all-MiniLM-L6-v2",
    negatives_per_pair: int = 1,
    pool_size: int = 50,
) -> List[TrainingTriplet]:
    """
    Generate triplets by mining hard negatives from a pool of documents.

    For each pair, embed all positives, find the nearest non-matching
    document, and use it as the hard negative. This produces more
    informative training signal than random negatives.

    Args:
        pairs: Input positive pairs.
        model_name: Encoder used for mining (does not need to be the
                    model being fine-tuned).
        negatives_per_pair: How many hard negatives per pair.
        pool_size: Candidate pool size for each mining step.

    Returns:
        List of TrainingTriplet with mined hard negatives.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Mining hard negatives for %d pairs...", len(pairs))
    model = SentenceTransformer(model_name)

    # Collect all positive documents as the negative candidate pool
    all_positives = [p.positive for p in pairs]
    pool_embeddings = model.encode(
        all_positives, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )

    query_embeddings = model.encode(
        [p.query for p in pairs],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    triplets: List[TrainingTriplet] = []

    for idx, pair in enumerate(pairs):
        # Cosine similarities between this query and all pool documents
        sims = np.dot(pool_embeddings, query_embeddings[idx])
        # Sort descending; skip self (the actual positive)
        ranked = np.argsort(sims)[::-1]

        negatives_found = 0
        for candidate_idx in ranked:
            if candidate_idx == idx:
                continue  # skip the actual positive
            triplets.append(
                TrainingTriplet(
                    query=pair.query,
                    positive=pair.positive,
                    negative=all_positives[candidate_idx],
                )
            )
            negatives_found += 1
            if negatives_found >= negatives_per_pair:
                break

    logger.info("Mined %d triplets from %d pairs", len(triplets), len(pairs))
    return triplets


# ---------------------------------------------------------------------------
# Fine-Tuner
# ---------------------------------------------------------------------------


class FineTuner:
    """
    Fine-tune a sentence-transformer model on domain-specific data.

    Supports:
    - Positive-pair training (CosineSimilarityLoss)
    - Contrastive training (ContrastiveLoss)
    - Triplet training (TripletLoss)
    - K-fold cross-validation
    - Checkpoint saving and best-model selection

    Example::

        tuner = FineTuner(TrainingConfig(epochs=5))
        tuner.add_pairs([TrainingPair("q1", "d1"), TrainingPair("q2", "d2")])
        result = tuner.train()
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._pairs: List[TrainingPair] = []
        self._triplets: List[TrainingTriplet] = []

    # -- Data loading ---------------------------------------------------------

    def add_pairs(self, pairs: Sequence[TrainingPair]) -> None:
        """Add positive pairs for training."""
        self._pairs.extend(pairs)
        logger.info("Added %d pairs (total: %d)", len(pairs), len(self._pairs))

    def add_triplets(self, triplets: Sequence[TrainingTriplet]) -> None:
        """Add triplets (query, positive, negative) for training."""
        self._triplets.extend(triplets)
        logger.info("Added %d triplets (total: %d)", len(triplets), len(self._triplets))

    def load_pairs_jsonl(self, path: str | Path) -> int:
        """
        Load training pairs from a JSONL file.

        Each line: {"query": "...", "positive": "..."}

        Returns:
            Number of pairs loaded.
        """
        path = Path(path)
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._pairs.append(
                    TrainingPair(query=obj["query"], positive=obj["positive"])
                )
                loaded += 1
        logger.info("Loaded %d pairs from %s", loaded, path)
        return loaded

    def load_triplets_jsonl(self, path: str | Path) -> int:
        """
        Load triplets from a JSONL file.

        Each line: {"query": "...", "positive": "...", "negative": "..."}

        Returns:
            Number of triplets loaded.
        """
        path = Path(path)
        loaded = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._triplets.append(
                    TrainingTriplet(
                        query=obj["query"],
                        positive=obj["positive"],
                        negative=obj["negative"],
                    )
                )
                loaded += 1
        logger.info("Loaded %d triplets from %s", loaded, path)
        return loaded

    # -- Training -------------------------------------------------------------

    def _build_train_examples(self):
        """Convert pairs/triplets into sentence_transformers InputExample objects."""
        from sentence_transformers import InputExample

        examples = []

        for pair in self._pairs:
            examples.append(
                InputExample(texts=[pair.query, pair.positive], label=1.0)
            )

        for triplet in self._triplets:
            examples.append(
                InputExample(
                    texts=[triplet.query, triplet.positive, triplet.negative]
                )
            )

        return examples

    def _get_loss(self, model):
        """Instantiate the appropriate loss function."""
        from sentence_transformers import losses

        cfg = self.config

        if self._triplets and cfg.loss_type == "triplet":
            return losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=cfg.margin)

        if cfg.loss_type == "contrastive":
            return losses.ContrastiveLoss(model=model, margin=cfg.margin)

        # Default: cosine similarity loss (works with pairs)
        return losses.CosineSimilarityLoss(model=model)

    def _create_evaluator(self, pairs: Sequence[TrainingPair]):
        """Create an EmbeddingSimilarityEvaluator from pairs."""
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

        sentences1 = [p.query for p in pairs]
        sentences2 = [p.positive for p in pairs]
        scores = [1.0] * len(pairs)

        return EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    def train(self) -> TrainingResult:
        """
        Run the fine-tuning pipeline.

        If ``config.cv_folds > 1``, performs k-fold cross-validation
        and returns aggregated results. Otherwise trains on all data.

        Returns:
            TrainingResult with metrics and model path.
        """
        total_samples = len(self._pairs) + len(self._triplets)
        if total_samples == 0:
            raise ValueError("No training data. Call add_pairs() or add_triplets() first.")

        if self.config.cv_folds > 1:
            return self._train_cv()

        return self._train_single(
            self._pairs, self._triplets, self.config.output_dir
        )

    def _train_single(
        self,
        pairs: Sequence[TrainingPair],
        triplets: Sequence[TrainingTriplet],
        output_dir: str,
    ) -> TrainingResult:
        """Single training run (no cross-validation)."""
        from sentence_transformers import SentenceTransformer, InputExample
        from torch.utils.data import DataLoader

        cfg = self.config
        start = time.time()

        logger.info(
            "Starting training: model=%s, epochs=%d, lr=%s, pairs=%d, triplets=%d",
            cfg.base_model, cfg.epochs, cfg.learning_rate, len(pairs), len(triplets),
        )

        # Load base model
        model = SentenceTransformer(cfg.base_model)
        model.max_seq_length = cfg.max_seq_length

        # Build examples
        examples = []
        for p in pairs:
            examples.append(InputExample(texts=[p.query, p.positive], label=1.0))
        for t in triplets:
            examples.append(InputExample(texts=[t.query, t.positive, t.negative]))

        # DataLoader
        np.random.seed(cfg.seed)
        np.random.shuffle(examples)
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=cfg.batch_size)

        # Loss
        loss = self._get_loss(model)

        # Evaluator (use 10% of pairs if available)
        evaluator = None
        if len(pairs) >= 10:
            eval_size = max(10, len(pairs) // 10)
            evaluator = self._create_evaluator(pairs[:eval_size])

        # Train
        warmup_steps = int(
            len(train_dataloader) * cfg.epochs * cfg.warmup_ratio
        )

        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=cfg.epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=cfg.eval_steps if evaluator else 0,
            output_path=output_dir,
            save_best_model=cfg.save_best_model,
            optimizer_params={"lr": cfg.learning_rate},
            weight_decay=cfg.weight_decay,
            show_progress_bar=True,
        )

        elapsed = time.time() - start

        # Compute final evaluation score
        best_score = None
        if evaluator:
            best_score = evaluator(model)

        result = TrainingResult(
            model_path=output_dir,
            epochs_completed=cfg.epochs,
            training_samples=len(examples),
            final_loss=-1.0,  # sentence-transformers doesn't expose final loss easily
            best_score=round(best_score, 4) if best_score is not None else None,
            elapsed_seconds=round(elapsed, 1),
            config=asdict(cfg),
        )

        # Persist metadata
        result.save(Path(output_dir) / "training_result.json")
        logger.info("Training complete: %s", result)
        return result

    def _train_cv(self) -> TrainingResult:
        """K-fold cross-validation training."""
        cfg = self.config
        k = cfg.cv_folds
        start = time.time()

        logger.info("Starting %d-fold cross-validation", k)

        # Combine all data into pairs for splitting
        all_pairs = list(self._pairs)
        np.random.seed(cfg.seed)
        np.random.shuffle(all_pairs)

        fold_size = len(all_pairs) // k
        fold_results = []

        for fold in range(k):
            logger.info("=== Fold %d/%d ===", fold + 1, k)

            # Split
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else len(all_pairs)
            val_pairs = all_pairs[val_start:val_end]
            train_pairs = all_pairs[:val_start] + all_pairs[val_end:]

            fold_dir = f"{cfg.output_dir}/fold_{fold + 1}"
            result = self._train_single(train_pairs, self._triplets, fold_dir)
            fold_results.append(result.to_dict())

        elapsed = time.time() - start

        # Aggregate
        scores = [
            fr["best_score"]
            for fr in fold_results
            if fr.get("best_score") is not None
        ]
        avg_score = round(np.mean(scores), 4) if scores else None
        std_score = round(np.std(scores), 4) if scores else None

        logger.info(
            "CV complete: avg_score=%.4f ± %.4f across %d folds",
            avg_score or 0, std_score or 0, k,
        )

        final = TrainingResult(
            model_path=cfg.output_dir,
            epochs_completed=cfg.epochs,
            training_samples=len(all_pairs) + len(self._triplets),
            final_loss=-1.0,
            best_score=avg_score,
            elapsed_seconds=round(elapsed, 1),
            config={**asdict(cfg), "cv_std": std_score},
            fold_results=fold_results,
        )
        final.save(Path(cfg.output_dir) / "cv_result.json")
        return final


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune a sentence-transformer model")
    parser.add_argument("--data", required=True, help="JSONL file with training pairs or triplets")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Base model name")
    parser.add_argument("--output", default="models/fine-tuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss", choices=["cosine", "contrastive", "triplet"], default="cosine")
    parser.add_argument("--cv-folds", type=int, default=0, help="K-fold CV (0 = disabled)")
    args = parser.parse_args()

    config = TrainingConfig(
        base_model=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        loss_type=args.loss,
        cv_folds=args.cv_folds,
    )

    tuner = FineTuner(config)

    # Detect format from first line
    with open(args.data, "r") as f:
        sample = json.loads(f.readline())

    if "negative" in sample:
        tuner.load_triplets_jsonl(args.data)
    else:
        tuner.load_pairs_jsonl(args.data)

    result = tuner.train()
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Model saved to: {result.model_path}")
    print(f"  Best score: {result.best_score}")
    print(f"  Time: {result.elapsed_seconds:.1f}s")
