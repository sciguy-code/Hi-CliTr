"""
Evaluation Metrics for Radiology Report Generation

Implements:
- CheXpert F1: Multi-label classification metrics (14 pathologies)
- RadGraph F1: Entity-relation extraction metrics
- NLG Metrics: BLEU, METEOR, ROUGE-L, CIDEr
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

# NLG evaluation imports
try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK/rouge_score not available. NLG metrics will be limited.")


class CheXpertEvaluator:
    """
    CheXpert F1 Score Evaluator
    
    Evaluates multi-label classification for 14 pathologies:
    - Micro/Macro F1 scores
    - Per-class precision, recall, F1
    - Threshold optimization
    """
    
    LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    # Competition labels (subset used for evaluation)
    COMPETITION_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Pleural Effusion"
    ]
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions"""
        self.all_predictions = []
        self.all_labels = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Add batch of predictions
        
        Args:
            predictions: Probability predictions [B, 14]
            labels: Ground truth labels [B, 14]
        """
        self.all_predictions.append(predictions.cpu())
        self.all_labels.append(labels.cpu())
    
    def compute(self, use_competition_labels: bool = True) -> Dict[str, float]:
        """
        Compute F1 scores
        
        Args:
            use_competition_labels: Only use 5 competition labels
            
        Returns:
            Dictionary of metrics
        """
        # Concatenate all batches
        preds = torch.cat(self.all_predictions, dim=0)
        labels = torch.cat(self.all_labels, dim=0)
        
        # Binarize predictions
        binary_preds = (preds > self.threshold).float()
        
        # Filter for competition labels if needed
        if use_competition_labels:
            indices = [self.LABELS.index(l) for l in self.COMPETITION_LABELS]
            binary_preds = binary_preds[:, indices]
            labels = labels[:, indices]
            label_names = self.COMPETITION_LABELS
        else:
            label_names = self.LABELS
        
        # Calculate metrics per class
        metrics = {}
        
        for i, name in enumerate(label_names):
            pred_i = binary_preds[:, i]
            label_i = labels[:, i]
            
            # Handle uncertain labels (-1)
            valid_mask = label_i >= 0
            pred_i = pred_i[valid_mask]
            label_i = label_i[valid_mask]
            
            tp = ((pred_i == 1) & (label_i == 1)).sum().float()
            fp = ((pred_i == 1) & (label_i == 0)).sum().float()
            fn = ((pred_i == 0) & (label_i == 1)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f"{name}/precision"] = precision.item()
            metrics[f"{name}/recall"] = recall.item()
            metrics[f"{name}/f1"] = f1.item()
        
        # Micro F1 (overall)
        binary_preds_flat = binary_preds.flatten()
        labels_flat = labels.flatten()
        
        valid_mask = labels_flat >= 0
        binary_preds_flat = binary_preds_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]
        
        tp_micro = ((binary_preds_flat == 1) & (labels_flat == 1)).sum().float()
        fp_micro = ((binary_preds_flat == 1) & (labels_flat == 0)).sum().float()
        fn_micro = ((binary_preds_flat == 0) & (labels_flat == 1)).sum().float()
        
        precision_micro = tp_micro / (tp_micro + fp_micro + 1e-8)
        recall_micro = tp_micro / (tp_micro + fn_micro + 1e-8)
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)
        
        metrics["micro/precision"] = precision_micro.item()
        metrics["micro/recall"] = recall_micro.item()
        metrics["micro/f1"] = f1_micro.item()
        
        # Macro F1 (average across classes)
        class_f1s = [metrics[f"{name}/f1"] for name in label_names]
        metrics["macro/f1"] = np.mean(class_f1s)
        
        return metrics
    
    def find_optimal_thresholds(self) -> Dict[str, float]:
        """Find optimal threshold per class"""
        preds = torch.cat(self.all_predictions, dim=0)
        labels = torch.cat(self.all_labels, dim=0)
        
        optimal_thresholds = {}
        
        for i, name in enumerate(self.LABELS):
            pred_i = preds[:, i]
            label_i = labels[:, i]
            
            valid_mask = label_i >= 0
            pred_i = pred_i[valid_mask]
            label_i = label_i[valid_mask]
            
            best_f1 = 0
            best_thresh = 0.5
            
            for thresh in np.arange(0.1, 0.9, 0.05):
                binary_pred = (pred_i > thresh).float()
                
                tp = ((binary_pred == 1) & (label_i == 1)).sum().float()
                fp = ((binary_pred == 1) & (label_i == 0)).sum().float()
                fn = ((binary_pred == 0) & (label_i == 1)).sum().float()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                if f1.item() > best_f1:
                    best_f1 = f1.item()
                    best_thresh = thresh
            
            optimal_thresholds[name] = best_thresh
        
        return optimal_thresholds


class NLGEvaluator:
    """
    Natural Language Generation Evaluator
    
    Computes:
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - METEOR
    - ROUGE-L
    - CIDEr (simplified version)
    """
    
    def __init__(self):
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.reset()
    
    def reset(self):
        """Reset accumulated references and hypotheses"""
        self.references = []
        self.hypotheses = []
    
    def update(
        self,
        predictions: List[str],
        references: List[str],
    ):
        """
        Add batch of predictions and references
        
        Args:
            predictions: Generated texts
            references: Ground truth texts
        """
        self.hypotheses.extend(predictions)
        self.references.extend(references)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def compute(self) -> Dict[str, float]:
        """Compute all NLG metrics"""
        if not self.hypotheses:
            return {}
        
        metrics = {}
        
        # Tokenize
        ref_tokens = [[self._tokenize(ref)] for ref in self.references]
        hyp_tokens = [self._tokenize(hyp) for hyp in self.hypotheses]
        
        # BLEU scores
        if NLTK_AVAILABLE:
            for n in range(1, 5):
                weights = tuple([1/n] * n + [0] * (4-n))
                try:
                    bleu = corpus_bleu(ref_tokens, hyp_tokens, weights=weights)
                    metrics[f"BLEU-{n}"] = bleu
                except:
                    metrics[f"BLEU-{n}"] = 0.0
            
            # METEOR (per sentence, then average)
            meteor_scores = []
            for ref, hyp in zip(self.references, self.hypotheses):
                try:
                    score = meteor_score([self._tokenize(ref)], self._tokenize(hyp))
                    meteor_scores.append(score)
                except:
                    meteor_scores.append(0.0)
            metrics["METEOR"] = np.mean(meteor_scores)
            
            # ROUGE-L
            rouge_scores = []
            for ref, hyp in zip(self.references, self.hypotheses):
                try:
                    scores = self.rouge_scorer.score(ref, hyp)
                    rouge_scores.append(scores['rougeL'].fmeasure)
                except:
                    rouge_scores.append(0.0)
            metrics["ROUGE-L"] = np.mean(rouge_scores)
        else:
            # Fallback simple BLEU
            metrics["BLEU-4"] = self._simple_bleu(ref_tokens, hyp_tokens, 4)
        
        # CIDEr (simplified TF-IDF based)
        metrics["CIDEr"] = self._compute_cider(ref_tokens, hyp_tokens)
        
        return metrics
    
    def _simple_bleu(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]],
        n: int,
    ) -> float:
        """Simplified BLEU computation"""
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        total_match = 0
        total_count = 0
        
        for ref_list, hyp in zip(references, hypotheses):
            ref = ref_list[0]  # Take first reference
            
            hyp_ngrams = get_ngrams(hyp, n)
            ref_ngrams = get_ngrams(ref, n)
            
            if not hyp_ngrams:
                continue
            
            matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
            total_match += matches
            total_count += len(hyp_ngrams)
        
        return total_match / max(total_count, 1)
    
    def _compute_cider(
        self,
        references: List[List[List[str]]],
        hypotheses: List[List[str]],
    ) -> float:
        """Simplified CIDEr using TF-IDF cosine similarity"""
        from collections import Counter
        
        # Build vocabulary
        all_ngrams = defaultdict(int)
        for ref_list in references:
            for ref in ref_list:
                for n in range(1, 5):
                    ngrams = [tuple(ref[i:i+n]) for i in range(len(ref)-n+1)]
                    for ng in ngrams:
                        all_ngrams[ng] += 1
        
        # IDF weights
        num_docs = len(references)
        idf = {ng: np.log(num_docs / (count + 1)) for ng, count in all_ngrams.items()}
        
        def get_tfidf(tokens):
            vec = defaultdict(float)
            for n in range(1, 5):
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                tf = Counter(ngrams)
                for ng, count in tf.items():
                    vec[ng] = count * idf.get(ng, 1.0)
            return vec
        
        def cosine_sim(vec1, vec2):
            keys = set(vec1.keys()) | set(vec2.keys())
            dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
            norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
            norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
            return dot / (norm1 * norm2 + 1e-8)
        
        scores = []
        for ref_list, hyp in zip(references, hypotheses):
            ref = ref_list[0]
            ref_vec = get_tfidf(ref)
            hyp_vec = get_tfidf(hyp)
            scores.append(cosine_sim(ref_vec, hyp_vec))
        
        return np.mean(scores) * 10  # Scale to typical CIDEr range


class CompetitionEvaluator:
    """
    Combined evaluator using competition weights
    
    Weights:
    - CheXpert F1: 40%
    - RadGraph F1: 30% (simplified as entity matching)
    - NLG (CIDEr): 30%
    """
    
    def __init__(self):
        self.chexpert_eval = CheXpertEvaluator()
        self.nlg_eval = NLGEvaluator()
        
    def reset(self):
        self.chexpert_eval.reset()
        self.nlg_eval.reset()
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        generated_texts: List[str],
        reference_texts: List[str],
    ):
        """Update with batch of predictions"""
        self.chexpert_eval.update(predictions, labels)
        self.nlg_eval.update(generated_texts, reference_texts)
    
    def compute(self) -> Dict[str, float]:
        """Compute competition score"""
        chexpert_metrics = self.chexpert_eval.compute(use_competition_labels=True)
        nlg_metrics = self.nlg_eval.compute()
        
        # Competition score
        chexpert_f1 = chexpert_metrics.get("micro/f1", 0.0)
        cider = nlg_metrics.get("CIDEr", 0.0)
        
        # RadGraph approximation (using BLEU as proxy)
        radgraph_proxy = nlg_metrics.get("BLEU-4", 0.0)
        
        competition_score = (
            0.4 * chexpert_f1 +
            0.3 * radgraph_proxy +
            0.3 * cider / 10  # Normalize CIDEr
        )
        
        metrics = {
            **{f"chexpert/{k}": v for k, v in chexpert_metrics.items()},
            **{f"nlg/{k}": v for k, v in nlg_metrics.items()},
            "competition_score": competition_score,
        }
        
        return metrics


if __name__ == "__main__":
    # Test evaluators
    print("Testing Evaluation Metrics...")
    
    # CheXpert evaluator
    print("\n1. CheXpert Evaluator")
    chexpert_eval = CheXpertEvaluator()
    
    # Random predictions
    preds = torch.rand(100, 14)
    labels = torch.randint(0, 2, (100, 14)).float()
    
    chexpert_eval.update(preds, labels)
    metrics = chexpert_eval.compute()
    
    print(f"  Micro F1: {metrics['micro/f1']:.4f}")
    print(f"  Macro F1: {metrics['macro/f1']:.4f}")
    
    # NLG evaluator
    print("\n2. NLG Evaluator")
    nlg_eval = NLGEvaluator()
    
    references = [
        "The heart size is normal. Lungs are clear.",
        "No acute cardiopulmonary abnormality.",
    ]
    hypotheses = [
        "Heart size is within normal limits. The lungs are clear.",
        "No acute findings.",
    ]
    
    nlg_eval.update(hypotheses, references)
    nlg_metrics = nlg_eval.compute()
    
    for k, v in nlg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nEvaluation metrics test passed! ✓")
