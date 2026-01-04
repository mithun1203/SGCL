"""
SCP (Semantic Consistency Preservation) Evaluation

Evaluates trained models on:
1. Semantic Consistency Score (how well model maintains coherent knowledge)
2. Contradiction Rate (frequency of contradictory outputs)
3. Forgetting Score (performance degradation on old tasks)
4. Task-specific Accuracy

This is the KEY evaluation that proves SG-CL works better than baselines.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util


@dataclass
class EvaluationConfig:
    """Configuration for SCP evaluation."""
    model_path: str
    test_tasks: List[List[str]]
    task_names: List[str]
    knowledge_base_path: Optional[str] = None
    contradiction_threshold: float = 0.3  # Cosine similarity threshold for contradictions
    max_gen_length: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SCPEvaluator:
    """
    Semantic Consistency Preservation Evaluator
    
    Measures how well a model preserves semantic consistency after
    continual learning on multiple tasks.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
        # Load model
        print(f"Loading model from: {config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer for semantic analysis...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load knowledge base if provided
        self.knowledge_base = []
        if config.knowledge_base_path and Path(config.knowledge_base_path).exists():
            with open(config.knowledge_base_path) as f:
                kb = json.load(f)
                self.knowledge_base = kb.get('facts', [])
            print(f"Loaded knowledge base: {len(self.knowledge_base)} facts")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run full evaluation suite.
        
        Returns:
            Comprehensive evaluation results
        """
        print("\n" + "="*70)
        print("  SCP Evaluation Started")
        print("="*70)
        print(f"Model: {self.config.model_path}")
        print(f"Tasks: {len(self.config.test_tasks)}")
        print("="*70 + "\n")
        
        results = {
            'model_path': self.config.model_path,
            'task_names': self.config.task_names,
            'metrics': {}
        }
        
        # 1. Semantic Consistency Score
        print("\n[1/4] Computing Semantic Consistency Score...")
        consistency_score = self._compute_semantic_consistency()
        results['metrics']['semantic_consistency'] = consistency_score
        print(f"✓ Semantic Consistency Score: {consistency_score:.4f}")
        
        # 2. Contradiction Rate
        print("\n[2/4] Computing Contradiction Rate...")
        contradiction_rate = self._compute_contradiction_rate()
        results['metrics']['contradiction_rate'] = contradiction_rate
        print(f"✓ Contradiction Rate: {contradiction_rate:.4f}")
        
        # 3. Forgetting Score
        print("\n[3/4] Computing Forgetting Score...")
        forgetting_scores = self._compute_forgetting()
        results['metrics']['forgetting'] = forgetting_scores
        print(f"✓ Average Forgetting: {forgetting_scores['avg_forgetting']:.4f}")
        
        # 4. Per-Task Accuracy
        print("\n[4/4] Computing Per-Task Accuracy...")
        task_accuracies = self._compute_task_accuracy()
        results['metrics']['task_accuracy'] = task_accuracies
        print(f"✓ Average Accuracy: {task_accuracies['avg_accuracy']:.4f}")
        
        # Compute overall score
        overall_score = self._compute_overall_score(results['metrics'])
        results['overall_score'] = overall_score
        
        print("\n" + "="*70)
        print("  SCP Evaluation Completed")
        print("="*70)
        print(f"\n🎯 Overall SCP Score: {overall_score:.4f}")
        print(f"   Semantic Consistency: {consistency_score:.4f}")
        print(f"   Contradiction Rate: {contradiction_rate:.4f} (lower is better)")
        print(f"   Avg Forgetting: {forgetting_scores['avg_forgetting']:.4f} (lower is better)")
        print(f"   Avg Accuracy: {task_accuracies['avg_accuracy']:.4f}")
        print("="*70 + "\n")
        
        return results
    
    def _compute_semantic_consistency(self) -> float:
        """
        Measure semantic consistency across all tasks.
        
        Generates outputs for all task samples and measures how semantically
        consistent they are with expected knowledge.
        """
        all_outputs = []
        all_expected = []
        
        for task_data in self.config.test_tasks:
            for sample in task_data:
                # Generate model output
                output = self._generate_response(sample)
                all_outputs.append(output)
                all_expected.append(sample)
        
        # Compute semantic similarity between generated and expected
        if len(all_outputs) == 0:
            return 0.0
        
        # Encode sentences
        output_embeddings = self.sentence_model.encode(all_outputs, convert_to_tensor=True)
        expected_embeddings = self.sentence_model.encode(all_expected, convert_to_tensor=True)
        
        # Compute average cosine similarity
        similarities = util.cos_sim(output_embeddings, expected_embeddings)
        consistency_score = torch.diagonal(similarities).mean().item()
        
        return float(consistency_score)
    
    def _compute_contradiction_rate(self) -> float:
        """
        Measure frequency of contradictory outputs.
        
        Checks if model generates outputs that contradict itself or
        known facts from knowledge base.
        """
        contradictions = 0
        total_comparisons = 0
        
        # Generate outputs for all samples
        all_outputs = []
        for task_data in self.config.test_tasks:
            task_outputs = []
            for sample in task_data[:10]:  # Limit for efficiency
                output = self._generate_response(sample)
                task_outputs.append(output)
            all_outputs.extend(task_outputs)
        
        if len(all_outputs) < 2:
            return 0.0
        
        # Encode all outputs
        embeddings = self.sentence_model.encode(all_outputs, convert_to_tensor=True)
        
        # Check for contradictions: pairs with low semantic similarity
        # indicating opposite meanings
        for i in range(len(all_outputs)):
            for j in range(i + 1, min(i + 10, len(all_outputs))):  # Limit comparisons
                similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                
                # Check for negation patterns indicating contradiction
                if self._are_contradictory(all_outputs[i], all_outputs[j], similarity):
                    contradictions += 1
                
                total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        return contradictions / total_comparisons
    
    def _are_contradictory(self, text1: str, text2: str, similarity: float) -> bool:
        """
        Check if two texts are contradictory.
        
        Uses semantic similarity and negation keywords.
        """
        # Low similarity might indicate contradiction
        if similarity < self.config.contradiction_threshold:
            # Check for negation patterns
            negation_words = ['not', 'no', 'never', 'cannot', "can't", "don't", "doesn't"]
            
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            
            has_negation_1 = any(neg in text1_lower for neg in negation_words)
            has_negation_2 = any(neg in text2_lower for neg in negation_words)
            
            # Contradiction if one has negation and texts are about similar topics
            if has_negation_1 != has_negation_2:
                return True
        
        return False
    
    def _compute_forgetting(self) -> Dict[str, Any]:
        """
        Measure catastrophic forgetting across tasks.
        
        Evaluates performance on early tasks after training on later tasks.
        """
        task_perplexities = []
        
        for task_id, task_data in enumerate(self.config.test_tasks):
            # Compute perplexity on this task
            perplexities = []
            
            for sample in task_data[:20]:  # Limit for efficiency
                ppl = self._compute_perplexity(sample)
                perplexities.append(ppl)
            
            avg_ppl = np.mean(perplexities) if perplexities else float('inf')
            task_perplexities.append(avg_ppl)
        
        # Forgetting: how much worse are early tasks compared to later ones
        if len(task_perplexities) < 2:
            return {
                'task_perplexities': task_perplexities,
                'avg_forgetting': 0.0
            }
        
        # Compute forgetting as increase in perplexity for early tasks
        # relative to average perplexity
        avg_ppl = np.mean(task_perplexities)
        first_task_ppl = task_perplexities[0]
        
        # Forgetting score: (first_task_ppl - avg_ppl) / avg_ppl
        # Positive means first task has higher perplexity (worse) → forgetting
        forgetting = (first_task_ppl - avg_ppl) / avg_ppl if avg_ppl > 0 else 0.0
        
        return {
            'task_perplexities': [float(p) for p in task_perplexities],
            'avg_perplexity': float(avg_ppl),
            'first_task_perplexity': float(first_task_ppl),
            'avg_forgetting': float(forgetting)
        }
    
    def _compute_task_accuracy(self) -> Dict[str, Any]:
        """
        Compute per-task accuracy.
        
        Measures how well model performs on each task individually.
        """
        task_scores = []
        
        for task_id, task_data in enumerate(self.config.test_tasks):
            # For each task, compute average semantic similarity
            # between generated outputs and expected outputs
            similarities = []
            
            for sample in task_data[:20]:  # Limit for efficiency
                output = self._generate_response(sample)
                
                # Compute similarity with expected
                output_emb = self.sentence_model.encode(output, convert_to_tensor=True)
                expected_emb = self.sentence_model.encode(sample, convert_to_tensor=True)
                
                sim = util.cos_sim(output_emb, expected_emb).item()
                similarities.append(sim)
            
            task_accuracy = np.mean(similarities) if similarities else 0.0
            task_scores.append(task_accuracy)
        
        return {
            'task_scores': [float(s) for s in task_scores],
            'avg_accuracy': float(np.mean(task_scores)) if task_scores else 0.0
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate model response for given prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_gen_length,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity for given text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def _compute_overall_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall SCP score.
        
        Combines all metrics into a single score (0-1, higher is better).
        """
        # Normalize and combine metrics
        consistency = metrics['semantic_consistency']
        contradiction = 1.0 - metrics['contradiction_rate']  # Invert (lower contradiction is better)
        forgetting = 1.0 - min(1.0, max(0.0, metrics['forgetting']['avg_forgetting']))  # Invert and clip
        accuracy = metrics['task_accuracy']['avg_accuracy']
        
        # Weighted average
        overall = (
            0.30 * consistency +
            0.25 * contradiction +
            0.25 * forgetting +
            0.20 * accuracy
        )
        
        return float(overall)
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        results = self.evaluate()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")


# ==============================================================================
# Convenience Functions
# ==============================================================================

def evaluate_model(
    model_path: str,
    test_tasks: List[List[str]],
    task_names: List[str],
    output_path: Optional[str] = None,
    **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained model.
    
    Args:
        model_path: Path to trained model
        test_tasks: List of test task data
        task_names: Names of tasks
        output_path: Where to save results (optional)
        **config_kwargs: Additional config overrides
    
    Returns:
        Evaluation results dictionary
    """
    config = EvaluationConfig(
        model_path=model_path,
        test_tasks=test_tasks,
        task_names=task_names,
        **config_kwargs
    )
    
    evaluator = SCPEvaluator(config)
    results = evaluator.evaluate()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return results


def compare_methods(
    model_paths: Dict[str, str],
    test_tasks: List[List[str]],
    task_names: List[str],
    output_dir: str = "evaluation_results"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple methods (SG-CL vs baselines).
    
    Args:
        model_paths: Dict mapping method name to model path
        test_tasks: Test data
        task_names: Task names
        output_dir: Where to save results
    
    Returns:
        Dict mapping method name to evaluation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for method_name, model_path in model_paths.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*70}\n")
        
        results = evaluate_model(
            model_path=model_path,
            test_tasks=test_tasks,
            task_names=task_names,
            output_path=str(output_path / f"{method_name}_results.json")
        )
        
        all_results[method_name] = results
    
    # Save comparison
    comparison_file = output_path / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    for method_name, results in all_results.items():
        score = results['overall_score']
        print(f"{method_name:20} | Overall Score: {score:.4f}")
    
    print(f"\nComparison saved to: {comparison_file}")
    
    return all_results


if __name__ == '__main__':
    # Demo evaluation
    print("SCP Evaluation Demo")
    print("="*70)
    
    # This would normally use real test tasks
    test_tasks = [
        ["Eagles have sharp talons.", "Birds can fly."],
        ["Penguins cannot fly.", "Penguins are birds."]
    ]
    
    task_names = ["General Birds", "Penguins"]
    
    # Evaluate (requires trained model)
    # results = evaluate_model(
    #     model_path="models/sgcl",
    #     test_tasks=test_tasks,
    #     task_names=task_names,
    #     output_path="evaluation_results/sgcl_eval.json"
    # )
    
    print("\nTo use:")
    print("  from scp_evaluation import evaluate_model, compare_methods")
    print("  results = evaluate_model('models/sgcl', test_tasks, task_names)")
