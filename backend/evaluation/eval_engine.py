import pandas as pd
import numpy as np
from typing import Callable, Optional, Generator, Any
from backend.metrics.eval_metrics import EvalMetric
from backend.llms.llm_client import LLMClient


class Evaluator:
    def __init__(self, llm_client: LLMClient, eval_metrics: list[EvalMetric]):
        self.llm_client = llm_client
        self.eval_metrics = eval_metrics
        print("Evaluator intialized with metrics:", [m.metric_name for m in eval_metrics])

    def get_total_api_calls(self, dataset: pd.DataFrame) -> int:
        """Calculate total number of API calls needed."""
        return len(dataset) * len(self.eval_metrics)

    def evaluate(self, dataset: pd.DataFrame, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Evaluate the dataset with optional progress callback.
        
        Args:
            dataset: DataFrame with prompt/response columns
            progress_callback: Optional callback(completed, total, current_metric) for progress updates
        """
        summaries = {}
        summaries['num_samples'] = len(dataset)
        total_calls = self.get_total_api_calls(dataset)
        completed_calls = 0
        print("Number of API calls estimated: ", total_calls)

        for metric in self.eval_metrics:
            ratings = []
            explanations = []
            for idx, row in dataset.iterrows():
                # Extract variables dynamically from metric template
                template_vars = {}
                for key in metric.metric_prompt_template.split('{'):
                    if '}' in key:
                        var_name = key.split('}')[0]
                        template_vars[var_name] = row.get(var_name)
                prompt = metric.metric_prompt_template.format(**template_vars)
                response = self.llm_client.generate(prompt=prompt)
                
                # Update progress
                completed_calls += 1
                if progress_callback:
                    progress_callback(completed_calls, total_calls, metric.metric_name)
                
                # Validate response format strictly
                try:
                    if response is not None:
                        # Handle both parsed objects and JSON strings
                        if isinstance(response, self.llm_client.LLMResponse):
                            response_parsed = response
                        else:
                            response_parsed = self.llm_client.LLMResponse.model_validate_json(response)
                        # Explicitly check both fields exist and are valid
                        if not hasattr(response_parsed, 'rating') or not hasattr(response_parsed, 'explanation'):
                            raise ValueError("Missing 'rating' or 'explanation' in LLM response")
                        if not isinstance(response_parsed.rating, int):
                            raise ValueError("'rating' must be an integer")
                        if not isinstance(response_parsed.explanation, str):
                            raise ValueError("'explanation' must be a string")
                        ratings.append(response_parsed.rating)
                        explanations.append(response_parsed.explanation)
                    else:
                        ratings.append(-1)
                        explanations.append("N/A (no response)")
                except Exception as e:
                    ratings.append(-1)
                    explanations.append(f"Invalid response format: {str(e)}")
                    print(f"[Evaluator] Invalid response at row {idx} for metric '{metric.metric_name}': {e}")
            dataset[f"{metric.metric_name}_rating"] = ratings
            dataset[f"{metric.metric_name}_explanation"] = explanations
            summaries[metric.metric_name] = {
                "mean": np.mean([r for r in ratings if isinstance(r, int) and r != -1]) if ratings else 0.0,
                "std": np.std([r for r in ratings if isinstance(r, int) and r != -1]) if ratings else 0.0
            }
        return dataset, summaries

    def evaluate_with_progress(self, dataset: pd.DataFrame) -> Generator[dict[str, Any], None, None]:
        """
        Generator-based evaluation that yields progress updates and final results.
        
        Yields:
            Progress dicts: {"type": "progress", "completed": int, "total": int, "metric": str, "percent": float}
            Final result: {"type": "complete", "dataset": DataFrame, "summaries": dict}
        """
        summaries = {}
        summaries['num_samples'] = len(dataset)
        total_calls = self.get_total_api_calls(dataset)
        completed_calls = 0

        for metric in self.eval_metrics:
            ratings = []
            explanations = []
            for idx, row in dataset.iterrows():
                # Extract variables dynamically from metric template
                template_vars = {}
                for key in metric.metric_prompt_template.split('{'):
                    if '}' in key:
                        var_name = key.split('}')[0]
                        template_vars[var_name] = row.get(var_name)
                prompt = metric.metric_prompt_template.format(**template_vars)
                response = self.llm_client.generate(prompt=prompt)
                
                # Update progress
                completed_calls += 1
                percent = (completed_calls / total_calls) * 100 if total_calls > 0 else 100
                yield {
                    "type": "progress",
                    "completed": completed_calls,
                    "total": total_calls,
                    "metric": metric.metric_name,
                    "percent": round(percent, 1)
                }
                
                # Validate response format strictly
                try:
                    if response is not None:
                        if isinstance(response, self.llm_client.LLMResponse):
                            response_parsed = response
                        else:
                            response_parsed = self.llm_client.LLMResponse.model_validate_json(response)
                        if not hasattr(response_parsed, 'rating') or not hasattr(response_parsed, 'explanation'):
                            raise ValueError("Missing 'rating' or 'explanation' in LLM response")
                        if not isinstance(response_parsed.rating, int):
                            raise ValueError("'rating' must be an integer")
                        if not isinstance(response_parsed.explanation, str):
                            raise ValueError("'explanation' must be a string")
                        ratings.append(response_parsed.rating)
                        explanations.append(response_parsed.explanation)
                    else:
                        ratings.append(-1)
                        explanations.append("N/A (no response)")
                except Exception as e:
                    ratings.append(-1)
                    explanations.append(f"Invalid response format: {str(e)}")
                    print(f"[Evaluator] Invalid response at row {idx} for metric '{metric.metric_name}': {e}")
            
            dataset[f"{metric.metric_name}_rating"] = ratings
            dataset[f"{metric.metric_name}_explanation"] = explanations
            summaries[metric.metric_name] = {
                "mean": np.mean([r for r in ratings if isinstance(r, int) and r != -1]) if ratings else 0.0,
                "std": np.std([r for r in ratings if isinstance(r, int) and r != -1]) if ratings else 0.0
            }
        
        # Yield final results
        yield {
            "type": "complete",
            "dataset": dataset,
            "summaries": summaries
        }
