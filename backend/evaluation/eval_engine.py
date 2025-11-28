import pandas as pd
import numpy as np
from backend.metrics.eval_metrics import EvalMetric
from backend.llms.llm_client import LLMClient

class Evaluator:
    def __init__(self, llm_client: LLMClient, eval_metrics: list[EvalMetric]):
        self.llm_client = llm_client
        self.eval_metrics = eval_metrics
        print("Evaluator intialized with metrics:", [m.metric_name for m in eval_metrics])

    def evaluate(self, dataset: pd.DataFrame):
        summaries = {}
        summaries['num_samples'] = len(dataset)
        print("Number of API calls estimated: ", len(dataset) * len(self.eval_metrics))

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
