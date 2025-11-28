class EvalMetric:
    def __init__(self, metric_name: str, metric_prompt_template: str):
        self.metric_name = metric_name
        self.metric_prompt_template = metric_prompt_template
