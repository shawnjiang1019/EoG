

from typing import Any

import numpy as np

def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics
