import json
import os

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'report', 'eval_results.json')


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}


def save_results(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def update_results(model_name, dataset, hyperparams, metrics):
    """
    Save results for a (model, dataset) pair.
    Only updates if the new f1_score is better than the existing one.

    Args:
        model_name (str): e.g. 'BaselineDNN', 'LSTM'
        dataset (str): e.g. 'MR', 'Semeval2017A'
        hyperparams (dict): e.g. {'emb_dim': 50, 'epochs': 55, ...}
        metrics (dict): e.g. {'accuracy': 0.73, 'f1': 0.72, 'recall': 0.74}
    """
    results = load_results()
    key = f"{model_name}_{dataset}"

    existing = results.get(key, {})
    existing_f1 = existing.get('metrics', {}).get('f1', -1)

    if metrics['f1'] > existing_f1:
        results[key] = {
            'model': model_name,
            'dataset': dataset,
            'hyperparams': hyperparams,
            'metrics': metrics,
        }
        save_results(results)
        print(f"Results updated for {key} (f1: {existing_f1:.4f} -> {metrics['f1']:.4f})")
    else:
        print(f"Results NOT updated for {key} (existing f1 {existing_f1:.4f} >= new f1 {metrics['f1']:.4f})")
