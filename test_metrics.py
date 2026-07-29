from src.training.metrics import calculate_metrics

y_true = [0, 1, 0, 1, 1]

y_pred = [0, 1, 0, 0, 1]

y_prob = [0.10, 0.90, 0.20, 0.40, 0.95]

metrics = calculate_metrics(
    y_true,
    y_pred,
    y_prob
)

print("=" * 60)

for k, v in metrics.items():
    print(f"{k:15} : {v:.4f}")

print("=" * 60)