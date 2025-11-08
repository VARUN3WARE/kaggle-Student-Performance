# Examples

This folder contains a small CLI example to demonstrate programmatic
inference using the saved model artifacts produced by `scripts/train.py`.

## infer_cli.py

- A simple command-line example to load the trained models and transformer
  from the `models/` directory and run a single prediction.
- Usage:

```bash
python examples/infer_cli.py --hours 8 --prev 50 --extra Yes --sleep 8 --papers 10
```

## Notes

- Ensure you have run `python scripts/train.py` (or the CI smoke train) so
  `models/transformer.pkl` and `models/best_*_model.pkl` exist.
- The example demonstrates how to call `src.predict.predict_from_inputs` in
  a small script (outside Streamlit).
