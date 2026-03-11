# Foundation Models Setup

This folder contains the pipeline for downloading, caching, and pretraining the foundation models required for embedding extraction (CXR, ECG, and EHR MOTOR).

## 🚀 Run the Full Pipeline

To execute all phases sequentially with strict error handling, run the master script:

```bash
./scripts/03_foundation_models.sh
```

## 🧪 Validation & Testing

After the pipeline completes, run the validation test suite.

```bash
# Pipeline integrity, artifact validation, and weight presence checks
pytest tests/test_motor_pipeline.py tests/test_foundation_weights.py
```

## 🚨 Important Note

The EHR MOTOR model and TorchXRay Vision model are not available on the Hugging Face Hub. The MOTOR foundation model requires pretraining at the moment.