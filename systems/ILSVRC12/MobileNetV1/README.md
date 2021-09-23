# MobileNetV1 for ImageNet

This package contains multiple example configurations for solving the
ImageNet classification problem with (quantized) MobileNetV1
configurations.

## Configurations

| Config              | Description                                       | Mixed-Prec. Search | Algorithm | Act. Rounding | Pretrained | Full Checkpoint | FP32 Accuracy | Final accuracy |
| ------------------- | ------------------------------------------------- | ------------------ | --------- | ------------- | ---------- | --------------- | ------------- | -------------- |
| `config.fp32.json`  | Full-precision configuration                      | N/A                | N/A       | N/A           | No         | TODO            | N/A           | 68.8%          |
| `config.pact.8b`    | 8b MobileNetV1 using the PACT algorithm           | No                 | PACT      | Yes           | TODO       | TODO            | 68.8%         | 69.2%          |
| `config.tqt.8b`     | 8b MobileNetV1 using the TQT algorithm            | No                 | TQT       | Yes           | TODO       | TODO            | 68.8%         | 69.4%          |
| `config.pact.400KB` | Memory-constrained mixed-precision MNv1 with PACT | PULP, 400KB L2     | PACT      | Yes           | TODO       | TODO            | 68.8%         | 65.9%          |
| `config.tqt.400KB`  | Memory-constrained mixed-precision MNv1 with TQT  | PULP, 400KB L2     | TQT       | Yes           | TODO       | TODO            | 68.8%         | 67.0%%         |
