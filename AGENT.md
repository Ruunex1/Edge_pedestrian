# AGENTS.md — EdgePed

## 1. Project Overview

**Project Name:** EdgePed — Real-Time Pedestrian Detection and Edge AI Optimization

**Objective:** Build a production-quality computer vision project demonstrating pedestrian detection, robustness evaluation, inference optimization, benchmarking, and maintainable Python software engineering.

The project is intended as a professional portfolio project for Senior/Expert Machine Learning Engineer roles, particularly in automotive computer vision, ADAS, and embedded AI.

The initial development environment is:

- Operating System: Windows
- Hardware: CPU only
- Python: 3.11
- Deep Learning: PyTorch
- Detection Model: Pretrained YOLO11n
- Computer Vision: OpenCV
- Inference Optimization: ONNX Runtime
- Testing: pytest
- Linting: Ruff
- Configuration: YAML
- Version Control: Git

The system must be designed so that GPU or embedded hardware support can be added later without major architectural changes.

## 2. Main Goals

1. Implement a working pedestrian detection pipeline.
2. Evaluate detection accuracy using a public driving dataset.
3. Analyze robustness across different environmental conditions.
4. Optimize model inference for CPU execution.
5. Compare PyTorch and ONNX Runtime performance.
6. Benchmark latency, throughput, and memory consumption.
7. Implement clean, modular, maintainable Python code.
8. Provide automated tests and reproducible experiments.
9. Produce a professional technical report and GitHub repository.

The project must go beyond a basic YOLO demonstration and emphasize measurable engineering decisions.

## 3. Engineering Principles

All development must follow these rules:

- Use Python 3.11 and type hints.
- Follow clean code principles and separate responsibilities.
- Avoid monolithic scripts and unnecessary abstractions.
- Use modular components with clearly defined interfaces.
- Make inference backends interchangeable.
- Keep configuration outside application logic.
- Use logging instead of print statements in reusable modules.
- Handle errors explicitly.
- Write unit tests for core functionality.
- Use pytest for testing and Ruff for linting.
- Avoid hardcoded machine-specific paths.
- Ensure compatibility with Windows and CPU-only execution.
- Make experiments reproducible through documented configurations and dependencies.
- Never fabricate benchmark results or evaluation metrics.
- Document assumptions, limitations, and technical trade-offs.
- Prefer simple, reliable implementations over premature optimization.

## 4. Proposed Repository Structure

```text
edgeped/
├── configs/
│   ├── baseline.yaml
│   └── benchmark.yaml
├── data/
│   ├── samples/
│   └── README.md
├── src/
│   └── edgeped/
│       ├── __init__.py
│       ├── datasets/
│       │   └── bdd100k.py
│       ├── models/
│       │   ├── base.py
│       │   ├── pytorch_backend.py
│       │   └── onnx_backend.py
│       ├── preprocessing.py
│       ├── postprocessing.py
│       ├── detector.py
│       ├── evaluation.py
│       ├── benchmark.py
│       └── cli.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_postprocessing.py
│   ├── test_detector.py
│   └── test_inference.py
├── scripts/
├── reports/
├── outputs/
├── .gitignore
├── pyproject.toml
├── README.md
└── AGENTS.md
```

Only create modules when needed. Avoid empty placeholder implementations.

## 5. Implementation Roadmap

### Week 1 — Baseline Pedestrian Detection

**Goal:** Implement a working and reproducible pedestrian detection application on Windows using CPU inference.

Tasks:

1. Initialize the Python project and virtual environment.
2. Configure dependencies using `pyproject.toml`.
3. Install CPU-only PyTorch.
4. Integrate a pretrained YOLO11n model.
5. Implement image loading and preprocessing.
6. Implement pedestrian detection using the COCO `person` class.
7. Extract bounding boxes, confidence scores, and class labels.
8. Draw detection results on images.
9. Save annotated images and structured JSON results.
10. Measure initial inference latency using `time.perf_counter`.
11. Implement a command-line interface.
12. Add basic unit tests.
13. Add README setup and execution instructions.

**Deliverables:**

- Working CPU pedestrian detector.
- Command-line inference application.
- Annotated image output.
- JSON detection results.
- Initial latency measurement.
- Passing unit tests.

**Acceptance Criteria:**

- The application runs on Windows without CUDA.
- The model loads successfully.
- Images can be processed through the CLI.
- Person detections are extracted correctly.
- Outputs are saved successfully.
- Tests pass.
- Ruff reports no errors.

### Week 2 — Dataset Evaluation and Robustness

**Goal:** Establish meaningful accuracy metrics and understand model failures.

Tasks:

1. Integrate the BDD100K dataset.
2. Implement dataset loading and annotation parsing.
3. Define and document mapping between BDD100K pedestrian categories and COCO person detections.
4. Implement detection evaluation with correct IoU matching.
5. Calculate precision, recall, and average precision.
6. Evaluate results across daytime and nighttime subsets.
7. Analyze false positives and false negatives.
8. Investigate the impact of object size and occlusion where annotation data supports it.
9. Generate visualizations and evaluation reports.
10. Add tests for dataset parsing and evaluation logic.

**Deliverables:**

- Reproducible dataset evaluation pipeline.
- Baseline detection accuracy metrics.
- Daytime-versus-nighttime comparison.
- Error analysis report.

**Acceptance Criteria:**

- Dataset annotations are parsed correctly.
- Evaluation uses documented matching rules.
- Results can be reproduced.
- Evaluation outputs contain measured values, not estimates.

### Week 3 — ONNX Runtime Deployment

**Goal:** Introduce an optimized inference backend while maintaining prediction correctness.

Tasks:

1. Export the PyTorch model to ONNX.
2. Validate the exported ONNX model.
3. Implement an ONNX Runtime CPU inference backend.
4. Define a common inference backend interface.
5. Ensure preprocessing and postprocessing are consistent.
6. Compare PyTorch and ONNX predictions.
7. Investigate numerical differences.
8. Benchmark both implementations.
9. Separate model-only inference latency from end-to-end latency.
10. Document export and compatibility limitations.

**Deliverables:**

- Working ONNX Runtime backend.
- Interchangeable inference implementations.
- Prediction consistency report.
- Initial PyTorch-versus-ONNX benchmarks.

### Week 4 — Inference Optimization

**Goal:** Improve CPU inference efficiency while quantifying accuracy trade-offs.

Tasks:

1. Profile inference bottlenecks.
2. Investigate ONNX Runtime execution settings.
3. Experiment with CPU-supported INT8 quantization.
4. Prepare representative calibration data where required.
5. Evaluate quantized model accuracy.
6. Compare model latency and memory usage.
7. Measure throughput and latency distributions.
8. Investigate input resolution versus accuracy and performance.
9. Document optimization decisions and limitations.

Do not assume FP16 provides CPU acceleration. Only evaluate optimizations supported by the available hardware and runtime.

**Deliverables:**

- Optimized inference configurations.
- Accuracy-versus-latency comparison.
- Performance benchmark report.
- Engineering recommendation based on measured results.

### Week 5 — Software Engineering and Reliability

**Goal:** Make the application maintainable, testable, and reproducible.

Tasks:

1. Refactor modules where needed.
2. Improve configuration management.
3. Add type hints and input validation.
4. Implement structured logging.
5. Add unit tests for preprocessing and postprocessing.
6. Add integration tests for inference.
7. Test error handling and invalid inputs.
8. Add automated linting and testing through GitHub Actions.
9. Document dependency installation and supported environments.
10. Optionally introduce Docker if useful and practical.

**Deliverables:**

- Maintainable Python package.
- Automated test suite.
- CI pipeline.
- Reproducible execution instructions.

### Week 6 — Documentation and Portfolio Publication

**Goal:** Produce a professional, demonstrable engineering portfolio project.

Tasks:

1. Write a comprehensive README.
2. Document system architecture.
3. Explain model selection and design decisions.
4. Present accuracy and performance benchmarks.
5. Document nighttime robustness results.
6. Explain optimization trade-offs.
7. Include representative detection examples.
8. Create a short demonstration video.
9. Write a technical engineering report.
10. Prepare concise CV bullet points based on actual results.

**Deliverables:**

- Professional GitHub repository.
- Technical report.
- Benchmark tables and visualizations.
- Demonstration video.
- CV-ready project description.

## 6. Evaluation Metrics

The project should track:

**Detection quality**
- Precision
- Recall
- Average Precision (AP)
- AP at IoU 0.50
- False positives and false negatives
- Performance by environmental condition

**Inference performance**
- Mean latency
- Median latency (p50)
- p95 latency
- Throughput (FPS)
- Peak memory usage where measurable
- Model file size

Benchmarking must include warm-up runs, repeated measurements, documented hardware information, and consistent input configurations.

Do not compare performance numbers obtained under incompatible conditions without explaining the differences.

## 7. Testing Requirements

Tests should cover:

- Class-label mapping
- Bounding-box coordinate conversions
- Confidence filtering
- Non-maximum suppression where implemented by the project
- Invalid or empty inputs
- Dataset annotation parsing
- Evaluation metric calculations
- Inference result serialization
- Backend output consistency
- Configuration validation

Prefer fast unit tests that do not require downloading model weights. Use separate integration tests for real model inference.

## 8. Safety and Dataset Considerations

- This is a research and portfolio project, not a production automotive safety system.
- A pretrained COCO person detector is not equivalent to an automotive-validated pedestrian detector.
- Document dataset licensing and model licensing.
- Do not commit large datasets or pretrained model weights unless redistribution is permitted.
- Never claim automotive safety compliance.
- Clearly distinguish experimental results from production guarantees.

## 9. Instructions for the Coding Agent

When implementing the project:

1. Follow the roadmap in order.
2. Focus on the current milestone unless instructed otherwise.
3. Explain significant architectural decisions briefly.
4. Implement complete working functionality rather than pseudocode.
5. Keep changes small and reviewable.
6. Add or update tests when introducing functionality.
7. Run relevant tests and lint checks after changes.
8. Fix errors before declaring a task complete.
9. Do not introduce unnecessary frameworks or dependencies.
10. Do not silently change the project scope.
11. Do not invent performance results.
12. Document anything that cannot be tested in the current environment.
13. Preserve Windows and CPU compatibility.
14. Provide exact PowerShell commands for setup and execution.
15. Summarize completed work, test results, and remaining tasks after each milestone.

## 10. Immediate Next Task

**Implement Week 1 only.**

Start by creating the repository structure, configuring Python dependencies, and implementing a minimal working pedestrian detector.

Use pretrained YOLO11n with CPU execution.

The initial CLI should support:

```powershell
python -m edgeped.cli --input data/samples/street.jpg
```

Expected functionality:

- Load an input image.
- Run pedestrian detection.
- Extract bounding boxes and confidence scores.
- Draw detections.
- Save the annotated image.
- Save structured JSON results.
- Report inference latency.
- Handle invalid inputs.
- Pass basic unit tests.

Do not proceed to BDD100K evaluation, ONNX export, or quantization until Week 1 is functional and verified.

**Final objective:** Build an independently reproducible computer vision engineering project that demonstrates practical ML deployment, model evaluation, performance optimization, and strong Python software engineering skills.