# Contributing to Fabric Defect Detection & Inspection

We welcome contributions from the community! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our Code of Conduct.

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the issue list to avoid duplicates. When filing a bug report, include:

- **Clear Description**: What the bug is and what you expected to happen
- **Steps to Reproduce**: Clear steps to reproduce the issue
- **Environment**: OS, Python version, GPU/CPU setup
- **Error Messages**: Full error traceback if applicable
- **Sample Data**: Small sample image that reproduces the issue (if possible)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub Issues. When creating one, include:

- **Clear Title**: Descriptive title for the enhancement
- **Motivation**: Why this enhancement would be useful
- **Proposed Implementation**: If you have an idea, describe it
- **Benefits**: How this benefits users or developers

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/fabric-inspection-prototype.git
   cd fabric-inspection-prototype
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b fix/issue-description
   ```

3. **Make Changes**
   - Follow PEP 8 style guidelines for Python code
   - Add comments for complex logic
   - Keep commits atomic and well-documented

4. **Test Locally**
   ```bash
   pip install -r requirements_cpu.txt
   python run.py
   # Test your changes thoroughly
   ```

5. **Commit with Clear Messages**
   ```bash
   git commit -m "Add feature: brief description of what was added"
   # Reference issues if applicable:
   git commit -m "Fix #123: description of the fix"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Go to GitHub and create a Pull Request
   - Write a clear PR description
   - Link related issues

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- Virtual environment tool (venv/virtualenv)

### Initial Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Unix

# Install development dependencies
pip install -r requirements_cpu.txt
pip install pytest pytest-cov black flake8

# Copy default config if needed
cp app/config.py app/config.py.example
```

## Code Style

- **Python**: Follow PEP 8
- **Tool**: Use `black` for formatting
  ```bash
  black app/ --line-length 88
  ```
- **Linting**: Use `flake8` to check for issues
  ```bash
  flake8 app/
  ```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_ml_models.py
```

## Documentation

- Update README.md if your changes affect user-facing behavior
- Add docstrings to new functions/classes
- Update relevant wiki pages
- Include inline comments for complex algorithms

## Commit Message Guidelines

- Start with a verb: "Add", "Fix", "Improve", "Refactor", "Document", etc.
- Keep first line under 50 characters
- Add detailed explanation if needed (blank line, then details)
- Reference issues: "Fixes #123" or "Related to #456"

Example:
```
Fix fabric type classifier accuracy on knitted fabrics

- Improved preprocessing pipeline for knitted patterns
- Added CLAHE contrast enhancement
- Updated test data with additional examples

Fixes #89
```

## Adding New Features

### Pattern-Based Anomaly Detectors
If adding support for a new fabric pattern:

1. Create a new directory: `Pattern_based_Anomaly_Detect/pattern_name/`
2. Train and save the model
3. Update `ml_models.py` to register the new pattern
4. Add test cases in `tests/test_anomaly_detection.py`
5. Document in wiki

### New Classification Models
If adding a new classifier:

1. Follow the existing model interface in `ml_models.py`
2. Implement `load()`, `predict()`, and `get_confidence()` methods
3. Add model discovery to configuration
4. Update documentation

## Performance Considerations

- Keep inference time < 500ms for real-time applications
- Optimize image preprocessing pipeline
- Use vectorized NumPy/PyTorch operations
- Profile with cProfile for bottlenecks

## Security

- Never commit API keys, passwords, or secrets
- Use environment variables for sensitive data
- Validate and sanitize user inputs
- Keep dependencies updated

## Questions?

- Check [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- Search existing issues
- Check wiki pages
- Create a new discussion or issue

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
