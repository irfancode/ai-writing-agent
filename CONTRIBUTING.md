# Contributing to AI Writing Agent

Thank you for your interest in contributing to the AI Writing Agent!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ai-writing-agent.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install dev dependencies: `pip install -r requirements.txt`
6. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Run tests: `pytest tests/`
4. Format code: `make format`
5. Lint code: `make lint`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep lines under 100 characters

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage

## Reporting Issues

- Use the GitHub Issues tab
- Include a minimal reproduction case
- Specify your environment (OS, Python version, etc.)

## Questions?

Feel free to open an issue for any questions or discussions.
