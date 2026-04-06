# AI Writing Agent

<div align="center">

![AI Writing Agent](docs/assets/banner.png)

**Your Personal AI Writing Assistant - Open Source, Privacy-First, Enterprise-Grade**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/ai-writing-agent?style=social)](https://github.com/yourusername/ai-writing-agent/stargazers)

</div>

---

## 🚀 Features

### 🤖 Multi-Model Support
- **OpenAI GPT-4/4o** - Best for versatile, high-quality content
- **Anthropic Claude 3.5/4** - Best for long-form, nuanced writing
- **Ollama (Local)** - Privacy-first, runs entirely on your machine
- **DeepSeek** - Cost-effective, excellent reasoning
- **Google Gemini** - Multimodal capabilities

### ✍️ Specialized Writing Agents

| Agent | Best For | Key Features |
|-------|----------|--------------|
| 🐦 **Social Media** | Twitter, Instagram, Facebook | Engagement optimization, hashtag strategy, viral hooks |
| 📧 **Email** | Cold outreach, follow-ups, newsletters | Personalization, A/B variants, CTA optimization |
| 💼 **LinkedIn** | Professional posts, articles, networking | Thought leadership, industry insights, networking |
| 📝 **Blog** | Articles, how-tos, opinion pieces | SEO optimization, structure, engaging intros |
| 🔬 **Technical** | Research papers, analysis, technical docs | Citations, accuracy, structured explanations |
| 📚 **Documentation** | API docs, user guides, READMEs | Clarity, examples, troubleshooting |

### 🎯 Core Capabilities

- **Multi-Agent Orchestration** - LangGraph-powered intelligent routing
- **Content Optimization** - Grammar, tone, style improvements
- **SEO Enhancement** - Keyword integration, readability scoring
- **Brand Voice Training** - Learn and maintain your unique voice
- **Template Library** - Pre-built prompts for common use cases
- **Export Options** - Markdown, HTML, PDF, plain text
- **History & Versioning** - Track all your content generations

---

## 📦 Quick Start

### Prerequisites

```bash
# Python 3.10+ required
python --version  # or python3 --version

# Optional: Ollama for local models
brew install ollama  # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-writing-agent.git
cd ai-writing-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Configuration

Edit `.env` with your API keys:

```env
# OpenAI (Recommended for best quality)
OPENAI_API_KEY=sk-...

# Anthropic (Best for long-form)
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Ollama (Local, privacy-first)
OLLAMA_BASE_URL=http://localhost:11434
```

### Run the Agent

```bash
# Interactive CLI
python -m src.cli

# Or with specific model
python -m src.cli --model claude

# Web UI (Gradio)
python -m src.ui.gradio_app

# Web UI (Streamlit)
streamlit run src/ui/streamlit_app.py
```

---

## 🏗️ Architecture

```
ai-writing-agent/
├── src/
│   ├── agents/              # Specialized writing agents
│   │   ├── base_agent.py    # Base agent class
│   │   ├── social_media.py  # Social media content
│   │   ├── email.py         # Email copywriting
│   │   ├── linkedin.py      # LinkedIn content
│   │   ├── blog.py          # Blog writing
│   │   ├── technical.py     # Technical writing
│   │   └── documentation.py # Documentation
│   ├── core/
│   │   ├── orchestrator.py  # LangGraph orchestration
│   │   ├── llm_manager.py   # Multi-model support
│   │   └── memory.py        # Conversation memory
│   ├── utils/
│   │   ├── optimizer.py      # Content optimization
│   │   ├── seo.py           # SEO tools
│   │   └── exporter.py      # Export utilities
│   └── cli.py               # CLI interface
├── prompts/                 # Specialized prompt templates
├── config/                  # Configuration files
├── tests/                   # Test suite
└── docs/                    # Documentation
```

---

## 💡 Usage Examples

### CLI Usage

```bash
# Generate a LinkedIn post
python -m src.cli generate linkedin \
  --topic "AI is transforming productivity" \
  --tone professional \
  --length medium

# Write a cold email
python -m src.cli generate email \
  --type cold-outreach \
  --recipient "CTO at TechCorp" \
  --goal "Schedule demo"

# Create social media thread
python -m src.cli generate social \
  --platform twitter \
  --topic "5 tips for remote work" \
  --count 5

# Generate technical documentation
python -m src.cli generate docs \
  --type api \
  --file src/core/orchestrator.py
```

### Python API

```python
from src.core.orchestrator import WritingAgentOrchestrator

# Initialize orchestrator
orchestrator = WritingAgentOrchestrator(
    primary_model="claude",
    secondary_model="gpt-4"
)

# Generate LinkedIn post
result = await orchestrator.generate(
    content_type="linkedin",
    topic="Building in public",
    tone="thought_leader",
    audience="founders and developers"
)

print(result.content)
print(result.metadata)  # engagement预测, hashtags, etc.
```

### Web Interface

```bash
# Launch Gradio UI
python -m src.ui.gradio_app

# Access at http://localhost:7860
```

---

## 🔧 Advanced Features

### Brand Voice Training

```python
from src.utils.brand_voice import BrandVoiceTrainer

trainer = BrandVoiceTrainer()
trainer.train_from_samples([
    "Your sample content here...",
    "More samples...",
])

# Save trained voice
orchestrator.set_brand_voice(trainer.get_voice_profile())
```

### Content Optimization Pipeline

```python
from src.utils.optimizer import ContentOptimizer

optimizer = ContentOptimizer()

# Full optimization
optimized = optimizer.optimize(
    content=raw_content,
    target="readability",  # or "seo", "engagement", "professional"
    grade_level=12
)

# Grammar check only
grammar_fixed = optimizer.check_grammar(content)
```

### SEO Enhancement

```python
from src.utils.seo import SEOEnhancer

enhancer = SEOEnhancer()
optimized = enhancer.optimize(
    content=article,
    keywords=["AI writing", "content creation"],
    target_search_intent="informational"
)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/agents/ -v

# Run with coverage
pytest --cov=src tests/

# Test specific agent
python -m src.cli test --agent linkedin
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Format code
make format

# Lint code
make lint
```

### Add New Agent

```python
# src/agents/my_agent.py
from .base_agent import BaseWritingAgent

class MyAgent(BaseWritingAgent):
    name = "my_agent"
    description = "Description of what this agent does"
    
    async def generate(self, input: dict) -> AgentResponse:
        # Your implementation
        pass
    
    def get_prompt_template(self) -> str:
        return "your-prompt-template"
```

---

## 📊 Performance

| Model | Speed | Quality | Cost | Privacy |
|-------|-------|---------|------|---------|
| GPT-4o | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | ☁️ Cloud |
| Claude 3.5 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | ☁️ Cloud |
| DeepSeek | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | ☁️ Cloud |
| Ollama (7B) | ⚡⚡⚡ | ⭐⭐⭐ | Free | 🔒 Local |
| Ollama (70B) | ⚡⚡ | ⭐⭐⭐⭐ | Free | 🔒 Local |

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.dev) - For the LangGraph framework
- [Anthropic](https://anthropic.com) - Claude models
- [OpenAI](https://openai.com) - GPT models
- [Ollama](https://ollama.com) - Local LLM infrastructure

---

<div align="center">

**Star us on GitHub** ⭐ | **Follow for Updates** 🐦 | **Join our Discord** 💬

</div>
