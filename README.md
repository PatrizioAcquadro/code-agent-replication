# CodeAgent: Tool-Integrated Agent Systems for Repository-Level Code Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://langchain.com/)

A Python implementation replicating the **CodeAgent** framework from the paper *"CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges"* (Zhang et al., 2024).

This project was developed as part of the course **062786 - Large Language Models: Applications, Opportunities and Risks** (A.Y. 2024-2025) at **Politecnico di Milano**, under the supervision of **Prof. Mark James Carman**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Running Evaluations](#running-evaluations)
  - [Using Individual Tools](#using-individual-tools)
  - [Configuring LLM Providers](#configuring-llm-providers)
- [Benchmarks](#benchmarks)
- [Replicating Results](#replicating-results)
- [Configuration](#configuration)
- [Testing](#testing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [License](#license)

---

## Overview

**Repository-level code generation** is a challenging task that goes beyond simple function completion. It requires understanding the entire software ecosystem, including:

- Documentation and README files
- Code dependencies and import structures
- Runtime environment and testing infrastructure
- Existing code patterns and conventions

Traditional LLMs struggle with this task due to context window limitations and lack of interaction with the codebase. **CodeAgent** addresses these challenges by augmenting LLMs with specialized programming tools, enabling them to:

1. **Navigate** code symbols and understand project structure
2. **Search** documentation and external resources
3. **Execute** code to verify correctness
4. **Format** code according to project standards

This implementation provides a modular, extensible framework for experimenting with tool-augmented code generation agents.

---

## Key Features

- **Five Programming Tools**: FormatCheck, CodeSymbolNavigation, CodeInterpreter, DocSearch, and WebSearch
- **Multiple LLM Providers**: Support for OpenAI, Google Gemini, DeepSeek (via OpenRouter), and HuggingFace models
- **Two Agent Strategies**: ReAct (reasoning + acting) and native tool-calling
- **Two Benchmarks**: MiniTransformers (15 tasks) for development and CodeAgentBench (57 tasks) for full evaluation
- **Comprehensive Evaluation Pipeline**: Automated testing with pytest verification
- **Baseline Comparison**: No-agent baseline for measuring tool effectiveness
- **HumanEval Integration**: Function-level code generation evaluation
- **Modular Architecture**: Clean separation of concerns for easy extension

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CodeAgent Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐   │
│  │   LLM    │◄──►│              Agent Executor              │   │
│  │ Provider │    │  (ReAct / Tool-Calling Strategy)        │   │
│  └──────────┘    └─────────────────┬───────────────────────┘   │
│                                    │                            │
│                    ┌───────────────┴───────────────┐           │
│                    ▼                               ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Programming Tools                     │   │
│  ├─────────────┬─────────────┬─────────────┬───────────────┤   │
│  │ FormatCheck │ CodeSymbol  │    Code     │   DocSearch   │   │
│  │   (black)   │ Navigation  │ Interpreter │    (BM25)     │   │
│  │             │(tree-sitter)│ (subprocess)│               │   │
│  ├─────────────┴─────────────┴─────────────┴───────────────┤   │
│  │                      WebSearch (DuckDuckGo)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                    │                            │
│                    ┌───────────────┴───────────────┐           │
│                    ▼                               ▼            │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │      Repository      │    │     Evaluation Pipeline      │  │
│  │   (Reconstructed)    │    │   (pytest verification)      │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Descriptions

| Tool | Purpose | Technology |
|------|---------|------------|
| **FormatCheck** | Validates and formats Python code | `black` formatter |
| **CodeSymbolNavigation** | Searches and navigates code symbols (classes, functions, variables) | `tree-sitter` AST parsing |
| **CodeInterpreter** | Executes Python code in isolated environment | `subprocess` with timeout |
| **DocSearch** | Searches project documentation | BM25 ranking algorithm |
| **WebSearch** | Searches the web for external information | DuckDuckGo API |

---

## Project Structure

```
codeagent/
├── pyproject.toml                 # Package configuration and dependencies
├── requirements.txt               # Pip-compatible dependencies
├── README.md                      # This file
├── CLAUDE.md                      # Development documentation
├── CodeAgent_Final.ipynb          # Thin orchestrator notebook
│
├── src/codeagent/                 # Main package
│   ├── __init__.py                # Package exports
│   │
│   ├── config/                    # Configuration management
│   │   ├── settings.py            # ProjectConfig, LLMConfig dataclasses
│   │   ├── secrets.py             # API key management
│   │   └── quantization.py        # HuggingFace quantization config
│   │
│   ├── llm/                       # LLM providers
│   │   ├── factory.py             # Unified create_llm() factory
│   │   ├── huggingface.py         # HuggingFace models with quantization
│   │   ├── openai_provider.py     # OpenAI and OpenRouter
│   │   └── gemini_provider.py     # Google Gemini
│   │
│   ├── tools/                     # Programming tools
│   │   ├── format_checker.py      # FormatCheckTool
│   │   ├── symbol_navigation.py   # CodeSymbolNavigationTool
│   │   ├── code_interpreter.py    # CodeInterpreterTool
│   │   ├── doc_search.py          # DocSearchTool
│   │   ├── web_search.py          # WebsiteSearchTool
│   │   └── summarizer.py          # Summarization utilities
│   │
│   ├── benchmarks/                # Benchmark datasets
│   │   ├── base.py                # Abstract Benchmark class
│   │   ├── mini_transformers.py   # MiniTransformers (15 tasks)
│   │   ├── codeagent_bench.py     # CodeAgentBench (57 tasks)
│   │   ├── loader.py              # JSONL loading utilities
│   │   └── analysis.py            # Code analysis functions
│   │
│   ├── agents/                    # Agent creation
│   │   ├── factory.py             # create_agent_executor()
│   │   └── prompts.py             # ReAct and tool-calling prompts
│   │
│   ├── evaluation/                # Evaluation infrastructure
│   │   ├── pipeline.py            # run_evaluation_pipeline()
│   │   ├── task_runner.py         # Single task evaluation
│   │   ├── repository_setup.py    # Repository reconstruction
│   │   ├── metrics.py             # Pass rate, reporting
│   │   ├── no_agent_baseline.py   # Baseline without tools
│   │   └── human_eval.py          # HumanEval benchmark
│   │
│   └── utils/                     # Utilities
│       ├── file_ops.py            # Safe file operations
│       ├── code_cleaning.py       # Code extraction from LLM output
│       ├── seed.py                # Random seed management
│       └── api_guide.py           # API documentation generation
│
└── tests/                         # Test suite
    ├── conftest.py                # Shared fixtures
    ├── test_utils.py              # Utility tests
    ├── test_benchmarks.py         # Benchmark tests
    ├── test_agents.py             # Agent tests
    ├── test_evaluation.py         # Evaluation tests
    └── tools/                     # Tool-specific tests
        ├── test_format_checker.py
        ├── test_code_interpreter.py
        ├── test_doc_search.py
        └── test_symbol_navigation.py
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for HuggingFace models

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/codeagent.git
   cd codeagent
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install as a package:
   ```bash
   pip install -e .
   ```

4. **Configure API keys**

   Set environment variables for your chosen LLM provider:
   ```bash
   # For Google Gemini
   export GOOGLE_API_KEY="your-gemini-api-key"

   # For OpenAI
   export OPENAI_API_KEY="your-openai-api-key"

   # For DeepSeek via OpenRouter
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   ```

5. **Install tree-sitter Python grammar** (for CodeSymbolNavigation)
   ```bash
   pip install tree-sitter-python
   ```

---

## Quick Start

### Using the Notebook

The easiest way to get started is with the orchestrator notebook:

```bash
jupyter notebook CodeAgent_Final.ipynb
```

### Using Python Scripts

```python
import sys
sys.path.insert(0, "./src")

from pathlib import Path
from codeagent import CodeAgentConfig, fix_random_seeds, create_llm, get_all_tools
from codeagent.agents import create_agent_executor
from codeagent.benchmarks import MiniTransformersBench
from codeagent.evaluation import run_evaluation_pipeline

# Setup
config = CodeAgentConfig(project_repo_path=Path("./repo"), random_seed=42)
fix_random_seeds(config.random_seed)

# Load LLM
llm, ready = create_llm("gemini")

# Load benchmark
benchmark = MiniTransformersBench()
codebase_df = benchmark.load_codebase()
tasks_df = benchmark.load_tasks()

# Setup tools and agent
tools = get_all_tools(config.project_repo_path)
agent = create_agent_executor(llm, tools, strategy="react")

# Run evaluation
results = run_evaluation_pipeline(
    agent_executor=agent,
    codebase_df=codebase_df,
    task_df=tasks_df,
    project_repo_path=config.project_repo_path,
)

# Results
from codeagent.evaluation import calculate_pass_rate
print(f"Pass@1 Rate: {calculate_pass_rate(results):.2%}")
```

---

## Usage Guide

### Running Evaluations

#### Full Benchmark Evaluation
```python
from codeagent.evaluation import run_evaluation_pipeline

results = run_evaluation_pipeline(
    agent_executor=agent,
    codebase_df=codebase_df,
    task_df=tasks_df,
    project_repo_path=config.project_repo_path,
    task_ids=None,  # Run all tasks
    delay_between_tasks=2.0,  # Rate limiting
)
```

#### Specific Tasks Only
```python
results = run_evaluation_pipeline(
    agent_executor=agent,
    codebase_df=codebase_df,
    task_df=tasks_df,
    project_repo_path=config.project_repo_path,
    task_ids=["miniformer-01", "miniformer-02", "miniformer-03"],
)
```

#### Resume from a Task
```python
results = run_evaluation_pipeline(
    agent_executor=agent,
    codebase_df=codebase_df,
    task_df=tasks_df,
    project_repo_path=config.project_repo_path,
    start_from_task="miniformer-05",  # Resume from here
)
```

### Using Individual Tools

```python
from codeagent.tools import (
    FormatCheckTool,
    CodeSymbolNavigationTool,
    CodeInterpreterTool,
    DocSearchTool,
)

# Format checking
formatter = FormatCheckTool()
result = formatter._run("def foo():return 42")
print(result)  # Formatted code

# Code navigation
navigator = CodeSymbolNavigationTool(project_path=Path("./my_project"))
result = navigator._run("MyClass")
print(result)  # Class definition and location

# Code execution
interpreter = CodeInterpreterTool(project_path=Path("./my_project"))
result = interpreter._run("print(2 + 2)")
print(result)  # "4"

# Documentation search
doc_search = DocSearchTool(project_path=Path("./my_project"))
result = doc_search._run("authentication API")
print(result)  # Relevant documentation snippets
```

### Configuring LLM Providers

#### Google Gemini (Recommended)
```python
llm, ready = create_llm("gemini", model_id="gemini-2.5-flash")
```

#### OpenAI
```python
llm, ready = create_llm("openai", model_id="gpt-4")
```

#### DeepSeek via OpenRouter
```python
llm, ready = create_llm("deepseek", model_id="deepseek/deepseek-chat")
```

#### HuggingFace (Local, with Quantization)
```python
llm, ready = create_llm("huggingface", model_id="codellama/CodeLlama-7b-hf")
```

---

## Benchmarks

### MiniTransformers Benchmark

A lightweight benchmark designed for iterative development:

| Metric | Value |
|--------|-------|
| Source Files | 22 |
| Total Tasks | 15 |
| Task Types | Additive, Fix, Refactoring |
| Avg. Instruction Length | ~98 words |

**Task Categories:**
- **Additive**: Add new functionality (e.g., bias field, activation functions)
- **Fix**: Correct existing code (e.g., bug fixes)
- **Refactoring**: Restructure code (e.g., extract methods)

### CodeAgentBench

The full benchmark from the numpy-ml repository:

| Metric | Value |
|--------|-------|
| Total Tasks | 57 |
| Class Generation | 51 |
| Function Generation | 6 |
| Avg. Instruction Length | ~340 words |
| Max File Size | ~9,000 lines |

---

## Replicating Results

To replicate the evaluation results from the original paper:

### 1. Setup Environment
```bash
git clone https://github.com/yourusername/codeagent.git
cd codeagent
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"  # Or other provider
```

### 2. Run Full Evaluation
```python
from pathlib import Path
from codeagent import CodeAgentConfig, fix_random_seeds, create_llm, get_all_tools
from codeagent.agents import create_agent_executor
from codeagent.benchmarks import MiniTransformersBench
from codeagent.evaluation import run_evaluation_pipeline, run_no_agent_baseline, compare_results

# Configuration
config = CodeAgentConfig(project_repo_path=Path("./mini_transformers_repo"), random_seed=42)
fix_random_seeds(42)

# Load components
llm, _ = create_llm("gemini")
benchmark = MiniTransformersBench()
codebase_df = benchmark.load_codebase()
tasks_df = benchmark.load_tasks()
tools = get_all_tools(config.project_repo_path)

# Create agent and run
agent = create_agent_executor(llm, tools, strategy="react")
agent_results = run_evaluation_pipeline(
    agent_executor=agent,
    codebase_df=codebase_df,
    task_df=tasks_df,
    project_repo_path=config.project_repo_path,
)

# Run baseline for comparison
baseline_results = run_no_agent_baseline(
    llm_instance=llm,
    codebase_df=codebase_df,
    task_df=tasks_df,
    project_repo_path=config.project_repo_path,
)

# Compare results
comparison = compare_results(agent_results, baseline_results)
```

### 3. Expected Results

The agent with tools should significantly outperform the no-agent baseline, demonstrating the value of tool-augmented code generation.

---

## Configuration

### ProjectConfig

```python
from codeagent import CodeAgentConfig

config = CodeAgentConfig(
    project_repo_path=Path("./repository"),  # Where to reconstruct the repo
    random_seed=42,                           # For reproducibility
)
```

### LLMConfig

```python
from codeagent.config import LLMConfig

llm_config = LLMConfig(
    provider="gemini",
    model_id="gemini-2.5-flash",
    temperature=0.0,  # Deterministic for evaluation
    max_tokens=4096,
)
```

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `GOOGLE_API_KEY` | Google AI API key | Gemini |
| `OPENAI_API_KEY` | OpenAI API key | OpenAI, GPT-4 |
| `OPENROUTER_API_KEY` | OpenRouter API key | DeepSeek |
| `HUGGINGFACE_TOKEN` | HuggingFace token | Gated models |

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_utils.py

# Run with coverage
pytest tests/ --cov=src/codeagent
```

---

## Authors

This project was developed by:

- **Patrizio Acquadro** - [patrizio.acquadro@mail.polimi.it](mailto:patrizio.acquadro@mail.polimi.it)
- **Zheng Maria Yu** - [zhengmaria.yu@mail.polimi.it](mailto:zhengmaria.yu@mail.polimi.it)

Master's Students at **Politecnico di Milano**

---

## Acknowledgments

- **Prof. Mark James Carman** - Course Instructor, Politecnico di Milano
- **Course**: 062786 - Large Language Models: Applications, Opportunities and Risks (A.Y. 2024-2025)
- The authors of the original CodeAgent paper for their innovative framework design
- The LangChain team for the agent framework infrastructure
- The tree-sitter team for the powerful AST parsing library

---

## References

### Primary Reference

Zhang, K., Li, J., Li, G., Shi, X., & Jin, Z. (2024). **CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges**. *arXiv preprint arXiv:2401.07339*.

```bibtex
@article{zhang2024codeagent,
  title={CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges},
  author={Zhang, Kechi and Li, Jia and Li, Ge and Shi, Xianjie and Jin, Zhi},
  journal={arXiv preprint arXiv:2401.07339},
  year={2024}
}
```

### Additional References

- Chen, M., et al. (2021). **Evaluating Large Language Models Trained on Code**. *arXiv preprint arXiv:2107.03374*. (HumanEval benchmark)
- Yao, S., et al. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models**. *arXiv preprint arXiv:2210.03629*. (ReAct agent strategy)
- Wei, J., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. *NeurIPS 2022*. (Chain-of-thought reasoning)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Developed at Politecnico di Milano, 2024-2025</i>
</p>
