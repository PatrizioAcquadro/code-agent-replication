# CODEAGENT: A Tool-Integrated Agent for Repository-Level Code Generation

This repository contains a Python implementation of the **CODEAGENT** framework, as described in the paper "[CODEAGENT: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges](https://arxiv.org/abs/2401.07339)".

This project was developed for the "062786 - LARGE LANGUAGE MODELS: APPLICATIONS, OPPORTUNITIES AND RISKS" course at Politecnico di Milano.

## Authors and Course Information

*   **Authors:**
    *   Acquadro Patrizio (`patrizio.acquadro@mail.polimi.it`)
    *   Zheng Maria Yu (`zhengmaria.yu@mail.polimi.it`)
*   **Institution:** Politecnico di Milano
*   **Course:** 062786 - LARGE LANGUAGE MODELS: APPLICATIONS, OPPORTUNITIES AND RISKS (A.Y. 2024-2025)
*   **Professor:** Mark James Carman

## Overview

Real-world software development involves more than just writing standalone functions; it requires understanding existing codebases, reading documentation, and using various tools. This project replicates the core idea of CODEAGENT: empowering a Large Language Model (LLM) with a suite of specialized tools to mimic a human developer's workflow.

The agent uses a **ReAct (Reasoning and Acting)** strategy to decompose problems, select the appropriate tool, and iteratively build a solution based on observations from the environment.

### Key Features

*   **Modular Toolset**: A collection of LangChain-compatible tools for various development tasks:
    *   `DocumentationReadingTool`: Searches and comprehends project documentation.
    *   `CodeSymbolNavigationTool`: Finds and inspects existing functions and classes in the codebase.
    *   `CodeInterpreterTool`: Executes code to test functionality and get runtime feedback.
    *   `FormatCheckerTool`: Ensures code quality using the `black` formatter.
    *   `WebSearcher`: Looks up external information (uses a mock for reproducibility).
*   **Micro-Benchmark Environment**: A self-contained Python repository (`benchmark_workdir`) is created on the fly, providing a consistent environment for the agent to work in.
*   **ReAct Agent Logic**: A clear implementation of the agent's reasoning and tool-use cycle.
*   **Colab Ready**: Designed to run on a free Google Colab T4 GPU instance.

## Setup and Installation

### Google Colab (Recommended)

1.  Open a new Google Colab notebook and connect to a T4 GPU runtime.
2.  Clone the repository:
    ```bash
    !git clone https://github.com/your-username/code-agent-replication.git
    %cd code-agent-replication
    ```
3.  Run the setup script to install all dependencies:
    ```bash
    !bash setup_colab.sh
    ```
4.  Set your Hugging Face token to download the CodeLlama model. You can add it as a secret in Colab with the name `HF_TOKEN`.

### Local Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/code-agent-replication.git
    cd code-agent-replication
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point is `main.py`. You can run the agent on predefined benchmark tasks.

```bash
python main.py --task <task_name>
```
Available tasks are defined in `benchmark/tasks.py`.

### Example: Running Task 1

Task 1 requires the agent to import a function from `utilities.py` and call it in `main_app.py`.

This command will:
1. Set up the benchmark workspace.
2. Load the CodeLlama-7b model.
3. Initialize the ReAct agent with all tools.
4. Run the agent to solve Task 1, showing its thought process.
5. Print the final state of the modified file (`main_app.py`).

```bash
python main.py --task simple_import_and_call
```
The agent's output, including its thoughts and actions, will be printed to the console. The final generated code in `benchmark_workdir/main_app.py` will be displayed at the end.

## Project Structure
- `main.py`: Main script to run the agent.
- `src/code_agent/`: Core Python package for the agent.
- `llm.py`: Handles loading the Hugging Face LLM with 4-bit quantization.
- `agent.py`: Contains the ReAct agent prompt and creation logic.
- `workspace.py`: Manages the file system interactions within the benchmark directory.
- `tools/`: Contains all the individual tool implementations.
- `benchmark/`: Contains the setup logic and task definitions for the micro-benchmark.
- `tests/`: Includes basic unit tests for the tools.

## Reference
This implementation is based on the concepts and architecture described in: Zhang, K., Li, J., Li, G., Shi, X., & Jin, Z. (2024). CODEAGENT: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges. arXiv preprint arXiv:2401.07339.
