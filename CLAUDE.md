# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **CodeAgent**, a replication of the framework described in https://arxiv.org/abs/2401.07339. CodeAgent is an LLM-based agent framework that leverages external tools to assist LLMs in repository-level code generation tasks. Repository-level code generation requires generating code based on all software artifacts in a repository (documentation, code dependencies, runtime environment).

The implementation is contained in a single Jupyter notebook: `CodeAgent_Final.ipynb` (129 cells: 47 markdown, 82 code).

## Architecture

### Core Components

1. **LLM Integration** (Phase 1)
   - Uses Hugging Face transformers with 4-bit quantization (BitsAndBytes)
   - Configurable to work with multiple models via `load_agent_llm` function
   - Supports Code Llama and other HF models
   - Uses LangChain for agent orchestration

2. **Five Programming Tools** (Phase 3)
   - **FormatCheckerTool**: Uses `black` formatter to ensure code style consistency
   - **CodeSymbolNavigationTool**: Uses AST parsing to navigate and search for code symbols (classes, functions, variables)
   - **CodeInterpreterTool**: Executes Python code in isolated temporary environments for testing
   - **DocSearchTool**: Searches documentation using BM25 ranking algorithm
   - **WebSearchTool**: Uses DuckDuckGo for web searches
   - All tools inherit from LangChain's `BaseTool` with Pydantic schemas for input validation

3. **Agent Strategies** (Phase 4)
   - Integration with LangChain agent executors
   - Support for multiple LLM providers:
     - OpenRouter API (Deepseek V3) for large open models
     - OpenAI API (GPT-4) for large closed models
     - Google API (Gemini 2.5 Flash) for large closed models
     - Small models via HuggingFace

4. **Benchmarks** (Phase 2)
   - **CodeAgentBench**: Uses `numpy-ml` repository with 57 tasks (51 class, 6 function generation)
     - Codebase: `all_py_content.jsonl`
     - Tasks: `final_dataset.jsonl`
   - **MiniTransformers**: Smaller benchmark for iterative development (22 files, 15 tasks)
     - Codebase stored in `miniformer_codebase` DataFrame
     - Tasks stored in `miniformer_tasks` DataFrame
     - Repository reconstructed on filesystem for tool interaction

5. **Evaluation** (Phase 5)
   - CodeAgentBench evaluation: Tests agent with repository code files, documentation, and tools
   - HumanEval: Function-level code generation benchmark
   - Ablation studies: Analyzes individual tool contributions

## Working with the Notebook

### Running the Notebook

The notebook is designed for Google Colab but can be adapted for local execution:

1. **Phase 0 (Setup)**: Mount Google Drive (for Colab) or configure local project path
2. **Install dependencies** (see below)
3. **Run phases sequentially**: Each phase depends on previous phases being executed

### Key Dependencies

```
transformers==4.52.4
pandas==2.2.2
accelerate==1.7.0
bitsandbytes==0.46.0
langchain==0.3.25
langchain-huggingface==0.2.0
human_eval==1.0.3
sentencepiece==0.2.0
rank_bm25==0.2.2
tree_sitter==0.21.3
black==24.4.2
torch==2.6.0
langchain-openai
google-generativeai
langchain_google_genai
langchain-community
duckduckgo-search
pytest
```

Install with: `pip install -q <package>`

### Critical Helper Functions (Phase 0.5)

The notebook defines helper functions for safe file operations within a simulated repository environment. These are fundamental utilities used throughout the project for:
- Safe file reading/writing within project boundaries
- Path validation and manipulation
- Repository reconstruction from DataFrame format

### Model Configuration

The `load_agent_llm` function handles model loading with:
- Model-specific tokenizer configuration
- 4-bit quantization settings via BitsAndBytesConfig
- Pipeline creation for LangChain integration
- Support for any Hugging Face model ID

### Testing Code Generation

Phase 1.3 provides a basic LLM test framework that verifies functionality using direct code generation (without the full agentic loop). This is useful for:
- Validating model loading
- Testing prompt formatting
- Quick iteration without agent overhead

## Tool Development

When modifying or adding tools:

1. **Inherit from `BaseTool`**: All tools must subclass LangChain's `BaseTool`
2. **Define Pydantic input schema**: Use `BaseModel` and `Field` for type safety
3. **Set descriptive attributes**:
   - `name`: Short identifier
   - `description`: Detailed explanation for LLM (crucial for tool selection)
   - `args_schema`: Pydantic model class
4. **Implement `_run` method**: Core tool logic with proper error handling
5. **Test in isolation**: Verify tool works before integrating into agent

Example structure (FormatCheckerTool):
- Input validation via Pydantic
- Uses subprocess/library call for actual work
- Returns formatted string or error message
- Clear description helps LLM know when to use it

## Benchmark Data Structure

Both benchmarks use similar structures:

**Codebase files** (`.jsonl`):
- Each line: `{"file_path": "...", "file_content": "..."}`
- Loaded into pandas DataFrame for analysis
- Reconstructed to filesystem for tool interaction

**Task files** (`.jsonl`):
- Task definitions with requirements, signatures, metadata
- Average instruction length: ~340 words (CodeAgentBench), ~98 words (MiniTransformers)
- Mix of class and function generation tasks

**Repository Reconstruction**:
- DataFrame format → Physical file system
- Required for tools that operate on actual files
- Uses helper functions to ensure safe writes within project boundaries

## Evaluation Workflow

1. Load benchmark (CodeAgentBench or MiniTransformers)
2. Reconstruct repository on filesystem
3. For each task:
   - Provide agent with code files, documentation, tools
   - Agent generates code using tools
   - Evaluate output against reference implementation
4. Track tool usage for ablation studies

## Important Notes

- **Memory constraints**: Large files in CodeAgentBench (up to ~9,000 lines) cannot be processed entirely by LLM context windows - this is why tools are essential
- **File complexity**: `layers.py` in numpy-ml contains 22 classes and 166 functions
- **Task complexity**: Instructions are heavily structured with detailed API documentation, parameter types, default values
- **Reproducibility**: Random seed is fixed in Phase 0.4
- **Quantization**: 4-bit quantization is critical for running large models in resource-constrained environments
