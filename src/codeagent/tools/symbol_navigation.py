"""CodeSymbolNavigationTool: AST-based code navigation using tree-sitter.

This tool inspects Python source files to list symbols (classes, functions)
and retrieve their source code.
"""

from pathlib import Path
from typing import Type, Optional, Any, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SymbolNavigationInput(BaseModel):
    """Input schema for the CodeSymbolNavigationTool."""

    query: str = Field(
        ...,
        description=(
            "A file system path, with an optional symbol name separated by a comma. "
            "Example: 'src/config.py' to list symbols, or 'src/config.py, MyClass' "
            "to get source code for MyClass."
        ),
    )


class TreeSitterParser:
    """Singleton manager for tree-sitter Python parser.

    This class handles lazy initialization of the tree-sitter parser
    to avoid initialization overhead when the tool is not used.
    """

    _instance: Optional["TreeSitterParser"] = None
    _parser: Optional[Any] = None
    _ready: bool = False

    @classmethod
    def get_instance(cls) -> "TreeSitterParser":
        """Get the singleton parser instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the parser (only called once)."""
        if TreeSitterParser._parser is not None:
            return

        try:
            self._setup_parser()
        except Exception as e:
            print(f"Warning: Tree-sitter parser initialization failed: {e}")
            TreeSitterParser._ready = False

    def _setup_parser(self) -> None:
        """Set up the tree-sitter parser."""
        import subprocess
        from pathlib import Path
        from tree_sitter import Language, Parser

        grammar_path = Path("./grammars")
        lib_path = grammar_path / "languages.so"

        # Clone grammar if needed
        python_grammar_path = grammar_path / "python"
        if not python_grammar_path.exists():
            grammar_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/tree-sitter/tree-sitter-python.git",
                    str(python_grammar_path),
                ],
                check=True,
                capture_output=True,
            )

        # Build the language library
        Language.build_library(str(lib_path), [str(python_grammar_path)])

        # Create and configure parser
        python_language = Language(str(lib_path), "python")
        TreeSitterParser._parser = Parser()
        TreeSitterParser._parser.set_language(python_language)
        TreeSitterParser._ready = True

    @property
    def is_ready(self) -> bool:
        """Check if the parser is ready."""
        return TreeSitterParser._ready

    def parse(self, code: bytes) -> Any:
        """Parse Python code.

        Args:
            code: Python source code as bytes.

        Returns:
            The parsed tree.

        Raises:
            RuntimeError: If parser is not initialized.
        """
        if not self.is_ready or TreeSitterParser._parser is None:
            raise RuntimeError("Tree-sitter parser is not initialized.")
        return TreeSitterParser._parser.parse(code)


class CodeSymbolNavigationTool(BaseTool):
    """A tool to inspect Python source files using tree-sitter.

    This tool has two modes:
    1. List Symbols: Provide only a file path to get a summary of all
       classes and functions in that file.
    2. Get Source Code: Provide a file path AND a symbol name (comma-separated)
       to retrieve the full source code for that specific class or function.

    Attributes:
        name: Tool name for agent reference.
        description: Detailed description for the agent.
        args_schema: Pydantic input validation schema.
        project_path: Base path for the project repository.
    """

    name: str = "CodeSymbolNavigation"
    description: str = (
        "Inspects the contents of a Python source file using its file system path. "
        "**CRITICAL: The input MUST be a file system path with slashes (`/`), "
        "not a Python import path with dots (`.`).** "
        "For example, to inspect the `TransformerConfig` class, the correct input is "
        "`'miniformer/config.py, TransformerConfig'`, NOT `'miniformer.config.TransformerConfig'`.\n"
        "It has two modes:\n"
        "1. **List Symbols:** Provide only a file path (e.g., `'miniformer/config.py'`) "
        "to get a summary of all classes and functions within that file.\n"
        "2. **Get Source Code:** Provide a file path AND a symbol name, separated by a comma "
        "(e.g., `'miniformer/config.py, TransformerConfig'`), to retrieve the full source code."
    )
    args_schema: Type[BaseModel] = SymbolNavigationInput
    project_path: Path = Path(".")

    def __init__(self, project_path: Union[str, Path] = ".", **kwargs: Any) -> None:
        """Initialize the tool with a project path.

        Args:
            project_path: The base path for the project repository.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.project_path = Path(project_path)

    def _get_node_text(self, node: Any, code_bytes: bytes) -> str:
        """Extract text from a tree-sitter node."""
        return code_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def _find_symbol_node(
        self, start_node: Any, symbol_name: str, code_bytes: bytes
    ) -> Optional[Any]:
        """Find a specific symbol node in the tree."""
        for child in start_node.children:
            if child.type in ("function_definition", "class_definition"):
                name_node = next(
                    (n for n in child.children if n.type == "identifier"),
                    None,
                )
                if name_node and self._get_node_text(name_node, code_bytes).strip() == symbol_name.strip():
                    return child
        return None

    def _list_top_level_symbols(self, tree: Any, code_bytes: bytes) -> str:
        """List all top-level classes and functions in the tree."""
        symbols = []
        for node in tree.root_node.children:
            if node.type in ("function_definition", "class_definition"):
                name_node = next(
                    (child for child in node.children if child.type == "identifier"),
                    None,
                )
                if name_node:
                    symbol_type = "Class" if node.type == "class_definition" else "Function"
                    symbols.append(f"- {self._get_node_text(name_node, code_bytes)} ({symbol_type})")

        if not symbols:
            return "No top-level classes or functions found."
        return "Found the following top-level symbols:\n" + "\n".join(symbols)

    def _run(self, query: str) -> str:
        """Execute the code navigation query.

        Args:
            query: A file path, optionally followed by a comma and symbol name.

        Returns:
            Either a list of symbols or the source code of a specific symbol.
        """
        # Get the parser instance
        try:
            parser_instance = TreeSitterParser.get_instance()
            if not parser_instance.is_ready:
                return "Error: Tree-sitter parser is not initialized."
        except Exception as e:
            return f"Error: Could not initialize tree-sitter parser: {e}"

        # Parse the query
        parts = [p.strip() for p in query.split(",")]
        file_path = parts[0].strip()
        symbol_name = parts[1].strip() if len(parts) > 1 else None

        # Read the file
        from ..utils.file_ops import read_file_content

        file_content = read_file_content(file_path, self.project_path)
        if file_content.startswith("Error:"):
            return file_content

        try:
            code_bytes = file_content.encode("utf-8")
            tree = parser_instance.parse(code_bytes)

            if symbol_name:
                node = self._find_symbol_node(tree.root_node, symbol_name, code_bytes)
                if node:
                    return self._get_node_text(node, code_bytes)
                else:
                    return f"Error: Symbol '{symbol_name}' not found in '{file_path}'."
            else:
                return self._list_top_level_symbols(tree, code_bytes)

        except Exception as e:
            return f"Error during code navigation for '{file_path}': {str(e)}"
