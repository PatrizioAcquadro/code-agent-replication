"""API documentation generation utilities.

This module provides functions to generate markdown API documentation
from Python source files using tree-sitter parsing.
"""

import os
from pathlib import Path
from typing import Union, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Parser


def generate_api_guide(
    repo_path: Union[str, Path],
    parser: "Parser",
    core_lib_name: str = "miniformer",
) -> str:
    """Generate markdown API documentation for a Python library.

    Walks through the source files in the library directory and extracts
    class and function signatures along with their docstrings.

    Args:
        repo_path: Path to the repository root.
        parser: A configured tree-sitter Parser instance.
        core_lib_name: Name of the core library directory to document.

    Returns:
        Markdown-formatted API documentation string.

    Example:
        >>> from tree_sitter import Parser, Language
        >>> # Setup parser...
        >>> docs = generate_api_guide("/path/to/repo", parser, "mylib")
        >>> with open("api_guide.md", "w") as f:
        ...     f.write(docs)
    """
    repo_path = Path(repo_path)
    api_guide_content = f"# {core_lib_name.title()} API Guide\n\n"
    api_guide_content += "This guide details the core library components.\n"

    core_lib_path = repo_path / core_lib_name

    if not core_lib_path.exists():
        return f"# API Guide\n\nError: Library path '{core_lib_path}' not found."

    for root, _, files in os.walk(core_lib_path):
        for file in sorted(files):
            if file.endswith(".py") and "__init__" not in file:
                rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                api_guide_content += f"\n---\n\n## File: `{rel_path}`\n"

                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    content_bytes = content.encode("utf-8")
                    tree = parser.parse(content_bytes)

                    api_guide_content += _extract_symbols_from_tree(
                        tree, content_bytes
                    )
                except Exception as e:
                    api_guide_content += f"\n*Error parsing file: {e}*\n"

    return api_guide_content


def _extract_symbols_from_tree(tree: Any, code_bytes: bytes) -> str:
    """Extract function and class definitions from a parsed tree.

    Args:
        tree: The tree-sitter parse tree.
        code_bytes: The source code as bytes.

    Returns:
        Markdown-formatted documentation for the symbols.
    """
    result = ""

    for node in tree.root_node.children:
        if node.type in ("function_definition", "class_definition"):
            # Get the name
            name_node = next(
                (n for n in node.children if n.type == "identifier"),
                None,
            )
            if not name_node:
                continue

            name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")

            # Get the parameters/signature
            params_node = next(
                (n for n in node.children if n.type == "parameters"),
                None,
            )
            signature = (
                code_bytes[params_node.start_byte:params_node.end_byte].decode("utf-8")
                if params_node
                else "()"
            )

            result += f"\n### `{name}{signature}`\n"

            # Get the docstring if present
            body_node = next(
                (n for n in node.children if n.type == "block"),
                None,
            )
            if body_node and body_node.children:
                first_stmt = body_node.children[0]
                if first_stmt.type == "expression_statement" and first_stmt.children:
                    docstring_node = first_stmt.children[0]
                    if docstring_node.type == "string":
                        docstring = (
                            code_bytes[docstring_node.start_byte:docstring_node.end_byte]
                            .decode("utf-8")
                            .strip()
                            .strip('"""')
                            .strip("'''")
                            .strip()
                        )
                        result += f"{docstring}\n"

    return result


def create_parser() -> Optional["Parser"]:
    """Create and configure a tree-sitter Python parser.

    Downloads and compiles the Python grammar if needed.

    Returns:
        A configured Parser instance, or None if setup fails.
    """
    try:
        import subprocess
        from tree_sitter import Language, Parser

        grammar_path = Path("./grammars")
        lib_path = grammar_path / "languages.so"

        # Clone grammar if needed
        python_grammar_path = grammar_path / "python"
        if not python_grammar_path.exists():
            grammar_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
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
        parser = Parser()
        parser.set_language(python_language)

        return parser

    except Exception as e:
        print(f"Error creating tree-sitter parser: {e}")
        return None
