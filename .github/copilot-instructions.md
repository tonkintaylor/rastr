# Custom Instructions

## General Guidelines

- Every time you need to fix linter errors and provide the error messages, update the Linter section in `.github/copilot-instructions.md` accordingly. Use concise, oneliner instruction. Ensure your future responses avoid repeating the same errors.
- Never create notebooks (ipynb files) unless asked explicitly.

## Testing  

- Write unit tests using `pytest` inside `tests/`, structured based on `src/`.  
  - Example: `src/x/y/z` → `tests/x/y/test_z.py`  
- Use test fixtures and group tests into classes when appropriate.  
- When testing methods on classes, create nested test classes within the test class associated with the overall tested class (e.g., `TestContour` nested within `TestRaster` when testing the `contour` method of `Raster`).
- After modifying a function, run its unit tests using pytest.
- Run tests in virtual environment: `.\.venv\Scripts\activate; python -m pytest tests/path/to/test_file.py -v`

## Python

- We use Python 3.12, so ensure that the code is compatible and up to date with this version.
- When adding a new package that requires installation, add it using `uv add <pkg>` so it gets properly declared.
- When running a CLI such as `pytest`, always run it using `uv run`, e.g. `uv run pytest`.
- Always run `uv run pre-commit run --all-files` to determine whether your code is compliant with the pre-commit hooks, including linters.
- When importing internal modules, do not include the "src" folder in the import path as it is already defined in pyproject.toml
- Limit line length to 100 characters.
- Never create functions that return more than one output value (i.e. Never return tuples). Use dedicated return classes.
- Do not add exceptions to functions unless explicitly requested.
- Prefer to type hint strictly with the likes of `Literal["a", "b"]` instead of hinting broader types like `str`. This means the constraints on the input arguments to a function can reside in the type annotation rather than the docstring. Consider @validate_call (from pydantic import validate_call) to avoid boilerplate case-checking in such cases.
- Prefer `isinstance` rather than `hasattr` for runtime type inference.
- Refrain from backslash unescaping in raw strings (e.g., `r"\\path"` should be `r"\path"`).
- When writing scripts, always use "Scripting Style" (Top-Level Code) unless stated otherwise. Write code directly at the module level instead of wrapping in functions or `if __name__ == "__main__":` blocks.
- For scripts that need to access package files (e.g., templates, data files):

  ```python
  from importlib.resources import files

  # Define package path as a module-level constant
  PACKAGE_PATH = files('package.subpackage.module')

  # Use joinpath for accessing files
  file_path = Path(str(PACKAGE_PATH.joinpath('filename')))
  ```

  This ensures consistent path resolution in both interactive and script modes.

## Documenting Functions

- Remove dtype specifications from all `Args:` sections (e.g., `text (str):` → `text:`)
- Use "Args:" instead of "Parameters:" for consistency

## Linter

- For file-level linter suppressions, use `# ruff: noqa: RULE1, RULE2` format (not `# ruff noqa:`). Avoid these where possible.
- Use `pathlib.Path` for all filesystem operations instead of `os.path`. Path objects provide a more readable and maintainable object-oriented interface (e.g., `Path('dir') / 'file.txt'` instead of `os.path.join()`, `path.exists()` instead of `os.path.exists()`, etc.)
- Remove trailing whitespace from blank lines (W293)
- Move statements after try blocks into else blocks when the statements depend on the try block's success (TRY300)
- Use keyword arguments for boolean parameters (FBT003) instead of positional arguments
- Make boolean default arguments keyword-only using `*` to prevent positional passing (FBT002)
- Avoid variable names that shadow Python builtins (A001)
- Use snake_case for all variables in global scope (N816), not camelCase or mixedCase
- Remove unused code instead of commenting it out (ERA001)
- Break long docstring lines at logical points to stay under the 88 character limit (E501)
- Include a blank line between docstring summary and description (D205)
- Add type annotations to function signatures (ANN201, ANN204)
- Use `dict` instead of `Dict` for type annotation (UP006)
- Avoid importing deprecated types like `typing.Dict` (UP035)
- Use `NDArray` without specific dtype parameter (e.g., `NDArray` instead of `NDArray[np.number]`) for generic array parameters

## Testing

- Use `uv run pytest` for testing, and ensure all tests are passing before committing changes.
- Do not perform equality checks with floating point values; instead, use `pytest.approx`.
- Use only one assert statement per test function to ensure clarity and simplicity.

## Documentation & Workflow Management

- Document reusable knowledge (e.g., library versions, fixes, corrections) in the `Lessons` section of `.github/scratchpad.md`.  
- Use `.github/scratchpad.md` to organise tasks:  
  - Clear old tasks when starting a new one.  
  - Plan steps and track progress using TODO markers:  
    - [X] Task 1  
    - [ ] Task 2  
  - Update task progress, especially after milestones.