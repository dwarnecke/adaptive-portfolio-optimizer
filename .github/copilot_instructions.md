# GitHub Copilot Instructions for This Repository

This repo uses **reStructuredText (rST) docstrings** for all Python code and enforces a strict file layout.
Copilot must follow these rules when generating or completing code.

---

## Top‑of‑file Order (no exceptions)
Copilot **must not** place any class/function before these items.

1. **Author/Email header (always first)** (use existing info if present)
   ```python
   __author__ = "Dylan Warnecke"
   __email__ = "dylan.warnecke@gmail.com"
   ```
2. **Imports** (grouped stdlib → third‑party → local; blank line between groups)
3. **Module‑level constants / type aliases / config**
4. *(Optional)* **Module rST docstring** describing the module
5. **Classes** (public API first, then private)
6. **Functions**
   - Helper/private functions (prefixed `_`) close to where they are used
   - Public functions after classes and organized by feature/section
7. `if __name__ == "__main__":` block (scripts only)

**Never** hoist functions or classes **above** the author/email header, imports, or constants.  
**Never** start a file with a function/class definition.

---

## Docstrings: rST Only
- Triple double quotes `"""`.
- Start with a short imperative summary, then a blank line.
- Use only `:param <name>:` and `:returns:` fields (no `:type:`, `:rtype:`, or `:raises:`).
- **Parameter and return types must be included in the function signature, not in the docstring.**

### Function template
```python
def foo(a: int, b: int) -> int:
    """
    <what/why>.
    :param a: <desc>
    :param b: <desc>
    :returns: <desc>
    """
```

### Method template
```python
class Thing:
    def bar(self, x: float) -> float:
        """
        <what/why>.
        :param x: <desc>
        :returns: <desc>
        """
```

### Class template
```python
class Widget:
    """
    <one‑line summary>.
    :param size: <desc>
    :param color: <desc>
    """
    def __init__(self, size: int, color: str) -> None:
        self.size = size
        self.color = color
```

---

## VS Code Setup
Save this file at: `.github/copilot-instructions.md`

Ensure Copilot reads instruction files (VS Code `settings.json`):
```json
{
  "github.copilot.chat.codeGeneration.useInstructionFiles": true
}
```
