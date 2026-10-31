"""Conservatively clean extracted document text."""


def clean_text(text: str) -> str:
    """Trim outer blank lines and reduce repeated blank lines to one.

    Nonblank lines are preserved exactly, including their indentation and
    horizontal whitespace. CRLF and CR line endings are normalized to LF.
    """
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []

    for line in normalized_text.split("\n"):
        if line.strip():
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)
