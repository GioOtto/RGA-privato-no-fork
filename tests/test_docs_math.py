"""The maths in the Markdown files must survive GitHub's renderer.

GitHub renders a Markdown file in two passes: the Markdown pass runs *first*
and treats ``$`` as ordinary text, then a second pass finds the delimiters and
hands the contents to MathJax. Two things get destroyed in between, and both
had silently broken this repository's documentation:

1. **Backslash escapes.** The Markdown pass strips a backslash before any
   non-letter, inside maths as much as outside, because ``\\,`` is a legal
   CommonMark escape of an ASCII punctuation character. ``docs/THEORY.md`` had
   57 such sequences on 22 lines - ``\\,``, ``\\;``, ``\\!``, ``\\{``, ``\\}``,
   ``\\_`` - so ``\\mathbb 1\\{Z \\ge z_0\\}`` reached MathJax as
   ``\\mathbb 1{Z \\ge z_0}`` with the set braces gone, and
   ``\\texttt{income\\_copy}`` arrived as ``\\texttt{income_copy}``, which
   MathJax renders with a subscript.
2. **Disallowed macros.** GitHub configures MathJax with a restricted macro
   allow-list and replaces the whole block with "The following macros are not
   allowed: ..." when it meets one. ``\\operatorname`` is the one this
   repository used, nine times, including for the central identity in the
   README.

Both are avoided by the delimiters GitHub documents for exactly this purpose:
a ```` ```math ```` fenced block for display maths and ``$`...`$`` for inline
maths. Their contents are code to the Markdown pass, so nothing is stripped.
Verified against GitHub's own renderer (``POST /markdown``, ``mode=gfm``),
which returns the payload MathJax will receive.

These tests are the reason the fix stays fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/THEORY.md", "CHANGELOG.md", "MIGRATION.md", "NOTICE.md"]

BACKSLASH = chr(92)

#: A backslash before anything that is not a letter. CommonMark eats it.
FRAGILE_ESCAPE = re.compile(re.escape(BACKSLASH) + r"[^a-zA-Z]")

#: Macros GitHub's MathJax configuration refuses outright.
BLOCKED_MACROS = (
    "operatorname",
    "def",
    "newcommand",
    "renewcommand",
    "DeclareMathOperator",
    "require",
    "label",
    "eqref",
)

#: ```math fenced block - contents are safe.
MATH_FENCE = re.compile(r"^```math\n(.*?)^```$", re.M | re.S)

#: $`...`$ inline - contents are safe.
SAFE_INLINE = re.compile(r"\$`[^`]*`\$")

#: $$...$$ display - contents are NOT safe.
FRAGILE_DISPLAY = re.compile(r"\$\$.*?\$\$", re.S)

#: $...$ inline - contents are NOT safe. Deliberately does not cross a blank
#: line, so an unpaired '$' cannot swallow the rest of the file.
FRAGILE_INLINE = re.compile(r"(?<![$`])\$(?!\$)([^$\n]+?)\$(?!\$)")


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def math_regions(text: str) -> list[str]:
    """Every span whose contents MathJax will be asked to typeset.

    A blocked macro only matters *inside* maths - the changelog and the
    migration notes name `\\operatorname` in prose, inside code spans, and must
    be allowed to.
    """
    return (
        MATH_FENCE.findall(text)
        + SAFE_INLINE.findall(text)
        + FRAGILE_INLINE.findall(prose(text))
        + FRAGILE_DISPLAY.findall(prose(text))
    )


def prose(text: str) -> str:
    """Strip everything where a '$' is not fragile inline/display maths.

    Order matters: ``$`x`$`` must be removed *before* plain inline code, or
    stripping the backticks would leave a bare ``$$`` behind and every safe
    formula would be reported as a fragile one.
    """
    text = MATH_FENCE.sub("", text)  # ```math blocks: safe
    text = re.sub(r"^```.*?^```$", "", text, flags=re.M | re.S)  # other fences
    text = SAFE_INLINE.sub("", text)  # $`...`$ : safe
    return re.sub(r"`[^`\n]*`", "", text)  # remaining inline code


@pytest.mark.parametrize("name", DOCS)
def test_no_blocked_macros(name):
    """A formula GitHub refuses to draw is worse than no formula.

    Only maths regions are checked: the changelog names ``\\operatorname`` in
    prose, in a code span, to explain why it was removed.
    """
    for region in math_regions(read(name)):
        for macro in BLOCKED_MACROS:
            assert BACKSLASH + macro not in region, (
                f"{name} uses {BACKSLASH + macro!r} inside maths, which "
                f"GitHub's MathJax allow-list rejects: the block renders as "
                f"'The following macros are not allowed'. Use "
                f"{BACKSLASH}mathrm (plus {BACKSLASH}, for the spacing an "
                f"operator would have added by itself). In: {region[:80]!r}"
            )


@pytest.mark.parametrize("name", DOCS)
def test_display_maths_uses_a_math_fence(name):
    """$$...$$ loses every backslash-escape; ```math does not."""
    offenders = FRAGILE_DISPLAY.findall(prose(read(name)))
    assert not offenders, (
        f"{name} delimits display maths with $$...$$, whose contents are "
        f"sanitised by the Markdown pass. Use a ```math fenced block. "
        f"First offender: {offenders[0][:90]!r}"
    )


@pytest.mark.parametrize("name", DOCS)
def test_inline_maths_with_escapes_uses_backticks(name):
    """Plain $...$ may only hold maths that survives the sanitiser."""
    body = prose(read(name))
    bad = [
        found for found in FRAGILE_INLINE.findall(body) if FRAGILE_ESCAPE.search(found)
    ]
    assert not bad, (
        f"{name} has inline maths in plain $...$ containing a backslash "
        f"before a non-letter, which GitHub strips before MathJax sees it. "
        f"Wrap it as $`...`$. Offenders: {bad[:3]!r}"
    )


@pytest.mark.parametrize("name", DOCS)
def test_inline_maths_never_spans_a_line_break(name):
    """GitHub does not recognise a $...$ region that contains a newline.

    ``docs/THEORY.md`` had one at lines 346-347; it rendered as literal LaTeX
    source while the ``$\\sigma_g$`` immediately after it rendered fine.
    """
    for paragraph in re.split(r"\n[ \t]*\n", prose(read(name))):
        if "$" not in paragraph:
            continue
        # Within a paragraph, count '$' per line: a line with an odd count
        # opens (or closes) a region that must be closed on another line.
        odd = [
            line
            for line in paragraph.splitlines()
            if line.count("$") % 2 == 1 and not line.strip().startswith("$$")
        ]
        assert not odd, (
            f"{name} appears to open an inline maths region on one line and "
            f"close it on another; GitHub will not render it. Keep $...$ on a "
            f"single line. Offending line(s): {odd[:2]!r}"
        )


@pytest.mark.parametrize("name", DOCS)
def test_math_fences_are_balanced(name):
    """Count only fences that open a line - the changelog quotes the syntax."""
    text = read(name)
    opened = len(re.findall(r"^```math$", text, flags=re.M))
    assert opened == len(MATH_FENCE.findall(text)), (
        f"{name} has an unterminated ```math fence "
        f"({opened} opened, {len(MATH_FENCE.findall(text))} closed)."
    )


def test_the_central_identity_is_present_and_renderable():
    """The one formula the whole package rests on, specifically."""
    text = read("README.md")
    blocks = MATH_FENCE.findall(text)
    identity = [block for block in blocks if "RGA" in block and "cov" in block]
    assert identity, "README.md no longer states the RGA/Gini-covariance identity"
    body = identity[0]
    assert BACKSLASH + "operatorname" not in body
    assert BACKSLASH + "mathrm{cov}" in body
    # The thin space that \operatorname used to supply between '2' and 'cov'.
    assert "2" + BACKSLASH + ",", body
