# MARKDOWN & MATH PORTABILITY RULESET
*(Markdown Monster + GitHub Compatible)*

This file defines the mandatory authoring rules for all Markdown and LaTeX
content in this repository. All tools and LLMs (including Claude Code) MUST
follow these rules.

---

## 1. Ordered & Nested Lists

### Top-Level Ordered Lists
- Always use `1.` for every item.
- Never manually increment numbers.

Example:

1. First item
1. Second item
1. Third item

### Nested Lists
- Nested items must use `-`
- Indent nested lists by exactly **3 spaces**.

Example:

1. Section
   - Subsection
   - Subsection
1. Next Section

---

## 2. Mathematics

### Inline Math
Use single-dollar delimiters:

$a^2 + b^2 = c^2$

---

### Display Math (Block)

- Use double-dollar blocks ONLY.
- `$$` must appear on their own lines.
- Surround block with blank lines.

Example:

$$
E = mc^2
$$

- Do NOT use fenced math blocks:

```math
...
```

---

### Indexing (Subscripts & Superscripts)

- Always brace indices:

x_{i}
x^{2}

- Never use literal `*` in superscripts.
- Use `\ast` instead:

x^{\ast}

---

### Identifiers & Function Names

- Do NOT use `\text{}`
- Do NOT use `\operatorname{}`

Use `\mathrm{}` for all identifiers:

\mathrm{Score}(x)
\mathrm{terminal\_value}(s)

---

### Operators

- Do NOT use `\substack`
- Do NOT use AMS environments (`align`, `cases`, `equation`, etc.)

Flatten conditions into comma-separated subscripts:

\max_{x \in A,\ \mathrm{(condition)}}

---

### Multiple Equations

Place in the same block separated by blank lines:

$$
a = b + c

d = e + f
$$

---

### Escaping Rules

- Inside math:
  - Dollar sign → `\$`
  - Underscore inside `\mathrm{}` → `\_`

---

## 3. Text Near Math

If text immediately precedes display math, force a line break:

Definition:\\

$$
x = y
$$

---

## 4. Prohibited LaTeX

Never use:

\text{}
\operatorname{}
\substack{}
\begin{align}
\begin{equation}
^*
$$inline$$

---

## 5. Headings & Anchors

- Use ATX headings only:

## Section Title

- Avoid punctuation in headings when possible.
- Prefer hyphenated words.

---

## 6. Code Blocks

- Always use triple backticks.
- Always specify language.

Example:

```python
print("hello")
```

---

## 7. Tables

- Pipes only.
- No alignment colons.

Example:

| Name | Value |
|-----|------|
| A | 1 |

---

## 8. General Principles

- Prefer simple primitives over fancy LaTeX.
- No HTML inside Markdown.
- No mixed list marker types at the same indentation level.
- Always prioritize portability over typographic elegance.

---

## 9. Claude Instruction

When generating or editing Markdown in this repository, Claude MUST follow
this ruleset exactly.

---

# End of File

