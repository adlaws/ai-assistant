---
applyTo: '**/*.md'
description: 'Markdown formatting guidance for JerryCAN. Ensures consistent styling for all markdown files.'
---

# Markdown Formatting Rules

## Mission

Keep markdown formatting consistent across all JerryCAN documentation and instructions files
and aligned with formatting rules.

## Formatting Rules

Markdown must conform to the following `markdownlint` rule set:

```json
{
    "default": false,
    "MD001": true,
    "MD002": false,
    "MD003": { "style": "atx" },
    "MD004": { "style": "asterisk" },
    "MD005": true,
    "MD006": false,
    "MD007": { "indent": 4, "start_indented": false },
    "MD009": { "br_spaces": 0, "list_item_empty_lines": false, "strict": true },
    "MD010": { "code_blocks": false },
    "MD011": true,
    "MD012": { "maximum": 1 },
    "MD013": false,
    "MD014": true,
    "MD018": true,
    "MD019": true,
    "MD020": false,
    "MD021": false,
    "MD022": { "lines_above": 1, "lines_below": 0 },
    "MD023": true,
    "MD024": { "allow_different_nesting": false, "siblings_only": false },
    "MD025": { "level": 1, "front_matter_title": ""},
    "MD026": { "punctuation": ".,;:!?。，；：！？" },
    "MD027": true,
    "MD028": true,
    "MD029": { "style": "ordered" },
    "MD030": { "ul_single": 1, "ol_single": 1, "ul_multi": 1, "ol_multi": 1 },
    "MD031": { "list_items": true },
    "MD032": true,
    "MD033": { "allowed_elements": ["br"] },
    "MD034": true,
    "MD035": { "style": "---" },
    "MD036": { "punctuation": ".,;:!?。，；：！？" },
    "MD037": true,
    "MD038": true,
    "MD039": true,
    "MD040": true,
    "MD041": { "level": 1, "front_matter_title": "" },
    "MD042": true,
    "MD043": false,
    "MD044": false,
    "MD045": true,
    "MD046": { "style": "fenced" },
    "MD047": true,
    "MD048": { "style": "backtick" }
}
```

## Plain-English Summary

The JSON above is the source of truth. In practical terms, it means:

* Use ATX headings (`#`, `##`, `###`) and keep heading levels ordered.
* Use `*` for unordered list items.
* Use fenced code blocks with backticks and always include a language/info string.
* Leave a single blank line where needed around lists, headings, and code fences.
* Do not leave trailing spaces or multiple consecutive blank lines.
* Keep one H1 per file.
* Do not use duplicate heading text within the same file (MD024). Every heading
  must be unique regardless of nesting level.
* Avoid bare URLs; use proper markdown links.
* Avoid raw HTML except `<br>`.

## Diagrams

Use Mermaid diagrams (refer to <https://mermaid.js.org/intro/syntax-reference.html> for syntax)
for...

* Flowcharts
* Sequence Diagrams
* Class Diagrams
* State Diagrams
* Entity Relationship Diagrams
* User Journey Charts
* Gantt CHarts
* Pie Charts
* Quadrant Charts
* Requirement Diagrams
* GitGraph (Git) Diagrams
* C4 Diagrams
* Mindmaps
* Timelines
* ZenUML Diagrams
* Sankey Chart
* XY Charts
* Block Diagrams
* Packet
* Kanban
* Architecture
* Radar
* Treemap
* Venn
* Ishikawa
* TreeView

ASCII Art diagrams may also be used.

## Formulae

While simple add, subtract, divide, and multiply operations may be represented with plaint text,
consideration should given to formatting them using Latex mathematical expression syntax, such as
inline syntax such as $s = ut + \frac{1}{2}at^2$, or multiline syntax, such as:

$$
a = \max\left(\frac{F_{\mathrm{net}}}{m \cdot (1 + k_{\mathrm{rot}})}, 0\right)
$$
