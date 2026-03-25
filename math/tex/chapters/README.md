# Chapter sources

These files are `\input` from `math.tex`. Run `pdflatex` from the **`math/tex/`** directory so paths resolve.

| File | Content |
|------|--------|
| `introduction.tex` | Unnumbered introduction |
| `architecture.tex` | Chapter 1 – Architecture |
| `stimulus.tex` | Chapter 2 – Stimulus Optimization… |
| `lazy-updates.tex` | Chapter 3 – Lazy Updates |
| `embellishments.tex` | Chapter 4 – Embellishments |

Use `\input` (not `\include`) so there is no forced `\clearpage` between chapters. Use `\include` if you want per-chapter aux files and `\includeonly{...}` for partial builds.
