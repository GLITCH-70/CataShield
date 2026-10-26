# IEEE conference paper

`main.pdf` / `main.docx` — *Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical
Instrument Segmentation* (IEEE conference format, two columns; PDF 9 pages, Word 8 pages).

| File | Contents |
|---|---|
| `main.tex` | Paper text (IEEEtran) |
| `refs.bib` | References; arXiv entries checked against the arXiv API. `khan2024guardian` still needs author initials (marked VERIFY) |
| `numbers.tex` | Every number quoted in the text, as `\n...` macros — generated, do not edit |
| `tables/*.tex` | All tables — generated, do not edit |
| `figures/` | All figures — generated |
| `build_paper_assets.py` | Regenerates `numbers.tex`, `tables/` and `figures/` from `results/` |
| `overleaf_upload.zip` | Everything needed to compile on Overleaf |
| `export_docx.py` | Converts the compiled paper to `main.docx` (IEEE two-column Word layout, native equations) |

## Rebuild

From the project root, regenerate the data-dependent parts after any change to `results/`:

```bash
python paper/build_paper_assets.py
```

Compile (Tectonic is in `tools/tectonic/`; it runs BibTeX automatically):

```bash
cd paper && ../tools/tectonic/tectonic.exe main.tex
```

On Overleaf: upload `overleaf_upload.zip` as a new project; it compiles with pdfLaTeX + BibTeX as is.

Word version (needs `pip install pypandoc_binary python-docx`; compile with `--keep-intermediates` first so the
citation and figure numbers are available):

```bash
cd paper && ../tools/tectonic/tectonic.exe --keep-intermediates main.tex
```

```bash
python paper/export_docx.py
```

## Before submission

- Confirm the author list, order, affiliations and e-mail addresses (author block in `main.tex`).
- Complete the author initials of the GUARDIAN reference (`refs.bib`, marked VERIFY).
- Check the target venue's page limit and template (this uses the standard IEEE conference class).
