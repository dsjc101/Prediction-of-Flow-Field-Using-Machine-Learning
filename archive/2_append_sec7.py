import os

cl_path = r".\Downloads\openfoam_pinn_pipeline\CHANGELOG.md"

sec7 = """

## 7. Open-Access Reference Overhaul and Final Clean Build

### DONE
1. **Paywalled Reference Elimination and Free Open-Access Replacement:**
   - Every citation in `report.tex` now links directly to an open-access PDF (arXiv, JMLR/PMLR, ICCFD) or public data repository (ODEN UTEXAS MKM, Princeton Superpipe).
   - Replaced Rao et al. (2020) paywalled DOI link with open-access arXiv PDF: `https://arxiv.org/pdf/2002.10558.pdf#page=2`.
   - Replaced Spalart-Allmaras (1992) AIAA paper with Allmaras, Johnson, Spalart (2012, ICCFD7-1902) open-access PDF: `https://iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf#page=3`. Reworded text to cite the no-ft2 variant (ct3 = 0).
   - Replaced paywalled Dean (1978) citation with Tsukahara et al. (2014, arXiv:1406.0248) open-access PDF: `https://arxiv.org/pdf/1406.0248.pdf#page=6`, where Dean's channel skin friction correlation Cf = 0.073 Rem^{-0.25} is explicitly quoted.
   - Replaced closed-access entrance length citations (Atkinson 1969, Chen 1973, Shah & London 1978) with an empirical correlation note, adding comparison to Durst et al. via Ray et al. (2017, arXiv:1707.04947) open-access PDF: `https://arxiv.org/pdf/1707.04947.pdf#page=8`.
   - Replaced closed-access Superpipe citations (Zagarola 1998, McKeon 2004) with the public Princeton Superpipe Pitot Data repository: `https://smits.princeton.edu/superpipe-pitot-data`.
   - Replaced general textbooks and user guides (White 2006, Schlichting & Gersten 2017, OpenFOAM 2019) with domain-specific descriptions, retaining clean mathematical formulation.
2. **Page Anchor Verification via pdftotext:**
   - Allmaras 2012: verified on PDF p. 3 (ct3 = 0).
   - Glorot 2010: verified on PDF p. 5 (Eq. 16, Xavier initialization).
   - Pirozzoli 2021: verified on PDF p. 7 (friction factor and pipe DNS profiles).
   - Raissi 2019: verified on PDF p. 6 (Navier-Stokes PINN formulation).
   - Rao 2020: verified on PDF p. 2 (incompressible Navier-Stokes PINN formulation).
   - Ray 2017: verified on PDF p. 8 (Eq. 2a and 2b entrance lengths).
   - Tsukahara 2014: verified on PDF p. 6 (Dean channel friction correlation).
   - Wang 2021: verified on PDF p. 10 (learning rate annealing for gradient pathologies).
3. **Typo Fix (Sec. 5.6):**
   - Verified that line 543 properly typesets `$U_{\\mathrm{ref}} = 0.075$ m/s` and `$p_{\\mathrm{scale}} = 0.10547$ m$^2$/s$^2$`.
4. **Clean Compilation and Artifact Verification:**
   - Compiled cleanly twice with `miktex-pdftex.exe`.
   - Exactly 0 undefined citations or references (`grep -c "undefined" report.log == 0`).
   - Exactly 0 overfull `\\hbox > 1pt`.
   - Exactly 0 overfull `\\vbox`.
   - Document length: exactly 47 pages.
   - Output diff saved in `final_diff.txt`.
   - Generated `link_audit.csv` covering all 11 active references.

### CHANGED (old to new)
- `bodoia1961` -> Replaced with Hornbeck (1973) NASA SP-297 reference.
- `white2006,shah1978,hornbeck1973` -> Replaced with Hornbeck (1973) NASA SP-297 reference.
- `openfoam2019` -> Replaced with descriptive reference to OpenFOAM elliptic finite-volume solver.
- `schlichting2017` -> Replaced with classical boundary layer theory description.
- `spalart1992,allmaras2012` -> `allmaras2012` [p. 3] (Allmaras, Johnson, Spalart 2012 ICCFD7-1902).
- `zagarola1998,mckeon2004` -> `superpipe_data` (Princeton Superpipe Pitot Data repository).
- `atkinson1969,shah1978` -> Replaced with empirical correlation note and Durst et al. comparison via `ray2017` [p. 8].
- `chen1973` -> Replaced with empirical correlation note and Durst et al. comparison via `ray2017` [p. 8].
- `dean1978` -> `tsukahara2014` [p. 6] (Tsukahara et al. 2014, arXiv:1406.0248 quoting Dean correlation).
- `rao2020` (DOI) -> `rao2020` (arXiv:2002.10558 [p. 2]).
- Bibliography entries removed: `atkinson1969`, `bodoia1961`, `chen1973`, `dean1978`, `openfoam2019`, `schlichting2017`, `shah1978`, `spalart1992`, `white2006`, `zagarola1998`.
- Bibliography entries added: `ray2017`, `superpipe_data`, `tsukahara2014`.

### UNVERIFIED
- **Hornbeck (1973) NASA SP-297 PDF Page Index:**
  - NASA NTRS record 19740008333 is open-access (HTTP 200).
  - The downloadable document is a scanned bitmap without an OCR text layer; automated `pdftotext` cannot extract text. Cited via NTRS record landing page without a `#page=N` fragment anchor.
- **Superpipe Logarithmic Law Constants (1.930, -0.537):**
  - Constants are not quoted in open-access PDFs; kept as descriptive empirical fit in Sec. 8 text; data repository `superpipe_data` is cited directly.
- **Empirical Entrance-Length Correlation Constants (Eq. 5 and Eq. 23):**
  - Constants in Eq. 5 (0.315 / (1 + 0.0175 Re) + 0.011 Re) and Eq. 23 (0.6 / (0.14 Re + 1.0) + 0.0575 Re) are maintained from prior technical reports and checked against Durst et al. (`ray2017` [p. 8]); original closed-access derivations are labelled as empirical correlations.

### NOT DONE
- **Fig 1/2 Minor Label Micro-Adjustments (Step 5):**
  - Left as-is per instruction ("If the fix takes more than one attempt, SKIP it"). The existing TikZ schematics have white badges (`fill=white, inner sep=2pt`) and readable margins, with zero overfull box errors.
"""

with open(cl_path, "a", encoding="utf-8") as f:
    f.write(sec7)

print("Appended section 7 to CHANGELOG.md successfully!")

