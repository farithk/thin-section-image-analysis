# Image analysis of a sandstone thin section

Exercise 1 of *Reservoir Geological Characterization* (2026): porosity, grains, pores, heterogeneity and grain shape
measured from two photographs of a blue-epoxy-impregnated sandstone thin section, solved in Python as an
image-processing problem.

## What is solved, and where

| Task | Question | Notebook |
|---|---|---|
| 1 | How much of the rock is pore, grain and cement? | [01](notebooks/01_three_phase_segmentation.ipynb) |
| 2 | How large must a sample be for a reliable porosity? (REV) | [04](notebooks/04_heterogeneity.ipynb) |
| 3 | Segment the individual grains | [02](notebooks/02_pore_and_grain_objects.ipynb) |
| C1 | Heterogeneity of the pore and grain distribution | [04](notebooks/04_heterogeneity.ipynb) |
| C2 | Segment the individual pores | [02](notebooks/02_pore_and_grain_objects.ipynb) |
| C3 | Area and size of every pore and grain | [03](notebooks/03_areas_and_size_distributions.ipynb) |
| C4 | Data for a Fourier analysis of grain shape | [05](notebooks/05_fourier_grain_shape.ipynb) |

Notebook 01 starts with a full overview: the method of each task, the sections where it is solved, the output
files and the main results.

## Method in short

1. **Pixel classification** (CIELAB colour space): Otsu brightness threshold for grains, a blue hue/chroma rule for
   pores, cement as the rest, small-speck clean-up.
2. **Object separation**: distance transform, h-maxima markers, watershed and neck-ratio merge; cross-checked with
   PoreSpy SNOW.
3. **Measurements**: areas, equivalent diameters, Folk & Ward sorting, pore throats, shape descriptors; both images
   put in one pixel unit by template matching.
4. **Heterogeneity**: REV from random windows, two-point correlation and semivariogram, integral range,
   lacunarity, Clark-Evans and Ripley point-pattern tests, Spearman correlations on blocks.
5. **Grain shape**: radius-angle Fourier harmonics r(θ) and elliptic Fourier descriptors, a resolution test with
   the two images, PCA, Gaussian mixtures and a fabric (orientation) test.

## Repository layout

| Folder | Content |
|---|---|
| `notebooks/` | the five analysis notebooks (01-05), saved with their results |
| `src/` | shared modules: `segmentation.py`, `objects.py`, `sizestats.py`, `heterogeneity.py`, `shape.py`, `figstyle.py` |
| `scripts/` | scripts that draw the explanatory diagrams (saved to `outputs/figures/`) |
| `outputs/` | tables (CSV), label maps (`.npy`) and figures produced by the notebooks |

## Running it

```bash
pip install -r requirements.txt
```

The two thin-section photographs of the course (`Lamina_alunos2.png` and `Lamina_alunos_grande.png`) are **not
included**. Place them in the repository root, then run the notebooks in order 01 to 05; each one reads the outputs
of the previous ones. Sizes are in pixels, because the photographs have no scale bar.

## Author

Farith Comas
