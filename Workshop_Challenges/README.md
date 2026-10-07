# PharmaPy workshop

These notebooks introduce PharmaPy, a Python framework for simulating pharmaceutical processes: reactors, crystallizers, filters and the flowsheets that connect them. You will build models, run them, and optimize them with `scipy.optimize`.

## Open the notebooks in Colab

| Notebook | Exercise | Solution |
| --- | --- | --- |
| Challenge 1: parameter estimation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/Challenge1_ParameterEstimation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/solutions/Challenge1_ParameterEstimation_solution.ipynb) |
| Challenge 2: crystallizer design | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/Challenge2_CrystallizerDesign.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/solutions/Challenge2_CrystallizerDesign_solution.ipynb) |
| Challenge 3: flowsheet design | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/Challenge3_FlowsheetDesign.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/solutions/Challenge3_FlowsheetDesign_solution.ipynb) |
| Syntax examples (reference) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZacharyHillman18/PharmapyWorkshop/blob/main/Workshop_Challenges/SyntaxExamples.ipynb) |  |

## Running on Google Colab

1. Click the notebook's **Open in Colab** badge in the table above.
2. Run the first code cell, **Colab setup**. It installs PharmaPy and unpacks `data/` and `workshop_utils.py` onto the Colab machine, and ends by printing `PharmaPy 0.3.1 - workshop files ready in /content` (or a newer version).
3. Run the cells from top to bottom.
4. Use *File → Save a copy in Drive* to keep your answers. Changes to a notebook opened from a badge are not saved otherwise.

Each notebook runs on its own Colab machine, so run the setup cell in every notebook you open, including the syntax examples. When the runtime restarts or disconnects, the files and variables are gone: run the setup cell again, then the cells above the one you were working on. Colab's free machines are slower than most laptops, so the optimization cells take longer than the times the notebooks give.

## What is in the folder

| File | What it is |
| --- | --- |
| Three numbered notebooks (see below) | The exercises, done in order |
| `SyntaxExamples.ipynb` | Small, runnable examples of every PharmaPy object. Look here when you are unsure of the syntax |
| `workshop_utils.py` | Plotting helpers and crystal size statistics used by the notebooks (you do not need to read it) |
| `data/` | Property files (`.json`) and experimental data |

The workshop comes in three versions. You received one of them:

| Folder | Notebooks | How much is filled in |
| --- | --- | --- |
| `Workshop_Notebooks` | `ReactorsParameterEstimation`, `CrystallizersOptimization`, `FlowsheetOptimization` | Most of the code; you complete the key lines |
| `Workshop_Notebooks_Advanced` | The same three notebooks | Less filled in: you write the unit set-up, objectives and optimizer calls |
| `Workshop_Challenges` | `Challenge1_ParameterEstimation`, `Challenge2_CrystallizerDesign`, `Challenge3_FlowsheetDesign` | A worked example, a guided part, then an open task you code yourself, followed by discussion questions |

The standard and advanced versions have the same structure, names and functions, so you can switch between them at any notebook.

## Completing the exercises

- A line ending in `# <-- COMPLETE THIS: ...` has a `...` for you to replace; the comment says what goes there.
- In the challenge notebooks, a cell that starts with `# <-- COMPLETE THIS: your code: ...` is yours to write. The markdown cell above it gives the math to implement and the PharmaPy syntax you need.
- Questions to think about end with an empty **Your answer:** cell.

## Good to know

- Most simulations take seconds. A few optimization cells take 1 to 2 minutes; the notebook says which.
- Build a **new** unit (reactor, crystallizer, filter) for every simulation. Calling `solve_unit` again on the same unit continues the previous run.
- Assign in this order: the unit's `Phases` first, then its kinetics (`RxnKinetics` or `CrystKinetics`), then any `Inlet`.
- Units: K, s, m³ and kg; concentrations in mol/L; crystal sizes in µm.
