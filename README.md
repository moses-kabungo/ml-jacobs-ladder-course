# Jacobs Ladder ML Course

This repository contains code and resources for the Jacobs Ladder Machine Learning course.

## Structure
- `visualization.py`: Contains visualization utilities and functions.
- `marketing_kpis.py`: Bayesian marketing KPI estimation functions for lead generation and conversion analysis.
- `__init__.py`: Marks the directory as a Python package.
- `Presentations/`: Contains course presentation materials in Quarto format (.qmd files).
  - `bayesian_inference.qmd`: Statistical Inference II — Bayesian Inference presentation.

## Usage
Import the modules as needed in your projects or notebooks. For example:

```python
from jacobs_ladder import visualization
import jacobs_ladder.marketing_kpis as mkpi

# Example: Estimate marketing KPIs
results = mkpi.estimate_lead_conversions_kpi('data/marketing_data.csv')
```

## Requirements
- Python 3.12 or higher
- Quarto (for rendering presentation slides)
- (Add any additional dependencies here)

## Getting Started
1. Clone the repository:
   ```zsh
   git clone <repo-url>
   ```
2. Navigate to the project directory:
   ```zsh
   cd jacobs_ladder
   ```

3. Install dependencies (if any):
   ```zsh
   pip install -r requirements.txt
   ```

## Using in a Jupyter Environment

To use this project in a Jupyter Notebook:

1. (Optional) Clone the repository directly from a notebook cell:
   ```python
   # Clone the repository (run this in a notebook cell)
   !git clone git@github.com:moses-kabungo/ml-jacobs-ladder-course.git
   ```

2. Install the requirements for the project:
   ```zsh
   pip install -r ml-jacobs-ladder-course/requirements.txt
   ```

3. In your notebook, import the visualization module as follows:
   ```python
   import jacobs_ladder.visualization as jvis
   # Now you can use jvis.function_name()
   ```

4. Make sure your notebook's kernel is using the same Python environment where you installed the dependencies.

## Presentations

The `Presentations/` folder contains course materials in Quarto format:

- **Bayesian Inference**: Statistical inference using Bayesian methods, covering priors, posteriors, conjugate pairs, and practical applications in email marketing and conversion analysis.

To render presentations:
```zsh
# Install Quarto if you haven't already
# Visit: https://quarto.org/docs/get-started/

# Render a specific presentation
quarto render Presentations/bayesian_inference.qmd

# Preview with live reload
quarto preview Presentations/bayesian_inference.qmd
```

## License
MIT
