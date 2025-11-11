# OnlineMarketplace_Presentation.ipynb - User Guide

**Created:** November 11, 2025

## Overview

This Jupyter notebook provides a comprehensive, presentation-ready overview of your e-commerce marketplace simulation project. It integrates all tasks (data cleaning, seller modeling, Nash equilibrium, sensitivity analysis, and network effects) into a single, coherent narrative.

---

## Notebook Structure

### 1. **Introduction & Executive Summary**
- Project objectives and methodology
- Dataset overview
- Key concepts (Game Theory, Nash Equilibrium, Network Effects)

### 2. **Task 1: Data Cleaning & EDA**
- Data cleaning pipeline explanation
- Outlier handling methodology (IQR with winsorization)
- Summary statistics
- **Displays:**
  - `outlier_handling_results.png`
  - `Task1_EDA_Results.png`

### 3. **Task 2: Seller Modeling & Market Structure**
- Seller creation strategy (price tiers)
- Mathematical model (demand and profit functions)
- Seller characteristics table
- **Displays:**
  - `seller_analysis.png`
  - `profit_landscape.png`

### 4. **Task 3: Game Theory Simulation - Nash Equilibrium**
- Nash equilibrium concept explanation
- Iterative best response algorithm
- Convergence analysis
- Results interpretation
- **Displays:**
  - `nash_equilibrium.png` (6-subplot convergence)
  - `profit_comparison.png` (initial vs Nash)
  - `nash_on_landscape.png` (equilibrium location)
  - `parameter_sensitivity.png`

### 5. **Task 4: Parameter Sensitivity Analysis**
- α (advertising), β (price sensitivity), ε (elasticity) testing
- Impact on profits and strategies
- Strategic implications

### 6. **Task 5: Social Network Effects**
- Network modeling approach
- Influence metrics (centrality)
- Updated demand function with γ (gamma)
- **Displays:**
  - `social_network.png`
  - `network_impact_comparison.png`

### 7. **Conclusions & Key Insights**
- Summary of findings
- Theoretical contributions
- Practical recommendations
- Limitations and future work

### 8. **How to Reproduce**
- Software requirements
- Execution steps
- Project structure
- Computational costs

---

## How to Use This Notebook

### Option 1: View Only (Fastest)
```bash
jupyter notebook OnlineMarketplace_Presentation2.ipynb
```
- Open the notebook
- Run ONLY the setup cell (imports)
- All visualizations display from pre-generated PNG files
- **Runtime:** < 5 seconds

### Option 2: Full Reproduction (Complete)
```bash
# 1. Run all analysis scripts first
cd Task1 && python DataCleaning.py
cd ../Task2 && python SellerModeling.py
cd ../Task3 && python GameTheorySimulation.py
cd ../Task4 && python NetworkIntegratedSimulation.py

# 2. Open notebook
jupyter notebook OnlineMarketplace_Presentation2.ipynb

# 3. Run all cells
```
- **Runtime:** ~15-20 minutes total

### Option 3: Export to PDF/HTML
```bash
# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf OnlineMarketplace_Presentation2.ipynb

# Convert to HTML (no dependencies)
jupyter nbconvert --to html OnlineMarketplace_Presentation2.ipynb
```

---

## Key Features

### ✅ Presentation-Ready
- Professional markdown formatting
- Clear section headers
- Concise explanations
- Mathematical formulas (LaTeX)

### ✅ Optimized for Speed
- Uses `IPython.display.Image()` for pre-rendered visualizations
- No expensive computations re-run during presentation
- Fast notebook loading and execution

### ✅ Comprehensive Coverage
- All 5 tasks integrated
- Theoretical explanations + practical results
- Visual evidence for every claim

### ✅ Educational
- Game theory concepts explained
- Economic interpretations provided
- Strategic insights highlighted

### ✅ Reproducible
- Complete execution instructions
- Requirements clearly listed
- Project structure documented

---

## Visualizations Referenced

The notebook displays these pre-generated images:

### Task 1:
- `Data/ProcessedData/outlier_handling_results.png`
- `Task1/Task1_EDA_Results.png`

### Task 2:
- `Task2/seller_analysis.png`
- `Task2/profit_landscape.png`

### Task 3:
- `Task3/nash_equilibrium.png`
- `Task3/profit_comparison.png`
- `Task3/nash_on_landscape.png`
- `Task3/parameter_sensitivity.png`

### Task 4:
- `Task4/social_network.png`
- `Task4/network_impact_comparison.png`

**Note:** All these files should already exist from running your analysis scripts.

---

## Customization Tips

### To Add More Content:
1. Insert new Markdown cells for explanations
2. Insert new Code cells for demonstrations
3. Add more `display(Image(...))` calls for additional figures

### To Modify Existing Content:
1. Edit Markdown cells for different wording
2. Adjust code cells if you want to show live computations
3. Update file paths if your structure differs

### To Create Variations:
```python
# Show specific subset of results
display(Image(filename='Task3/nash_equilibrium.png', width=800))

# Load and display data dynamically
df = pd.read_csv('Data/ProcessedData/cleaned_online_retail_data.csv')
print(f"Dataset has {len(df):,} rows")
```

---

## Troubleshooting

### Issue: Images Don't Display
**Solution:** Ensure all PNG files exist. Run analysis scripts first:
```bash
python Task1/DataCleaning.py
python Task2/SellerModeling.py
python Task3/GameTheorySimulation.py
python Task4/NetworkIntegratedSimulation.py
```

### Issue: "File Not Found" Errors
**Solution:** Check file paths are relative to notebook location:
```python
# Correct (relative paths from notebook root)
display(Image(filename='Task3/nash_equilibrium.png'))

# Incorrect (absolute paths)
display(Image(filename='/home/user/...'))
```

### Issue: Notebook Runs Slowly
**Solution:** Don't re-run expensive computations. Just display images:
```python
# Fast (recommended)
display(Image(filename='Task3/nash_equilibrium.png'))

# Slow (avoid)
nash_result = find_nash_equilibrium(market, seller_A, seller_B, ...)
```

---

## Presentation Tips

### For Live Demo:
1. **Run setup cell first** (imports)
2. **Execute cells sequentially** (Shift+Enter)
3. **Pause to explain** complex concepts
4. **Highlight key insights** in each section

### For PDF Export:
1. **Run all cells** before exporting
2. **Check image display** quality
3. **Adjust figure sizes** if needed
4. **Test PDF rendering** (some LaTeX may need adjustment)

### For Academic Submission:
1. **Add your name/ID** in title cell
2. **Include references** (already in Conclusions)
3. **Add acknowledgments** if needed
4. **Check university formatting** requirements

---

## Next Steps

### To Enhance This Notebook:
1. ✅ Add your personal information to title
2. ✅ Include instructor/course details
3. ✅ Add more economic interpretation
4. ✅ Expand theoretical background
5. ✅ Include comparison with real-world markets

### To Create Additional Materials:
1. **Slides:** Use RISE extension for slideshow mode
2. **Report:** Export to LaTeX → compile to PDF
3. **Interactive:** Add ipywidgets for parameter exploration
4. **Dashboard:** Use Voilà for interactive web app

---

## Credits

**Project:** Online Marketplace Simulation with Game Theory & Social Networks  
**Date:** November 11, 2025  
**Framework:** Jupyter Notebook  
**Visualization:** Matplotlib, Seaborn  
**Analysis:** Pandas, NumPy, NetworkX  

---

## Questions?

If you encounter issues or want to customize further, check:
1. **Python scripts** in Task1-4 folders (source of truth)
2. **Markdown files** (NASH_EQUILIBRIUM_FINAL_REPORT.md, etc.)
3. **Generated reports** (nash_equilibrium_report.txt, etc.)

---

**Enjoy your presentation! 🎉**

