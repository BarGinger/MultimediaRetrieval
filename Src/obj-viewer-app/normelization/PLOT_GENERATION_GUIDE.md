# Validation Plot Generation - Complete Package

## 📦 What's Included

This package provides comprehensive validation plotting capabilities for your normalization pipeline. Three Python scripts work together to generate, preview, and manage publication-quality figures for your report.

### Scripts

1. **`generate_validation_plots.py`** - Main plot generator (creates 6 comprehensive figures)
2. **`run_full_workflow.py`** - Automated workflow (runs normalization + plotting)
3. **`preview_plots.py`** - Plot viewer (preview generated figures)

## 🚀 Quick Start

### Option 1: Full Workflow (Recommended for first run)
```bash
python run_full_workflow.py
```
This will:
1. Run the normalization pipeline (if not already done)
2. Generate all 6 validation figures
3. Display summary and next steps

### Option 2: Just Generate Plots (if normalization is complete)
```bash
python generate_validation_plots.py
```

### Option 3: Preview Existing Plots
```bash
python preview_plots.py
```

## 📊 Generated Figures

All figures are **300 DPI PNG** files, ready for publication:

### Figure 1: Validation Overview Panel (`fig1_validation_overview.png`)
- **Purpose**: Executive summary of all validation metrics
- **Subplots**: 
  - Centering error distribution (log scale)
  - Scaling error distribution (log scale)
  - PCA alignment quality histogram
  - Success rate by top 20 categories
  - Statistical summary box
- **Key Insights**: Overall success rate, error compliance, per-category performance

### Figure 2: Step-by-Step Progression (`fig2_step_progression.png`)
- **Purpose**: Track how metrics evolve through the pipeline
- **Subplots**:
  - Barycenter drift (shows centering is maintained)
  - Vertex displacement per transformation
  - Bounding box convergence to unit cube
  - Success rates per validation criterion
- **Key Insights**: Cross-step consistency, transformation magnitudes

### Figure 3: PCA Quality Analysis (`fig3_pca_analysis.png`)
- **Purpose**: Assess alignment quality and numerical stability
- **Subplots**:
  - Eigenvalue ratio scatter (λ₁/λ₂ vs λ₂/λ₃)
  - Condition number distribution
  - Anisotropy score histogram
  - Eigenvector-to-axis alignment boxplots
  - Condition number by category
  - Eigenvalue ordering verification pie chart
- **Key Insights**: PCA numerical stability, alignment quality, shape complexity

### Figure 4: Moment Test and Symmetry (`fig4_moment_symmetry.png`)
- **Purpose**: Validate flipping step and analyze shape symmetry
- **Subplots**:
  - Moment value distributions after flipping
  - Moment sign consistency
  - Flipping frequency by axis
  - Symmetry classification (spherical/cylindrical/asymmetric)
  - Dimension ratio scatter
  - Moment magnitude by symmetry type
- **Key Insights**: Flipping effectiveness, shape symmetry patterns

### Figure 5: Numerical Precision (`fig5_numerical_precision.png`)
- **Purpose**: Demonstrate numerical robustness and precision
- **Subplots**:
  - Centering error CDF with threshold markers
  - Scaling error CDF with threshold markers
  - Two-pass recentering requirements
  - Error threshold compliance bar chart
  - Aspect ratio preservation
  - Precision summary table
- **Key Insights**: Compliance with target thresholds, numerical robustness

### Figure 6: Category Performance Heatmap (`fig6_category_heatmap.png`)
- **Purpose**: Detailed per-category performance analysis
- **Content**: Heatmap showing success rates for:
  - Centering (< 10⁻¹⁰ threshold)
  - Scaling (< 10⁻⁶ threshold)
  - Alignment (quality > 0.9)
  - Flipping (moment test success)
  - Overall normalization success
- **Key Insights**: Category-specific challenges, consistent performance

## 📝 Using Figures in Your Report

### LaTeX Integration

```latex
% In your preamble
\usepackage{graphicx}

% In your document
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/fig1_validation_overview.png}
    \caption{Comprehensive validation overview of the normalization pipeline. 
             (a) Centering errors are well below the $10^{-10}$ target threshold
             with mean $9.49 \times 10^{-15}$. (b) Scaling errors meet the 
             $10^{-6}$ target with mean $3.30 \times 10^{-17}$. (c) PCA alignment
             achieves high quality scores (mean 0.95). (d) Category-specific 
             success rates exceed 95\% for most categories. Overall pipeline 
             success rate: 99.8\%.}
    \label{fig:validation_overview}
\end{figure}
```

### Referencing in Text

```latex
As shown in Figure~\ref{fig:validation_overview}, the normalization pipeline
achieves exceptional precision, with centering errors averaging 
$9.49 \times 10^{-15}$, well below the target threshold of $10^{-10}$.
```

## 🎯 Key Statistics to Highlight

Based on your validation data, emphasize these findings in your report:

1. **Overall Success Rate**: 99.8% (2,478 out of 2,483 shapes)

2. **Centering Precision**:
   - Mean error: $9.49 \times 10^{-15}$
   - Max error: $7.53 \times 10^{-13}$
   - 100% below $10^{-10}$ threshold

3. **Scaling Precision**:
   - Mean error: $3.30 \times 10^{-17}$
   - Max error: $2.22 \times 10^{-16}$
   - 100% below $10^{-6}$ threshold

4. **PCA Alignment**:
   - Mean quality score: 0.95
   - 100% correct eigenvalue ordering

5. **Numerical Robustness**:
   - Two-pass recentering triggered for [X]% of shapes
   - All shapes achieve machine epsilon precision

## 🔧 Customization

### Changing Output Directory
```bash
python generate_validation_plots.py --output my_figures/
```

### Using Different Validation Data
```bash
python generate_validation_plots.py --input path/to/validation_detailed.json
```

### Listing Plots Without Preview
```bash
python preview_plots.py --list
```

## 📐 Figure Dimensions and Resolution

All figures are optimized for academic publications:

- **Resolution**: 300 DPI
- **Format**: PNG (high quality, widely compatible)
- **Figure 1**: 12" × 8" (multi-panel overview)
- **Figure 2**: 12" × 8" (step progression)
- **Figure 3**: 14" × 8" (PCA analysis, 6 subplots)
- **Figure 4**: 14" × 8" (moment test, 6 subplots)
- **Figure 5**: 14" × 8" (precision, 6 subplots)
- **Figure 6**: 10" × variable (heatmap, scales with categories)

### File Sizes
Typical file sizes at 300 DPI:
- Figures 1-5: ~1-2 MB each
- Figure 6: ~0.5-1.5 MB (depends on category count)

## 🎨 Plot Style Guidelines

The plots follow academic publication standards:

✅ **Best Practices Used**:
- Serif fonts for professional appearance
- High-contrast colors for print clarity
- Clear axis labels with units
- Grid lines for value reading
- Statistical annotations (means, thresholds)
- Color-blind friendly palettes where applicable
- Consistent styling across all figures

✅ **LaTeX-Ready**:
- Fonts match typical LaTeX documents
- Sizes appropriate for column/page width
- Caption-friendly layout (subplots labeled a, b, c, etc.)

## 🐛 Troubleshooting

### "Input file not found"
**Solution**: Run `normalize_database.py` first to generate validation data

### "No module named 'seaborn'"
**Solution**: Install required packages:
```bash
pip install matplotlib seaborn numpy pandas scipy
```

### "Figure window not responding"
**Solution**: Use `preview_plots.py --list` to skip interactive viewing

### Empty or missing subplots
**Cause**: Some validation metrics may not be present for all shapes
**Solution**: Re-run normalization with updated `normalize_database.py` that includes all enhanced validation code

### Low memory during generation
**Solution**: Process figures individually by commenting out unwanted plots in `generate_all_plots()`

## 📚 Additional Resources

### Understanding the Metrics

- **Centering Error**: Distance of barycenter from origin after normalization
- **Scaling Error**: Deviation of max bounding box dimension from 1.0
- **Alignment Quality**: Mean dot product of eigenvectors with coordinate axes
- **Condition Number**: Ratio of largest to smallest eigenvalue (numerical stability)
- **Anisotropy Score**: Measure of shape elongation (0 = sphere, 1 = very elongated)

### Validation Criteria

| Metric | Target | Your Result |
|--------|--------|-------------|
| Centering | < 10⁻¹⁰ | ✅ 9.49 × 10⁻¹⁵ |
| Scaling | < 10⁻⁶ | ✅ 3.30 × 10⁻¹⁷ |
| Alignment | > 0.9 | ✅ 0.95 |
| Flipping | Success | ✅ 99.8% |

## 🎓 Report Writing Tips

1. **Introduction to Results Section**:
   - Start with Figure 1 (overview) to establish overall success
   - Reference specific metrics in text

2. **Detailed Analysis**:
   - Use Figures 2-5 to discuss specific aspects
   - Compare to technical requirements/literature

3. **Category-Specific Discussion**:
   - Use Figure 6 to identify challenging categories
   - Explain why certain shapes are harder to normalize

4. **Numerical Precision Emphasis**:
   - Highlight that errors are at machine epsilon level
   - Discuss two-pass recentering as a robustness feature

5. **Conclusion**:
   - Summarize overall success rate (99.8%)
   - Emphasize compliance with all technical requirements

## 📞 Support

If you encounter issues:

1. Check that `validation_detailed.json` exists and is valid JSON
2. Verify all Python dependencies are installed
3. Review console output for specific error messages
4. Ensure sufficient disk space for figure generation (~10-15 MB total)

## 🎉 Quick Reference

```bash
# Full workflow (first time)
python run_full_workflow.py

# Regenerate plots only
python generate_validation_plots.py

# Preview plots
python preview_plots.py

# List available plots
python preview_plots.py --list

# Custom output location
python generate_validation_plots.py -o ../../../Reports/Step3/figures/
```

---

**Happy plotting! 📊📈📉**

These figures will provide strong visual evidence of your normalization pipeline's quality and precision in your academic report.
