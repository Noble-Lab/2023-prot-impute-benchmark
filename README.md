# Evaluating proteomics imputation methods with improved criteria

*Lincoln Harris, William E. Fondrie, Sewoong Oh, William S. Noble*        
**University of Washington**, Departments of Genome Sciences and Computer Science & Engineering       

Quantitative measurements produced by tandem mass spectrometry proteomics experiments typically contain a large proportion of missing values. 
This missingness hinders reproducibility, reduces statistical power, and makes it difficult to compare across samples or experiments. 
Although many methods exist for imputing missing values in proteomics data, in practice, the most commonly used methods are among the worst performing. 
Furthermore, previous benchmarking studies have focused on relatively simple measurements of error, such as the mean-squared error between the imputed and the held-out observed values. 
Here we evaluate the performance of a set of commonly used imputation methods using three practical, “downstream-centric” criteria, which measure the ability of imputation methods to reconstruct differentially expressed peptides, identify new quantitative peptides, and improve peptide lower limit of quantification. 
Our evaluation spans several experiment types and acquisition strategies, including data- dependent and data-independent acquisition. 
We find that imputation does not necessarily improve the ability to identify differentially expressed peptides, but that it can identify new quantitative peptides and improve peptide lower limit of quantification. 
We find that missForest is generally the best performing method per our downstream-centric criteria.
We also argue that existing methods do not properly account for the distribution of peptide quantifications.

You can find the manuscript [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley). 
