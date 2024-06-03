## Main parameters for generation

1) k: number of clusters 
2) labels extraction on clustered data
    - Clustering can be either a) Single sample, or b) across multiple samples
    - Multiple samples requires consistency in clustered regions
    - Clustering can exclude labels or include labels
3) Means / std extraction
    - Can do random sampling or can extract means
    - keep_strictly_positive = keeps sampled means and std positive or negative
4) Sampling
    - Scaling
    - Distortion