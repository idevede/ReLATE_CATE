# ReLATE_NeurIPS

This is the code of paper "Representation Learning for Treatment Effects Estimation in Heterogeneous Environments" that we submitted to NeurIPS. In the spirit of open research and transparency, we are committed to releasing all of our code, including the model architectures and hyperparameters, training scripts, evaluation metrics, and baseline implementations upon acceptance of our paper.


## Code Transparency 

1. The code in the `SpeedData_Exp` folder is used to reproduce Experiment 5.2 and Appendix F.2 in our paper. This includes:
    - The Mod 1 dataset used in the experiments
    - The proposed model architectures and hyperparameters
    - The full training and evaluation scripts 
    - Outputs collection from multiple runs to demonstrate reproducibility  

To run this code:
```
bash run_relate.sh
```

2. The Jupyter notebook file in the `Covid_Exp_Appendix` folder contains the code to reproduce the Covid-related experiments discussed in Appendix D. This includes:
    - Data processing steps
    - Model implementations
    - Full results and analyses
