# thesis direction

At this point, I need to finally determine what will be included in
my thesis, and how it will be glued together.

## Overall layout by chapter, section, subsection

**Introduction**

**Background**

 1. NLDAS and Noah-LSM: History and Implementation
 2. Distinctions in Modeling Techniques
    1. Noah-LSM as a Discrete Dynamical System
    2. Process-based vs Data-driven Models
 3. Deep Learning of Time Series

**Data and Methodology**

 1. Dataset Overview
    1. Data Storage System
    2. Regional Variability of Input Data
    3. Handling Snow
    4. Input Data Value Distributions
    5. Soil Moisture Distribution and Metrics
 2. Model Architectures
    1. Prediction Target: Increment Not State
    2. Naive RNN Failure
    3. Accumulating Model Architectures
 3. Training Paradigm
    1. Sporadic Sampling Strategy
    2. Learning Rate Schedule
    3. Loss Function Modifications (norming, biasing, error balance)
    4. Selecting Hyperparameters
    5. Training Procedure and `tracktrain` Framework
 4. Evaluation System
    1. Gridded and sequence-based evaluation
    2. Efficiency Metrics
    3. MAE and Bias Histograms

**Results**

 1. Exploratory Model Runs
 2. Best Models' Bulk Statistics Comparison
 3. Spatial, Temporal, and Situational Evaluation
 4. Parameter Variations
 5. Case Studies

**Future Work**

Selective sampling or sample weighting, process-specific estimators,


**Conclusion**

## ToDo

**Data and Methodology**

 - Elaborate on soil moisture regions and their unique
   characteristics, ie causes of high/low stdev, means
 - Flesh out sections 3 and 4 in their entirety

**Results**

 - Present and discuss exploratory bulk metrics with tables including
   descriptions of how the parameters were varied.
 - Discuss the problem of high loss rates for early sequence steps,
   the different rates of error accumulation, etc w/ horizons
