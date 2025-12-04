
# Introduction

This repository contains a JAX re-implementation of the Message-Passing Neural Network model introduced in K. Rusek, P. Chołda, *“Message-Passing Neural Networks Learn Little’s Law”*. The code is based on the authors’ original TensorFlow 1 implementation and was developed as part of the engineering thesis **“Reproducibility of results from paper Message-Passing Neural Networks Learn Little's Law implemented in JAX”** supervised by dr inż. Krzysztof Rusek.

The JAX implementation is provided in two APIs: **Flax Linen** and **Flax NNX**, allowing a direct comparison of modern JAX frameworks.

Below we present instructions on how to run our code and compare the results obtained by the original TensorFlow 1 model with the JAX implementations.

# How to run
Below you can find a detailed instruction on how to prepare the data, train the model, and test it.

## Generating samples

First, you need samples to train the model. To generate them, use the **samples.py** script, which creates three sets of samples:

-   training set

-   validation set (used during training)
    
-   test set (used after training)
    

The configuration for sample generation is located in the **config.py** file, with the following settings:

-   **output_dir** - directory where the generated samples will be saved
    
-   **seed** - RNG seed used during sample generation
    
-   **n_nodes** - number of nodes in the generated graphs
    
-   **train_size** — number of samples in the training set
    
-   **val_size** - number of samples in the validation set
    
-   **test_size** - number of samples in the test set
    
-   **rl** - lower bound for traffic load generation
    
-   **rh** - upper bound for traffic load generation
    
-   **graph_type** - graph type (BarabasiAlbert or ErdosRenyi)
    
-   **snd_path** - optional XML path to an SNDlib network, if samples should be generated from real topologies
    

## Training the model

Training is configured through the TrainingConfig class in **config.py**. The key parameters are:

-   **train_dataset_path** - directory containing the training samples
    
-   **val_dataset_path** - directory containing the validation samples
    
-   **output_path** - directory where the trained model will be saved
    
-   **batch_size** - number of graphs per batch
    
-   **learning_rate** - learning rate used during optimization
    
-   **seed** - RNG seed for training
    
-   **steps** - number of training steps
    
-   **log_interval** - how often validation is performed
    
-   **use_early_stopping** - whether to enable early stopping
    
-   **early_stopping_patience** - patience parameter for early stopping
    
-   **norm_profile** - normalization settings for μ and W depending on the graph model
    

After setting all training parameters, run one of the training scripts:

```
python train_linen.py
```

or
```
python train_nnx.py
```
  

## Testing the model

Before testing, configure the TestConfig section in **config.py**:

-   **test_dataset_path** - directory containing the test samples  


-   **checkpoint_path** - directory where the trained model is stored  
      
    
-   **output_path** - directory where test results will be saved  
      
    
-   **norm_profile** - same normalization settings used during training  



Run the desired test script:

```
python test_linen.py
```
Or
```
python test_nnx.py
```
  
## Test output

After testing, two plots will be saved in output_path:

### **Predicted vs. True Values**  

   A scatter plot showing how close the predicted labels are to the true labels.  
    Points aligned along the diagonal indicate accurate predictions.  
      
    
### **Histogram of Residuals**  
   A histogram showing the distribution of prediction errors (predicted - true).  
    A narrow peak near zero indicates low residuals and good model performance.

## Training curves

The plots below show how the loss decreases over time for both graph models (BA and ER) and both JAX APIs (Linen and NNX). Each curve is computed from raw MSE values logged during training.

### Barabási-Albert (BA) training

<table>
  <tr>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="plots/training_curves/training_curve_linen_ba.svg" width="450">
    </td>
    <td style="text-align:center;">
      <img src="plots/training_curves/training_curve_nnx_ba.svg" width="450">
    </td>
  </tr>
</table>

### Erdős-Rényi (ER) training

<table>
  <tr>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="plots/training_curves/training_curve_linen_er.svg" width="450">
    </td>
    <td style="text-align:center;">
      <img src="plots/training_curves/training_curve_nnx_er.svg" width="450">
    </td>
  </tr>
</table>
    
# Evaluation Results

Below we present the evaluation results for both JAX implementations, as well as TensorFlow 1 tested on multiple datasets, including synthetic random graphs and real topologies from **SNDlib**.

Each evaluation reports:

- **MSE**
- **R²**
- **Pearson correlation**

The training procedure followed the setup described in the original paper. Each model was trained on three separate datasets: **20 000** samples for training, **200** samples for validation during training, and **2 000** samples for testing. The training process consisted of **200 000** iterations, using the same hyperparameters for both APIs to ensure a fair comparison.


Throughout the results section, notation such as BA / ER indicates train -> test pairing:

- **BA / ER** - the model was trained on Barabási-Albert graphs and tested on Erdős-Rényi graphs

- **BA / SNDlib** - the model was trained on BA synthetic graphs and evaluated on real SNDlib topologies

and analogously for all other combinations

---

### BA / BA

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0069 |  0.9929 |  0.9974 |
| **JAX Linen**    |  0.0043 |  0.9954 |  0.9978 |
| **JAX NNX**      |  0.0054 |  0.9943 |  0.9975 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_ba/ba-ba1_white.png" width="450"></td>
    <td><img src="plots/linen/ba_ba/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_ba/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_ba/ba-ba2_white.png" width="450"></td>
    <td><img src="plots/linen/ba_ba/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_ba/residuals_hist.svg" width="450"></td>
  </tr>
</table>



---

### BA / ER

| Model            |    MSE |       R² | Pearson |
| :--------------- | -----: | -------: | ------: |
| **TensorFlow 1** | 11.5073 | -20.2154 |  0.8631 |
| **JAX Linen**    |  12.4450 |  -22.5724 | 0.7992 |
| **JAX NNX**      |  10.5444 | -18.9725 | 0.8056 |
<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_er/ba-er1_white.png" width="450"></td>
    <td><img src="plots/linen/ba_er/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_er/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_er/ba-er2_white.png" width="450"></td>
    <td><img src="plots/linen/ba_er/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_er/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### BA / ER ( ER )

| Model            |    MSE |       R² | Pearson |
| :--------------- | -----: | -------: | ------: |
| **TensorFlow 1** |  9.0614 |  -7.8326 | 0.7191 |
| **JAX Linen**    |  16.074 |  -15.0038  | 0.6738 |
| **JAX NNX**      |  7.9911 |  -7.0799  | 0.7079 |
<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_er_ernorm/eval.png" width="450"></td>
    <td><img src="plots/linen/ba_er(er)/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_er_ernorm/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_er_ernorm/residuals_hist.png" width="450"></td>
    <td><img src="plots/linen/ba_er(er)/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_er_ernorm/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### BA / germany50

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | -------: | ------: |
| **TensorFlow 1** |   2.2480 |  -4.7743 | 0.9433 |
| **JAX Linen**    |   2.0744 | -4.3719 | 0.9256 |
| **JAX NNX**      |   2.1572 |  -4.5864 | 0.8830 |

<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_germany50/ba-germany501_white.png" width="450"></td>
    <td><img src="plots/linen/ba_germany50/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_germany50/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_germany50/ba-germany502_white.png" width="450"></td>
    <td><img src="plots/linen/ba_germany50/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_germany50/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### BA / SNDlib

| Model            |     MSE |       R² | Pearson |
| :--------------- | ------: | -------: | ------: |
| **TensorFlow 1** |   1.4945 |  -2.3634 | 0.5122 |
| **JAX Linen**    |   1.5623 |  -2.2661 | 0.4808 |
| **JAX NNX**      |   1.9015 |  -2.9751 | 0.4662 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_sndlib/ba-sndlib1_white.png" width="450"></td>
    <td><img src="plots/linen/ba_sndlib/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_sndlib/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_sndlib/ba-sndlib2_white.png" width="450"></td>
    <td><img src="plots/linen/ba_sndlib/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_sndlib/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### ER / ER

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0188 |  0.9817 |  0.9943 |
| **JAX Linen**    |  0.0124 |  0.9876 |  0.9952 |
| **JAX NNX**      |  0.0143 |  0.9857 |  0.9935 |



<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er/er-er1_white.png" width="450"></td>
    <td><img src="plots/linen/er_er/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_er/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er/er-er2_white.png" width="450"></td>
    <td><img src="plots/linen/er_er/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_er/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### ER / BA

| Model            |    MSE |     R² | Pearson |
| :--------------- | -----: | -----: | ------: |
| **TensorFlow 1** | 0.1157 | 0.9371 | 0.9769 |
| **JAX Linen**    | 0.4211 | 0.7716 | 0.9124 |
| **JAX NNX**      | 0.7545 | 0.5908 | 0.8964 |



<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_ba/er-ba1_white.png" width="450"></td>
    <td><img src="plots/linen/er_ba/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_ba/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_ba/er-ba2_white.png" width="450"></td>
    <td><img src="plots/linen/er_ba/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_ba/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / ER60

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.1146 |  0.9244 |  0.9715 |
| **JAX Linen**    |  0.1338 |  0.9108 |  0.9633 |
| **JAX NNX**      |  0.0810 |  0.9460 |  0.9767 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er60/er-er601_white.png" width="450"></td>
    <td><img src="plots/linen/er_er60/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_er60/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er60/er-er602_white.png" width="450"></td>
    <td><img src="plots/linen/er_er60/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_er60/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / janos_us

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0206 | 0.9468 |  0.9893 |
| **JAX Linen**    |  0.0224 | 0.9415 | 0.9929 |
| **JAX NNX**      |  0.0113 | 0.9705 | 0.9897 |

<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us/er-janos-us1_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us/er-janos-us2_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / germany50

| Model            |     MSE |     R² | Pearson |
| :--------------- | ------: | -----: | ------: |
| **TensorFlow 1** |  0.1946 |  0.7374 | 0.9531 |
| **JAX Linen**    |  0.3144 |  0.5719 | 0.8872 |
| **JAX NNX**      |  0.4891 |  0.3341 | 0.8817 |




<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_germany50/er-germany501_white.png" width="450"></td>
    <td><img src="plots/linen/er_germany50/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_germany50/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_germany50/er-germany502_white.png" width="450"></td>
    <td><img src="plots/linen/er_germany50/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_germany50/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / cost266

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0350 | 0.9362 | 0.9872 |
| **JAX Linen**    |  0.0124 | 0.9768 | 0.9887 |
| **JAX NNX**      |  0.0148 | 0.9725 | 0.9879 |

<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_cost266/er-cost2661_white.png" width="450"></td>
    <td><img src="plots/linen/er_cost266/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_cost266/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_cost266/er-cost2662_white.png" width="450"></td>
    <td><img src="plots/linen/er_cost266/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_cost266/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / janos_us_ca

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0427 | 0.9259 | 0.9845 |
| **JAX Linen**    |  0.0198 | 0.9646 | 0.9849 |
| **JAX NNX**      |  0.0190 | 0.9660 | 0.9840 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us-ca/er-janos-us-ca1_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us-ca/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us-ca/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us-ca/er-janos-us-ca2_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us-ca/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us-ca/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### ER / SNDlib

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0725 | 0.9142 | 0.9776 |
| **JAX Linen**    |  0.0419 | 0.9538 | 0.9834 |
| **JAX NNX**      |  0.0515 | 0.9433 | 0.9768 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_sndlib/er-sndlib1_white.png" width="450"></td>
    <td><img src="plots/linen/er_sndlib/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_sndlib/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_sndlib/er-sndlib2_white.png" width="450"></td>
    <td><img src="plots/linen/er_sndlib/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_sndlib/residuals_hist.svg" width="450"></td>
  </tr>
</table>
