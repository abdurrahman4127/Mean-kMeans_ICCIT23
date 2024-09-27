# Mean-KMeans [ICCIT23]
# README for 

## Overview

This repository contains an implementation of the **Mean-KMeans** clustering algorithm, an enhanced version of the traditional KMeans algorithm. Our Mean-KMeans algorithm improves clustering performance by optimizing the iteration process and utilizing in-place mean calculations for cluster assignments, thus resulting in faster convergence.

## Mean-KMeans Algorithm Details

Mean-KMeans introduces key enhancements:

- **Dynamic Mean Calculation**: Instead of fixed centroids, the algorithm calculates the mean of points currently assigned to each cluster during each iteration.
- **Optimized Convergence**: By refining the centroid selection and leveraging in-place calculations, the algorithm significantly reduces the number of iterations needed for convergence.


## Installation

To run the code, ensure you have Python installed. You can create a virtual environment for better package management. Use the following commands to set up your environment:

```bash
python3 -m venv venv
source venv/bin/activate
```
And install **numpy**, **matplotlib**, **scikit-learn**, and **pandas** following:
```
pip install numpy matplotlib scikit-learn pandas
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/abdurrahman4127/Mean-k-Means-ICCIT23.git
   ```
   Then navigate to the repository directory

2. Run the scripts:
   
   - To execute the KMeans clustering algorithm:

     ```bash
     python normal_kmeans.py
     ```

   - To execute the Mean-KMeans clustering algorithm:

     ```bash
     python mean_kmeans.py
     ```

   - To visualize the Elbow method and clustering results, run:

     ```bash
     python elbow_method.py
     ```

3. Notebook:
   Notebook is provided at [Code]()


## Citation

This study was published in 2023 at the 26th **International Conference on Computer and Information Technology (ICCIT)**, held from December 13 to 15 in Cox’s Bazar, Bangladesh. If you find our work useful, consider citing our work:

``` bash
@INPROCEEDINGS{10441078,
  author={Hasan, Emam and Rahman, Md. Abdur and Shojib Talukder, MD. and Utsho, Md Farnas and Shakhan, Md. and Farid, Dewan Md.},
  booktitle={2023 26th International Conference on Computer and Information Technology (ICCIT)}, 
  title={Data Segmentation with Improved K-Means Clustering Algorithm}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Technological innovation;Clustering methods;Clustering algorithms;Machine learning;Partitioning algorithms;Unsupervised learning;Convergence;Leaning by Observation;Partition-based Clustering;Unsupervised Learning},
  doi={10.1109/ICCIT60459.2023.10441078}}
```