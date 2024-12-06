# Data-Driven Particle Position Estimation

This project explores a machine learning approach to estimate particle positions in a two-dimensional space using signals from a Resistive Silicon Detector (RSD). The pipeline leverages regression models to deduce positions based on signal data collected during particle events.

## 📋 Table of Contents

1. [🔍 Problem Overview](#-problem-overview)  
2. [📂 Dataset](#-dataset)  
4. [🛠️ Proposed Approach](#-proposed-approach)  
5. [📊 Results](#-results)  
6. [⚡ Limitations & Future Developments](#-limitations--future-developments)

## 🔍 Problem Overview
 
Particle position estimation is a critical task in particle physics and involves detecting particles using an RSD. The dataset consists of events, each corresponding to a particle crossing the sensor and generating signals on 12 metallic pads. These signals include metrics like positive and negative peaks, time delay, area, and RMS values. However, the data also contains noise, making accurate predictions challenging.

## 📂 Dataset

The dataset includes readings for 18 features per event (derived from hardware limitations), but only 12 pads are directly relevant. Key features:
	•	Pmax: Positive peak magnitude (mV)
	•	Negpmax: Negative peak magnitude (mV)
	•	Tmax: Time delay for the positive peak (ns)
	•	Area: Area under the signal
	•	RMS: Root mean square value

Outlier detection and feature correlation analysis were performed to refine the dataset.

## 🛠️ Proposed Approach

1. Preprocessing

	•	Feature Selection: Removed noisy or redundant pads (e.g., Pads 0, 7, 12, 16, 17) based on correlation analysis.
	•	Normalization: Applied standard normalization to ensure consistent feature scaling.
	•	Dimensionality Reduction: Retained only highly correlated features (e.g., Pmax over Area).

2. Model Selection

Two tree-based models were employed:
	•	Random Forest Regressor
	•	Extra Trees Regressor

These models are robust against outliers and suitable for handling large datasets.

3. Hyperparameter Tuning

Key hyperparameters (e.g., number of estimators, maximum depth, min samples split) were optimized using Euclidean distance as the evaluation metric.

## 📊 Results

The project achieved significant improvements over naive solutions:
	•	Random Solution: Euclidean distance = 209.847
	•	Naive Solution: Euclidean distance = 8.897
	•	Our Model: Euclidean distance ≈ 4.734

## ⚡ Limitations & Future Developments

Limitations

•	Limited computational resources constrained hyperparameter optimization.
•	Lack of domain expertise impacted feature selection.

Future Developments

•	Incorporate Gradient Boosting and Neural Networks to explore nonlinear relationships.
•	Enhance feature engineering with better domain insights.
•	Utilize distributed computing for hyperparameter tuning.
