# ML-Pipelines-and-GridSearchCV


# ML Pipelines & GridSearchCV Lab

This repository contains a guided lab notebook that demonstrates how to build, evaluate, and optimize machine learning pipelines using `Scikit-Learn`, with an emphasis on `Pipeline` objects and `GridSearchCV` for hyperparameter tuning and model validation.

## About the Project

The lab is part of the Skills Network curriculum and covers essential techniques in structuring machine learning workflows. It showcases both simple and complex pipelines using synthetic and real-world datasets.

---

## Objectives

After completing this lab you will be able to:

- Build and evaluate a machine learning pipeline
- Implement GridSearchCV for hyperparameter tuning with crossvalidation
- Implement and optimize a complex classification pipeline using real-world data
- Extract feature importances from a trained pipeline

---

## Introduction
In machine learning workflows, the Pipeline class from Scikit-Learn is invaluable for streamlining data preprocessing and model training into a single, coherent sequence. A pipeline is essentially a sequence of data transformers that culminates with an optional final predictor. This structure enables seamless integration of preprocessing and predictive modeling, ensuring that the same data transformations applied during training are consistently applied to new data during prediction.

Each intermediate step in a pipeline must be a transformer, meaning it should implement both fit and transform methods. The final step, which is typically a predictive model, or estimator, only requires a fit method. The entire pipeline can be trained simultaneously using a method like GridSearchCV, resulting in self-contained predictor that can be used to make predictions on unseen data.

Importantly, the pipeline allows you to set the parameters of each of these steps using their names and parameter names connected by a double underscore `__`. For example, if a pipeline step is named `imputer` and you want to change its strategy, you can pass a parameter like `imputer__strategy='median'`. Additionally, steps can be entirely swapped out by assigning a different estimator or even bypassed by setting them to `'passthrough'` or `None`.

A major advantage of using a pipeline is that it enables comprehensive cross-validation and hyperparameter tuning for all steps simultaneously. By integrating the pipeline within GridSearchCV, you can fine-tune not only the model but also the preprocessing steps, leading to optimized overall performance. Pipelines are essential for scenarios where preprocessing involves estimators performing operations like scaling, encoding categorical variables, imputing missing values, and dimensionality reduction. Pipelines ensure these steps are reproducibly applied to both training and test data.

----

## 📂 Contents

| Section                   | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| **Pipeline Introduction** | Concept of pipelines, transformers, and estimators               |
| **Synthetic Data**        | Training KNN on Iris dataset via pipelines                       |
| **GridSearchCV**          | Applying cross-validation to tune hyperparameters                |
| **Real-World Data**       | Building a pipeline for a Random Forest model with preprocessing |
| **Feature Importance**    | Extracting feature importances from pipeline steps               |
---

## 📈 Sample Datasets

* **Iris Dataset** (via `sklearn.datasets.load_iris`)
* Additional real-world datasets (used in classification examples)

