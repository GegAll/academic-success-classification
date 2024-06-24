# Academic Success Classification

In this repository, you can find a step-by-step guide for beginners, written by a beginner (me), modeling for the Kaggle Playground Competition "Classification with an Academic Success Dataset." This notebook covers various feature engineering techniques and hyperparameter tuning (with necessary visualizations) to help you practice and get inspired on your machine learning journey.

The datasets used can be found in the `data` folder as `test.csv` and `train.csv`, which were downloaded from [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e6/data).

## Project Overview

This project aims to build a classification model to predict academic success based on various features from the dataset. The notebook guides you through the entire process, from data preprocessing to model evaluation.

## Key Steps in the Notebook

1. **Data Exploration and Preprocessing**:
   - Loading and inspecting the dataset.

2. **Feature Engineering**:
   - Deleting correlated features to improve model performance.
   - Selecting the most relevant features using statistical methods and domain knowledge.

3. **Model Training and Evaluation**:
   - Splitting the data into training and validation sets.
   - Training various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, Gradient Boosting).
   - Hyperparameter tuning using techniques like Grid Search and Random Search.
   - Evaluating model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

4. **Visualization**:
   - Visualizing the data distribution and relationships between features.
   - Plotting model performance metrics and feature importances.

## How to Use

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/academic-success-classification.git
   ```

2. Navigate to the project directory:
   ```sh
   cd academic-success-classification
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebook to see the step-by-step guide:
   ```sh
   jupyter notebook academic_success.ipynb
   ```

## Repository Structure
`academic_success.ipynb`: The main Jupyter notebook containing the step-by-step guide for the classification task.
`data/`: Directory containing the dataset files (train.csv and test.csv).
requirements.txt: List of required Python packages.

## Datasets
The datasets used in this project can be found in the data folder:

`train.csv`: Training dataset with features and target variable.
`test.csv`: Test dataset for evaluating model performance.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
