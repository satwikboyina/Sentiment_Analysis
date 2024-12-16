# Sentiment Analysis of Restaurant Reviews

## Overview
This project focuses on performing sentiment analysis of restaurant reviews to help the client understand customer behavior and feedback. The primary goal was to uncover actionable insights that would enable the client to identify areas for improvement and enhance the overall customer experience.

## Objectives
The sentiment analysis aimed to:
- **Understand Customer Behavior**: Analyze feedback to gauge customer sentiment (positive, negative, neutral) about various aspects of the dining experience.
- **Identify Improvement Areas**: Pinpoint key issues related to:
  - **Food Quality**: Complaints about taste, freshness, or variety.
  - **Hygiene**: Concerns about cleanliness of dining areas, utensils, or the kitchen.
  - **Staff Behavior**: Feedback on politeness, responsiveness, or efficiency of staff members.
  - **Hospitality**: General satisfaction with ambiance, service, and overall experience.

## Methodology
- **Natural Language Processing (NLP)**: Applied advanced NLP techniques to classify reviews into sentiment categories and extract recurring themes from the text data.
- **Topic Modeling**: Identified and grouped common issues raised by customers.
- **Insights Generation**: Generated actionable recommendations based on sentiment trends and customer priorities.

## Deliverables
- **Sentiment Classification**: Categorized reviews as positive, negative, or neutral.
- **Thematic Insights**: Highlighted common pain points in food quality, hygiene, staff behavior, and hospitality.
- **Improvement Suggestions**: Provided data-driven recommendations for enhancing service quality.

## Impact
By leveraging sentiment analysis, the client gained:
- A deeper understanding of customer preferences and pain points.
- A clear roadmap for addressing critical issues and improving operations.
- Enhanced decision-making capabilities, driven by customer-centric data insights.

## Technologies Used
- **Programming Language**: Python
- **Libraries/Frameworks**: NLTK, spaCy, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Storage/Processing**: Pandas, NumPy

---

### Creating the virtual Environment
python -m virtualenv flask or python -m venv flask
### Install the packages
pip install -r requirements.txt
### Activate environment in git bash
source ./virtualname/Scripts/activate

# Data Ingestion Module

## Overview
The Data Ingestion module automates the process of downloading and extracting datasets. It consists of two primary functions:

- **`download_file()`**: Downloads a dataset from a specified Google Drive URL and saves it to a local path.
- **`extract_zip_file()`**: Extracts the contents of the downloaded `.zip` file into a designated directory.

Configuration details, such as the source URL, local file path, and extraction directory, are managed using a configuration object. This module streamlines the initial steps of data preparation for further processing.

# Data Transformation Module

## Overview
The Data Transformation module processes and cleans text data for sentiment analysis. It provides various text preprocessing methods and handles data labeling, shuffling, loading, and saving.

## Functions

- **`removePuntuations(text)`**: Removes punctuation marks from the input text.
- **`removeStopwords(text)`**: Removes English stopwords from the input text.
- **`lemmatizeText(text)`**: Lemmatizes the words in the input text.
- **`chatConversion(text)`**: Converts chat abbreviations to their full forms based on a predefined dictionary.
- **`decodeEmoji(text)`**: Converts emojis in the input text to their descriptive text.
- **`correctText(text)`**: Corrects spelling in the input text using TextBlob.
- **`labelling(sentiment)`**: Converts sentiment labels (`positive`, `negative`, `1`, `0`) to a standard numeric format.
- **`shuffle(sentiData)`**: Randomly shuffles the rows of the input DataFrame multiple times.
- **`load()`**: Loads sentiment data from an Excel file.
- **`saveToExcel(sentiData)`**: Saves the processed data to an Excel file at the specified path.

## Usage
```python
from Sentiment.entity.config_entity import DataTransformationConfig
from Sentiment.data_transformation import DataTransformation

config = DataTransformationConfig(
    data_path="path/to/input_data.xlsx",
    save_path="path/to/output_data.xlsx"
)

data_transformation = DataTransformation(config)

# Example usage
data = data_transformation.load()
data['clean_text'] = data['text'].apply(data_transformation.removePuntuations)
data = data_transformation.shuffle(data)
data_transformation.saveToExcel(data)
```

# Data Validation Module

## Overview
The Data Validation module ensures that all required files are present in the specified directory. It performs file validation and logs the validation status.

## Functionality

- **`validate_all_files_exist()`**: 
  - Checks whether all required files, as specified in the configuration, exist in the `artifacts/data_ingestion` folder.
  - Writes the validation status (`True` or `False`) to a status file.

## Usage
```python
from Sentiment.entity.config_entity import DataValidationConfig
from Sentiment.data_validation import DataValiadtion

config = DataValidationConfig(
    ALL_REQUIRED_FILES=["file1.csv", "file2.csv"],
    STATUS_FILE="artifacts/validation_status.txt"
)

data_validation = DataValiadtion(config)

# Validate files
validation_status = data_validation.validate_all_files_exist()
print(f"Validation Status: {validation_status}")
```

# Model Evaluation Module

## Overview
The Model Evaluation module is responsible for evaluating the performance of a trained machine learning model. It includes functions for loading a pre-trained model, predicting test sentences, converting text to numeric format, calculating evaluation metrics, logging results to MLflow, and storing the evaluation status.

## Functions

- **`loadModel()`**: Loads the pre-trained model from the artifacts folder.
- **`predictTest(testSentences, model)`**: Uses the trained model to make predictions on the test data.
- **`sentimentClassify(predictions)`**: Classifies predictions as 1 (positive sentiment) or 0 (negative sentiment) based on a threshold.
- **`TextToNumeric(testSentences)`**: Converts the input text into numeric sequences using a tokenizer and pads the sequences to a specified length.
- **`log_into_mlflow(accuracy, model)`**: Logs the model and evaluation metrics (such as accuracy) into MLflow for tracking.
- **`EvaluationMetrics(predictions, testLabels)`**: Calculates accuracy score based on model predictions and test labels.
- **`ModelEvaluationStatus(accuracyScore)`**: Stores the evaluation metrics in a text file and returns the evaluation status (`True` or `False`).

## Usage
```python
from Sentiment.entity.config_entity import ModelEvaluationConfig
from Sentiment.model_evaluation import ModelEvaluation

config = ModelEvaluationConfig(
    model="path/to/model",
    mlflow_uri="your_mlflow_uri",
    METRIC_FILE="path/to/metrics.txt",
    all_params={"param1": "value1", "param2": "value2"}
)

model_evaluation = ModelEvaluation(config)

# Load model
model = model_evaluation.loadModel()

# Convert text to numeric
test_data = ["Sample sentence for testing"]
test_padded = model_evaluation.TextToNumeric(test_data)

# Predict test data
predictions = model_evaluation.predictTest(test_padded, model)

# Evaluate metrics
accuracy = model_evaluation.EvaluationMetrics(predictions, test_labels)

# Log evaluation results into MLflow
model_evaluation.log_into_mlflow(accuracy, model)

# Check evaluation status
evaluation_status = model_evaluation.ModelEvaluationStatus(accuracy)
print(f"Evaluation Status: {evaluation_status}")
```
# Model Trainer Module

## Overview
The Model Trainer module is responsible for training a machine learning model using sentiment analysis data. It includes functions for loading and preprocessing data, splitting data into train and test sets, converting text data into numeric format, preparing the model, training it, and saving the model.

## Functions

- **`loadData()`**: Loads the sentiment data from an Excel file, drops unnecessary columns, and handles missing values.
- **`convertToLower(data)`**: Converts the review text to lowercase for uniformity.
- **`splitData(sentiData)`**: Splits the dataset into training and test sets, ensuring stratification based on sentiment.
- **`TextToNumeric(trainSentences, testSentences)`**: Converts the review text to numeric sequences using the `Tokenizer` and pads them to a fixed length.
- **`prepareModel()`**: Prepares and compiles the LSTM-based deep learning model using the specified configuration parameters.
- **`trainModel(model, trainPadded, trainLabels)`**: Trains the model using the provided training data for a specified number of epochs.
- **`saveModel(model)`**: Saves the trained model to disk in the specified directories.

## Usage
```python
from Sentiment.entity.config_entity import ModelTrainerConfig
from Sentiment.model_trainer import ModelTrainer

config = ModelTrainerConfig(
    data_path="path/to/data.xlsx",
    vocab_size=3000,
    oov_tok="<OOV>",
    max_length=200,
    padding_type='post',
    embedding_dim=100,
    units=64,
    dense_layers=64,
    last_layer=1,
    loss='binary_crossentropy',
    optimizer='adam',
    metrics='accuracy',
    num_epochs=10,
    validation_split=0.2,
    verbose=1,
    root_dir="path/to/save/model",
    model_ckpt="path/to/save/checkpoints"
)

model_trainer = ModelTrainer(config)

# Load and preprocess data
sentiData = model_trainer.loadData()
sentiData = model_trainer.convertToLower(sentiData)

# Split data into train and test sets
trainSentences, testSentences, trainLabels, testLabels = model_trainer.splitData(sentiData)

# Convert text data to numeric sequences
trainPadded, testPadded = model_trainer.TextToNumeric(trainSentences, testSentences)

# Prepare the model
model = model_trainer.prepareModel()

# Train the model
model = model_trainer.trainModel(model, trainPadded, trainLabels)

# Save the model
model_trainer.saveModel(model)
```

