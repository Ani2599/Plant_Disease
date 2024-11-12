# Plant Disease Classifier

This project is a **Streamlit** web application that uses a pre-trained **TensorFlow** model to classify plant diseases based on images. The model predicts the disease class from an uploaded plant image and displays the result on the app interface.

## Features

- **Plant Disease Prediction**: Upload a plant image, and the model will classify it into one of the predefined disease categories.
- **Interactive Interface**: Built with Streamlit, the app provides a user-friendly interface to upload images and view predictions.
- **Model and Class Names Loading**: The TensorFlow model and class indices are loaded from local files for prediction.

## How It Works

1. **Upload Image**: Users upload an image of a plant leaf.
2. **Image Preprocessing**: The image is resized, normalized, and prepared for input into the model.
3. **Prediction**: The pre-trained model predicts the class of the plant disease.
4. **Output**: The class of the disease is displayed on the app.

## Project Structure

- `app.py`: The main file containing the Streamlit app code.
- `trained_model/plant_disease_prediction_model.h5`: The pre-trained TensorFlow model file.
- `class_indices.json`: A JSON file containing the class names and indices.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-classifier.git
   cd plant-disease-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the model (`plant_disease_prediction_model.h5`) and `class_indices.json` files in the `trained_model` directory.

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the app with `streamlit run app.py`.
2. Upload a plant image by clicking on "Upload an image...".
3. Click on the **Classify** button to see the predicted disease class for the uploaded image.

## Code Explanation

- **Image Preprocessing**: The `load_and_preprocess_image` function loads, resizes, and scales the image to prepare it for the model.
- **Prediction**: The `predict_image_class` function takes the preprocessed image and uses the model to predict the disease class.
- **Streamlit Interface**: The Streamlit code sets up an interactive interface for uploading images and displaying predictions.

## Dependencies

- `tensorflow`: For loading and running the pre-trained model.
- `streamlit`: For building the interactive web interface.
- `numpy`: For numerical operations.
- `Pillow`: For image loading and preprocessing.
- `json`: For loading class indices.

## Example

1. **Input**: Upload an image of a plant leaf.
2. **Output**: The app displays the predicted disease class.

## Future Enhancements

- **Expand Model Capabilities**: Use a more robust model trained on a larger dataset of plant diseases.
- **Confidence Score**: Display the model's confidence level for each prediction.
- **Additional Features**: Allow users to upload multiple images at once or provide more detailed information about each disease.

## License

This project is open-source and available under the MIT License.

Kaggle Dataset Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

