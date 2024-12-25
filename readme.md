# Plant Leaf Analysis Project

This project aims to build an image classification model for detecting diseases in plant leaves. It combines:

-   A custom Convolutional Neural Network (CNN) for image feature extraction.
-   The OpenAI API (GPT-3.5 Turbo) for generating detailed textual explanations.
-   Structural similarity score and image highlighting for difference analysis.


## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set OpenAI API Key:** Make sure to create a environment variable called `OPENAI_API_KEY` with a valid API key from OpenAI platform
3.  **Prepare Data:** Place your dataset in a directory named dataset. The dataset must be split into `train`, `val`, and `test` folders inside the main directory of `dataset/Apple_split` (can be changed by specifying it in code). Inside these directories, create subdirectories for each class that contain the relevant images.
4.  **Train the Model:** Run `train.py` to train your model and save weights in `/models/`.
    ```bash
    python train.py
    ```
5.  **Run the Flask app:** Start the web app using `app.py`.
    ```bash
    python app.py
    ```
6.  **Access the app:** Go to `http://127.0.0.1:5000` in a web browser and use the app.
7. **Upload and Test:** You can upload images for analysis using the form. You can test a single image using the `/upload` route. You can also compare two images and see the highlighted difference with the `/compare` route. Or, you can also test the model prediction and the explanations with the `/test_results` route.

## Key Code Files

*   **`train.py`**: Contains a custom CNN for image classification. You can adjust the learning rate, number of epochs, or model architecture inside this file.
*   **`app.py`**: Contains Flask application which loads model weights using custom classes, and performs prediction and analysis.
