# LSTM Model Deployment

This project deploys a trained LSTM model with an attention mechanism using Flask. The model is designed for making predictions based on input data.

## Project Structure

```
lstm-model-deployment
├── src
│   ├── app.py                  # Main application script
│   └── model
│       └── best_lstm_attention_autoencoder.h5  # Saved LSTM model
├── requirements.txt            # Project dependencies
├── Procfile                    # Deployment commands
├── runtime.txt                 # Python version
└── README.md                   # Project documentation
```

## Requirements

To run this project, you need the following dependencies:

- Flask
- TensorFlow

These dependencies are listed in the `requirements.txt` file.

## Deployment

### Heroku

1. Create a new Heroku app.
2. Push the code to Heroku using Git.
3. Ensure that the `Procfile` is correctly set up to run the Flask application.
4. Set the necessary environment variables if required.
5. Open the app in your browser.

### Render

1. Create a new web service on Render.
2. Connect your GitHub repository containing the project.
3. Set the build command to install dependencies from `requirements.txt`.
4. Set the start command to run the Flask application as specified in the `Procfile`.
5. Deploy the service.

## Usage

Once deployed, you can make predictions by sending a POST request to the appropriate endpoint defined in `app.py`. Ensure to provide the required input data in the request body.

## License

This project is licensed under the MIT License.