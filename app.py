# filepath: app.py
from flask import Flask, request, jsonify
import fine_tuning_llama2

app = Flask(__name__)

@app.route('/adjust_parameters', methods=['POST'])
def adjust_parameters():
    data = request.json
    # Adjust parameters in fine_tuning_llama2.py
    fine_tuning_llama2.training_params.num_train_epochs = data.get('num_train_epochs', 1)
    fine_tuning_llama2.training_params.learning_rate = data.get('learning_rate', 2e-4)
    # Add more parameters as needed
    return jsonify({"status": "success", "num_train_epochs": fine_tuning_llama2.training_params.num_train_epochs, "learning_rate": fine_tuning_llama2.training_params.learning_rate})

if __name__ == '__main__':
    app.run(debug=True)