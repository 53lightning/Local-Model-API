from flask import Flask, request, jsonify
from transformers import pipeline
import torch

# all possible models for inference and their repos
possible_models = {
    "1": ("Meta Llama 3.1 8b", "meta-llama/Meta-Llama-3.1-8B"),
    "2": ("Meta Llama 3.1 8b Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "3": ("Meta Llama 3.1 70b", "meta-llama/Meta-Llama-3.1-70B"),
    "4": ("Meta Llama 3.1 70b Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct")
} 

# function to get the users preferred model
def get_user_model(possible_models: dict[str, tuple[str, str]]):

    print("Please choose one of the following options:\n")
    for key, (name, path) in possible_models.items():
        print(f"{key}: {name}")

    while True:
        choice = input("\nEnter the number of your choice: ")

        if choice in possible_models:
            print(f"\nYou selected: {possible_models[choice][0]}")
            return possible_models[choice][1]
        else:
            print("\nInvalid choice. Please try again.")

# getting the model path, or repo in this case            
model_path = get_user_model(possible_models)

# setting up the pipeline for text generation
llm = pipeline(
    "text-generation",
    model = model_path,
    model_kwargs = {"torch_dtype": torch.bfloat16},
    device_map = "auto"
)

# initialize flask app
app = Flask(__name__)

# setup app functionality
@app.route('/inference', methods = ['POST'])
def inference():
    data = request.json # retrieve the users request
    llm_rsp = llm(data['text'], max_length = data.get('max_length', 512), temperature = data.get('temperature', 0.0), truncation = False)[0]["generated_text"] # call the llm
    return jsonify({"response": llm_rsp}) # return the response

# starting server
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000)