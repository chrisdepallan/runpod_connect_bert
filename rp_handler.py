import runpod
import torch
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Hugging Face model repo (replace with your actual repo)
MODEL_NAME = "chrisdepallan/ner-skills-distilbert"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, from_tf=True)

# Load id2tag mapping from Hugging Face
id2tag_url = f"https://huggingface.co/{MODEL_NAME}/resolve/main/id2tag.json"
response = requests.get(id2tag_url)

if response.status_code == 200:
    id2tag = response.json()  # Load JSON directly
    id2tag = {int(k): v for k, v in id2tag.items()}  # Ensure keys are integers
else:
    raise ValueError("Failed to load id2tag.json from Hugging Face")

print("Model, Tokenizer, and id2tag mapping loaded successfully!")

def predict_entities(text):
    """Predict named entities from the input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",        # PyTorch tensors
        truncation=True,            # Truncate if longer than max length
        padding="max_length",       # Pad sequences
        max_length=512              # Max sequence length
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class indices
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).numpy()[0]

    # Convert token IDs to actual words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Convert prediction indices to label names
    predicted_labels = [id2tag[idx] for idx in predictions]

    # Combine tokens and labels into a result
    result = list(zip(tokens, predicted_labels))
    return result

def process_input(input):
    """
    Execute the application code
    """
    text = input['text']
    entities = predict_entities(text)

    return {
        "entities": entities
    }

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    """
    This is the handler function that will be called by RunPod serverless.
    """
    return process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})