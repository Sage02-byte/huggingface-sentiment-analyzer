import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load our trusted checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

def analyze_sentiment(texts):
    print("--- Running Sentiment Analysis ---")
    
    # Tokenize inputs (with padding and truncation for safety)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Feed numbers into the model brain
    outputs = model(**inputs)
    
    # Convert raw logits to probabilities using softmax with dim=-1
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # DistilBERT SST-2 labels: 0 = NEGATIVE, 1 = POSITIVE
    labels = ["NEGATIVE", "POSITIVE"]
    
    for i, text in enumerate(texts):
        probs = probabilities[i]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100
        
        print(f" Text: '{text}'")
        print(f" Prediction: {labels[pred_idx]} ({confidence:.2f}% confidence)\n")

if __name__ == "__main__":
    my_sentences = [
        "I absolutely love learning how transformers work under the hood!",
        "This bug has been crashing my code for an hour and I hate it.",
        "It's an okay model, nothing too special."
    ]
    analyze_sentiment(my_sentences)

print(analyze_sentiment)








    
