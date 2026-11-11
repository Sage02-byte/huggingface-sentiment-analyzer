Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import torch
... from transformers import AutoTokenizer, AutoModelForSequenceClassification
... 
... # 1. Load checkpoint
... checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
... tokenizer = AutoTokenizer.from_pretrained(checkpoint)
... model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
... 
... # 2. Define a mixed-length batch of reviews/sentences
... sentences = [
...     "I have been waiting for this course my whole life, absolutely love it!",
...     "It was okay, but a bit confusing at times.",
...     "Terrible experience, completely broken and frustrating."
... ]
... 
... # 3. Tokenize with padding, truncation, and tensor conversion
... inputs = tokenizer(
...     sentences, 
...     padding=True, 
...     truncation=True, 
...     max_length=32, 
...     return_tensors="pt"
... )
... 
... print("--- Batch Tensor Shapes ---")
... print("Input IDs Shape:", inputs["input_ids"].shape)
... print("Attention Mask Shape:", inputs["attention_mask"].shape)
... 
... # 4. Run model inference using dictionary unpacking (**inputs)
... with torch.no_grad():
...     outputs = model(**inputs)
... 
... logits = outputs.logits
... 
... # 5. Convert logits to probabilities using softmax
... probabilities = torch.nn.functional.softmax(logits, dim=-1)
... 
print("\n--- Sentiment Results ---")
for i, text in enumerate(sentences):
    pos_prob = probabilities[i][1].item() * 100
    neg_prob = probabilities[i][0].item() * 100
    print(f"Text: '{text}'")
