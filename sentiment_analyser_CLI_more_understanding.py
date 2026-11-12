import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sentence = ["This will be my first time trying other flows", "This seems like a mid-sized coding option", "I don't feel good about this code"]

#converting them into tokens 
inputs = tokenizer(
    sentence, 
    padding = True,
    truncation=True,
    max_length = 10,
    return_tensors = "pt"
)
print(inputs)
print("Input Ids: ",inputs["input_ids"].shape)
print("Attention Mask: ",inputs["attention_mask"].shape)

# batching up the sentences
with torch.no_grad():
    output = model(**inputs)

#getting the raw score 
logits = output.logits
print(logits)

#converting the raw score into probability using a linear formula 
prob = torch.nn.functional.softmax(logits, dim = -1)
print(prob)

id2label = model.config.id2label  # e.g., {0: 'NEGATIVE', 1: 'POSITIVE'}

print("--- Final Predictions ---")
for i, text in enumerate(sentence): 
    print(f"Text : {text}")
    
    # Loop dynamically through all label outputs for this sentence
    for class_idx, score in enumerate(prob[i]):
        label_name = id2label[class_idx]
        percentage = score.item() * 100
        print(f" -> {label_name}: {percentage:.2f}%")
    print()
