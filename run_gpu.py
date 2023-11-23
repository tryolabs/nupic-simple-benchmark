# Imports
import csv
import time
from transformers import BertTokenizer, BertModel
import torch

device = torch.device("cuda") # or "cpu" for CPU

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model in evaluation mode
model = BertModel.from_pretrained("bert-large-uncased")
model.eval().to(device)

# Get the embedding and measure time
start_time = time.time()
with torch.no_grad():
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader)
        # Iterate through the sentences
        for row in reader:
            sentences = [row[0]]
            inputs = tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = inputs.to(device)
            outputs = model(**inputs)

end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

print("Execution Time (Hugging Face BERT):", execution_time, "seconds")
