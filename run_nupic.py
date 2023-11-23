# Import NuPIC Client
import csv
import time
from nupic.client.inference_client import ClientFactory

# Define the URL and model name
URL = "localhost:8000"
MODEL = "numenta-sbert-2-v2-wtokenizer"

# Create a client instance
client = ClientFactory.get_client(MODEL, URL, "http")

# Get the embedding and measure time
start_time = time.time()
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    # Skip the header
    next(reader)
    # Iterate through the sentences
    for row in reader:
        sentences = [row[0]]
        embedding = client.infer(sentences)
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

print("Execution Time (NuPIC SBERT):", execution_time, "seconds")
