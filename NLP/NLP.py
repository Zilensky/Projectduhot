import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW

# Initialize the model and tokenizer
model_name = "onlplab/alephbert-base"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dummy data for example purposes
context = "The Eiffel Tower is in Paris."
question = "Where is the Eiffel Tower located?"

# Tokenize the input data
inputs = tokenizer(question, context, return_tensors='pt')
print("Tokenized inputs:", inputs)

# Extract input_ids, token_type_ids, and attention_mask
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention_mask = inputs['attention_mask']

# Dummy positions (use valid positions according to your actual data)
start_positions = torch.tensor([2])
end_positions = torch.tensor([6])

# Check the dimensions of the input tensors
print("Input IDs shape:", input_ids.shape)
print("Token Type IDs shape:", token_type_ids.shape)
print("Attention Mask shape:", attention_mask.shape)

# Ensure that positions are within bounds
if start_positions.item() >= input_ids.size(1) or end_positions.item() >= input_ids.size(1):
    raise ValueError("Start or end positions are out of bounds.")

# Define labels as a tuple of (start_positions, end_positions)
labels = (start_positions, end_positions)

# Define an optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training step (dummy example)
model.train()
outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                start_positions=start_positions, end_positions=end_positions)
loss = outputs.loss
loss.backward()
optimizer.step()

# Define the directory to save the model
save_directory = r"C:\Users\yairb\PycharmProjects\Projectduhot\NLP\saved_model"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save the model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Save the optimizer state
optimizer_save_path = os.path.join(save_directory, 'optimizer.pth')
torch.save(optimizer.state_dict(), optimizer_save_path)

print("Model and optimizer saved successfully.")
