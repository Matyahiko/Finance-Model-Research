from transformers import AutoTokenizer, AutoModel

model_name = "intfloat/e5-mistral-7b-instruct"
output_dir = "models/e5-mistral-7b-instruct"

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)  
tokenizer.save_pretrained(output_dir)

# Save model
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(output_dir)

# Now you can load from the saved directory
tokenizer = AutoTokenizer.from_pretrained(output_dir)  
model = AutoModel.from_pretrained(output_dir)