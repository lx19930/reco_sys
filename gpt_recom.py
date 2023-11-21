import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-medium'  # Choose the desired model size
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define a function to generate next words
def predict_next_word(sentence, top_k=5, top_p=0.9, max_length=20):
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    
    # Generate next words using the GPT-2 model
    output = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length + input_ids.size(-1),
        top_k=top_k,
        top_p=top_p,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    
    # Decode the generated output
    predicted_words = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return predicted_words

# Example sentence for prediction
input_sentence = "At 2 PM, Jason plans: TBD, skiing, bubble tea, dinner. TBD is likely to be"

# Predict the next word
predicted_next_word = predict_next_word(input_sentence)
print("Predicted next word:", predicted_next_word)