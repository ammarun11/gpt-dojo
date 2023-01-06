import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_ptretrained("gpt2, pad_token_id=tokenizer.eos_token_id")

def generate_text(inp):
    input_ids = tokenizer.encode(inp, return_tensors='tf')
    beam_output = model.generate(input_ids, max_lenth=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

output_text = gr.outputs.Textbox()
iface = gr.Interface(generate_text, "textbox", output_text, tittle= "GPT-2", description= "OpenAI's GPT-2 is an unsupervised language model that can generate coherent text")

iface.launch(server_name="0.0.0.0", server_port=7200)