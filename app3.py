import gradio as gr
from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
#generator("EleutherAI has", do_sample=True, min_length=50)


def generate(text):
    result = generator(text, max_length=30, num_return_sequences=1)
    # result = generator("EleutherAI has", do_sample=True, min_length=50)
    return result[0]["generated_text"]

examples = [
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

demo = gr.Interface(
    fn=generate,
    inputs=gr.inputs.Textbox(lines=5, label="Input Text"),
    outputs=gr.outputs.Textbox(label="Generated Text"),
    examples=examples
)

#demo.launch()
demo.launch(server_name="0.0.0.0", server_port=7310)

