import gradio as gr

title = "gpt-neo-1.3B"

my_examples = [
    ["The tower is 324 metres (1,063 ft) tall,"],
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]

demo = gr.Interface.load(
    "huggingface/EleutherAI/gpt-neo-1.3B",
    inputs=gr.Textbox(lines=5, max_lines=6, label="Input Text"),
    title="Ammar Test gpt-neo-1.3B",
    examples = my_examples,
)

demo.launch(server_name="0.0.0.0", server_port=7300)

