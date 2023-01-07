import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPTNeoForCausalLM 

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

my_examples = [
    ['Covid-19 vaccines are regarded as an effective way to'],
    ['Whos the real goat in football'],
    ['Whos the best pro player dota']
]

def textgen(Start_of_my_setence):
    input_ids = tokenizer(Start_of_my_setence, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        min_lenth = 30, 
        #temperature = 0.9,
        max_length=127,
        num_return_sequences=2,
    )

    test_list=[]
    for i, gen_tokens in enumerate(gen_tokens):
        #print("{}: {}".format(i, tokenizer.decode(gen_tokens,skip_special_token")
        test_list.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return test_list

    my_article = "<p style='text-align: center'><a href='https://ammarun.my.id' target='_blank'>Ammarun Dojo ML</a> | <a href='https://github.com/ammarun11/gpt-dojo' target='_blank'>Github Repo</a> | <a href='https://huggingface.co/docs/transformers/model_doc/gpt_neo' target='_blank'>GPT Neo Model</a></p></center>"

    iface = gr.Interface(
        fn = textgen,
        inputs=gr.inputs.Textbox(lines=2, default="NeW year 2023"),
        theme='grass'
        title="AI Text Generator GPT-NEO 07/01/2023"
        description='>> model : gpt-neo-1.3B <<'
        outputs=[gr.outputs.Textbox(type="auto", label="AI storytelling"), gr.outputs.Textbox(type="auto", label="AI #2")],
        article=my_article
        layout='vertical',   
    )
iface.launch(server_name="0.0.0.0", server_port=7300)

# Docs : https://huggingface.co/docs/transformers/model_doc/gpt_neo