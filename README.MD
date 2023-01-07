# chatgpt-clone
Build Yo'own ChatGPT with OpenAI API &amp; Gradio Tested by : Ammar

### Instructions:

1. Get your OpenAI API key here - https://beta.openai.com/account/api-keys
2. Replace that key in the `app.py` code 
3. Install the required libraries `pip install -r requirements.txt` 
4. run `python app.py` 

### Complete Tutorial: https://youtu.be/n5nn3mQxrE8

### Demo

https://user-images.githubusercontent.com/5347322/207718196-c5fccff3-1531-4402-99db-fe0fc6bf0e5a.mp4

## Instructions

1. Install libraries

```
pip install -r requirements.txt
```

2. Run

```
python app.py
```

## Build and run container

1. Build container (uncomment launch call in app.py)

```
docker build --tag palopalepalo/gpt-neo-text-generator:v1 .
```

2. Run container

```
docker run -it -d --name gpt-neo-text-generator -p 7300:7300  palopalepalo/gpt-neo-text-generator:v1
```

## Structure

```
.
├── app.py
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```



2023-01-06 18:15:48.793989: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-01-06 18:15:49.005646: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-01-06 18:15:49.005707: I tensorflow/compiler/xla/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-01-06 18:15:50.376223: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
2023-01-06 18:15:50.376506: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
2023-01-06 18:15:50.376563: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
Downloading: 100% 1.04M/1.04M [00:00<00:00, 1.43MB/s]
Downloading: 100% 456k/456k [00:00<00:00, 1.12MB/s]
Downloading: 100% 665/665 [00:00<00:00, 344kB/s]
Traceback (most recent call last):
  File "app.py", line 6, in <module>
    model = TFGPT2LMHeadModel.from_ptretrained("gpt2, pad_token_id=tokenizer.eos_token_id")
AttributeError: type object 'TFGPT2LMHeadModel' has no attribute 'from_ptretrained'