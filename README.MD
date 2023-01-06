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
docker build --tag palopalepalo/chat-gpt-dojo:v1 .
```

2. Run container

```
docker run -it -d --name chat-gpt -p 7000:7000  palopalepalo/chat-gpt-dojo:v1
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