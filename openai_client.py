import openai

openai.api_key = ""
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = "2023-03-15-preview"


def ask_openai(messages, temperature=0.3):
    resp = openai.ChatCompletion.create(    
        engine="gpt-4o",
        messages=messages,
        temperature=temperature
    )
    resp = resp["choices"][0]["message"]["content"]
    return resp
    

if __name__ == '__main__':
    print(ask_openai("Tell me something I don't know."))