from openai import OpenAI

client = OpenAI(api_key="")

def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()
