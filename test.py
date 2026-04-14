import ollama

text = "ใส่เนื้อหา HTML ที่ clean แล้วตรงนี้..."

response = ollama.chat(
    model="phi3",
    messages=[
        {
            "role": "user",
            "content": f"Summarize this article in Thai:\n\n{text}"
        }
    ]
)

print(response['message']['content'])

