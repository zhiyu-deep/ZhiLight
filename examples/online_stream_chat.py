from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

chat_generator = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "三星堆文明是外星文明吗？"
    }, {
        "role":
        "assistant",
        "content":
        "并没有确凿证据表明三星堆文明是外星文明。"
    }, {
        "role": "user",
        "content": "能给我详细讲一讲吗？"
    }],
    model=model,
    stream = True,
)

for chunk in chat_generator:
    if not chunk.choices[0].delta.content:
        continue
    print(chunk.choices[0].delta.content, end = "", flush = True)
print()