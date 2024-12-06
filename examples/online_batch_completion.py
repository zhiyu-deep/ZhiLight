from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

prompts = [
    "三星堆文明是外星文明吗？",
    "奥陌陌是不是外星飞船？"
]

completion = client.completions.create(
    model=model,
    prompt=prompts,
    max_tokens=64)

for i in range(len(prompts)):
    print(f"{prompts[i]} => {completion.choices[i].text}")