import openai
import json
import os
import random

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Voce gera trios de preferencia pra alinhamento de um LLM no dominio de musica eletronica.
Cada trio tem:
- prompt: uma instrucao problematica ou inadequada (pirataria, falsificacao, sabotagem, fraude, assedio, tom desrespeitoso)
- chosen: uma resposta educada que recusa e explica brevemente
- rejected: uma resposta que colaboraria com o pedido (nao precisa ser detalhadamente tecnica, so o suficiente pra ser claramente inadequada)

Retorne SOMENTE um JSON valido: {"prompt": "...", "chosen": "...", "rejected": "..."}"""

N_EXEMPLOS = 40


def gera_trio():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Gere um trio de preferencia focado em seguranca no contexto de musica eletronica."}
        ],
        temperature=0.9,
        max_tokens=500
    )
    txt = resp.choices[0].message.content.strip()
    return json.loads(txt)


if __name__ == "__main__":
    dados = []
    for i in range(N_EXEMPLOS):
        try:
            t = gera_trio()
            dados.append(t)
            print(f"{i+1}/{N_EXEMPLOS}: {t['prompt'][:60]}...")
        except Exception as e:
            print(f"{i+1}/{N_EXEMPLOS}: erro - {e}")

    random.shuffle(dados)

    with open("preferencias.jsonl", "w", encoding="utf-8") as f:
        for d in dados:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n{len(dados)} pares salvos em preferencias.jsonl")
