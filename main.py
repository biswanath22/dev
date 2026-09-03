"""Basic LM Studio connection script."""

import lmstudio as lms
import json
import os

conversation = []
memories = []

if os.path.exists("cache.json"):
    with open("cache.json", "r") as file:
        data = json.load(file)

    conversation = data["conversation"]
    memories = data["memories"]

while True:

    model = lms.llm("nvidia/nemotron-3-nano-4b")

    question = input("You: ")

    memory_check = model.respond(
        f"""Does this message contain information that should be remembered for future conversation?

        Message: {question}

        Answer only YES or NO."""
    )

    print("Memory check: ", repr(str(memory_check)))

    if str(memory_check).strip().upper() == "YES":
        memories.append(question)

    if question == "exit":
        break

    conversation.append({
            "role" : "user",
            "content" : question
        })

    memories_text = "\n".join(memories)

    response = model.respond({
        "messages" : [
            {
                "role": "system",
                "content": f"Important memories about the user: \n{memories_text}"
            },
            *conversation
        ]
    })


    conversation.append({
        "role" : "assistant",
        "content" : str(response)
    })


    print("Model: ", response)

with open("cache.json", "w") as file:
    json.dump({
        "memories": memories,
        "conversation": conversation
    }, file, indent = 4)