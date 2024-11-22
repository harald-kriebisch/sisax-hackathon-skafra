import aisisax.object_detection.lsa_interface as aisax_object_detection
import aisisax.llm.openai_connector as aisax_openai
import aisisax.llm.ollama_connector as aisax_ollama

import logging

# Mein Bild
#pImage = "assets/car.jpeg"
#pImage = "assets/Jack_of_clubs_de.png"
pImage = "assets/Skatblatt_02.jpg"

# Object detection
#res = aisax_object_detection.call_lsa(pImage, "card")
#print(type(res))
#res.show()
#res.show()

# GPT Shell

messages = []
f = open('prompts/system.skat2')
system_prompt = "".join(f.readlines())
model = 'gpt-4o'
temp = 0.1
#model = 'gpt-3.5-turbo'
#model = 'o1-preview'

test_case = """
Let's suppose these are the three suits:

P1: [jack-hearts, jack-clubs, ace-diamonds, 10diamonds, 9diamonds, ace-hearts, 10hearts, 8hearts, 7hearts, queen-spades]

P2: [jack-spades, queen-hearts, king-hearts, king-diamonds, queen-diamonds, 8diamonds, 7diamonds, ace-clubs, 10clubs, 9clubs]

P3: [jack-diamonds, king-clubs, queen-clubs, 8clubs, 7clubs, ace-spades, king-spades, 10spades, 9spades, 8spades]

We're pre-bidding -- what's in the Skat?

"""

result = aisax_openai.generate_answer(test_case, temperature=temp, model=model, system_prompt=system_prompt)
messages.append({'role': 'user', 'content': test_case})
messages.append({'role': 'assistant', 'content': result})
print(result)
print("Expected: 9hearts, 7spades")

#while True:
#    myinput = input("> ")
#    if myinput.strip() in ('exit', 'quit'):
#        break
#    #logging.info("Calling ...")
#    result = aisax_openai.generate_answer(myinput, messages=messages,
#                                          temperature=temp, model=model, system_prompt=system_prompt)
#    messages.append({'role': 'user', 'content': myinput})
#    messages.append({'role': 'assistant', 'content': result})
#
#    print(result)

## OpenAI
#query = "Warum ist der Himmel blau?"
#query = "What would be a good programmatic representation for a Skat card game (for use with LLMs)?"
#result = aisax_openai.generate_answer(query)
#print(result)

#print("-----")
#result = aisax_openai.generate_multimodal_answer("Beschreibe das Bild", image_path=pImage)
#print(result)

## LLAMA3.1 via OLLAMA
#result = aisax_ollama.generate_answer("Warum ist der Himmel blau?", model='mistral')
#print(result)
