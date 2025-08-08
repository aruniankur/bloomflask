import base64
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
import ast
import json


def generatetextquestion(context,instruction,number,length,info):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),)
    model = "gemini-2.5-flash"
    contents = [types.Content(role="user",parts=[
                types.Part.from_text(text=f"""context : {context}
                                            Question Generation Guidelines: {number} Text question 
                                            Question Length: {length}
                                            Bloom's Taxonomy Distribution:
                                            Distribute the questions according to the following cognitive levels (approximate weightage based on total 100%):
                                            {info}                                             
                                            user extra instruction : {instruction}"""),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,thinking_config = types.ThinkingConfig(thinking_budget=0,),
        system_instruction=[types.Part.from_text(text="""You are a question generation AI. Based on the context I provide, generate questions according to the detailed specifications. follow the format {1:\" question text \",2:\" question text \" .....}"""),],)

    response = client.models.generate_content(model=model,contents=contents,config=generate_content_config,)
    questions_dict = ast.literal_eval(response.text)
    return list(questions_dict.values())
    
    
def generatemcqquestion(context, instruction, number, length, info):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash"

    # User prompt content
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=f"""Context: {context}
MCQ Generation Guidelines:
- Number of Questions: {number}
- Question Length: {length}
- Bloom's Taxonomy Distribution: {info}
- Extra Instructions: {instruction}

Return Format (JSON-like dictionary):
{{ 
    1: {{
        "question": "What is the capital of France?",
        "options": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": "Paris"
    }},
    2: {{
        "question": "Which of the following is a mammal?",
        "options": ["Snake", "Frog", "Whale", "Lizard"],
        "answer": "Whale"
    }},
    ...
}}""")
        ])
    ]

    # System instruction
    system_instruction = [
        types.Part.from_text(text="""
You are an MCQ generation AI. Based on the provided context and specifications, generate multiple-choice questions.
Make sure each question includes:
- The question text
- A list of 4 options (labeled A, B, C, D is optional)
- One correct answer (exactly matching one of the options)
Return the result in the following format (dictionary):
{
    1: {
        "question": "...",
        "options": ["...", "...", "...", "..."],
        "answer": "..."
    },
    ...
}

don't write json and all , just the above format only
""")
    ]

    # Generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_instruction
    )

    # API call
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )

    # Convert the response to Python dict
    mcq_dict = ast.literal_eval(response.text)
    return list(mcq_dict.values())



def generatetruefalsequestion(context, instruction, number, length, info):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash"

    # Prompt contents
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_text(text=f"""Context: {context}
True/False Question Generation Guidelines:
- Number of Questions: {number}
- Question Length: {length}
- Bloom's Taxonomy Distribution: {info}
- Extra Instructions: {instruction}

Return Format (dictionary):
{{
    1: {{
        "question": "Water boils at 100°C at sea level.",
        "answer": True
    }},
    2: {{
        "question": "The moon is a planet.",
        "answer": False
    }},
    ...
}}""")
        ])
    ]

    # System instruction
    system_instruction = [
        types.Part.from_text(text="""
You are a question generation AI specializing in True/False questions.
Generate factual statements based on the provided context, following the given structure:
- Each item must have:
    - A 'question' field with a full statement.
    - An 'answer' field: either True or False.
Return output in dictionary format like:
{
    1: {"question": "...", "answer": True},
    2: {"question": "...", "answer": False},
    ...
}

don't write json and all , just the above format only
""")
    ]

    # Generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_instruction
    )

    # API call
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )

    # Convert response to Python dict
    
    tf_dict = ast.literal_eval(response.text)
    return list(tf_dict.values())


# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generatebloomscore(type, data):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(mime_type=type,data=base64.b64decode(data),),
                types.Part.from_text(text="""I am providing you with a file containing a set of questions (possibly from an exam, workbook, or learning material). Your task is to:

                1. Extract All Questions:
                Identify and list every individual question in the file.
                If there are sub-questions (e.g., parts a, b, c under a main question), treat each sub-question as a separate question unless they are clearly dependent on each other.
                Number the questions sequentially. in case of MCQ , take the base question only.

                2. Score Each Question Using Bloom’s Taxonomy:
                Use Bloom's Revised Taxonomy to analyze the cognitive level of each question. Bloom’s taxonomy classifies learning into six hierarchical categories, from lower-order to higher-order thinking:

                Bloom's Level	Description
                Remembering	Recalling facts, terms, basic concepts (e.g., list, define, memorize)
                Understanding	Explaining ideas or concepts (e.g., summarize, explain, describe)
                Applying	Using information in new situations (e.g., solve, demonstrate, use)
                Analyzing	Breaking information into parts to explore relationships (e.g., compare, contrast, organize)
                Evaluating	Justifying a decision or course of action (e.g., critique, argue, support)
                Creating	Generating new ideas, products, or ways of viewing things (e.g., design, compose, invent)

                3. Assign Scores (0–100) per Category for Each Question:
                For each question, assign a score from 0 to 100 for each of the six Bloom's levels.

                The score should reflect how much the question targets that level of cognition.

                Example: A pure recall question might get:
                {\"remembering\": 100, \"understanding\": 0, \"applying\": 0, \"analyzing\": 0, \"evaluating\": 0, \"creating\": 0}

                A more complex question might span multiple levels. For example:
                {\"remembering\": 10, \"understanding\": 20, \"applying\": 40, \"analyzing\": 30, \"evaluating\": 0, \"creating\": 0}

                Ensure the total doesn’t need to sum to 100 – each level is scored independently, on its own scale, and output format is a list of dict {\"question\": , \"remembering\":, \"Understanding\": , \"Applying\": , \"Analyzing\": , \"evaluating\":, \"creating\": }
                
                don't write anything except the array
                """),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )

    print(response.text.replace("json",""))
    text = response.text.replace("json","")
    text = text.replace("```","")
    return json.loads(text)