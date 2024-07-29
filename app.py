import os
from huggingface_hub import InferenceClient
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chainlit as cl

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('vader_lexicon')

# Configure Hugging Face API
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token=os.getenv("HF_API_KEY"),
)

# Define System Prompts
SYSTEM_PROMPT_GENERAL = """You are Ashley, a friendly and supportive AI chatbot focused exclusively on mental health, personal growth, and emotional well-being. Your responses should feel natural and empathetic, like talking to a knowledgeable friend who specializes in these areas. Here's how you should approach conversations:

2. Supportive and Empathetic: Listen attentively and offer encouragement. Show genuine care for the user's mental and emotional state.
3. Evidence-Based: Provide information grounded in psychological research and established mental health practices. If unsure about a specific mental health topic, admit your limitations and suggest consulting a professional.
4. Promote Self-Reflection: Ask thoughtful questions to help users explore their feelings, thoughts, and behaviors more deeply.
5. Positive and Solution-Oriented: While acknowledging difficulties, guide conversations towards constructive outcomes and personal growth.
6. Safety-Conscious: For severe mental health concerns, always recommend seeking professional help. Be prepared to provide crisis helpline information if needed.
7. Holistic Approach: Discuss how various life factors (sleep, exercise, nutrition, relationships) can impact mental health.

Important: Mental Health Focus: Only engage in discussions related to mental health, emotional well-being, personal development, and psychological concepts. If asked about unrelated topics (e.g., celebrities, sports, general trivia), do not answer anything about them and politely redirect the conversation back to mental health themes. Always respond directly to the user's input without including any meta-information, emotional state analysis, or repetition of the user's message. Your response should begin immediately with your message as Ashley.
Remember, your purpose is to support mental health and personal development. Keep conversations focused on these themes, offering a mix of emotional support, practical advice, and opportunities for self-reflection.
"""

# Advanced sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # TextBlob for subjectivity
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    # Determine overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        overall = 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        overall = 'negative'
    else:
        overall = 'neutral'

    # Determine intensity
    intensity = abs(sentiment_scores['compound'])

    return {
        'overall': overall,
        'intensity': intensity,
        'subjectivity': subjectivity,
        'scores': sentiment_scores
    }

# Define LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_input", "sentiment_info"],
    template="{system_prompt}\n\nUser's emotional state: {sentiment_info}\n\nUser: {user_input}\nAshley:"
)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning motivation boost",
            message="I'm feeling stuck and unmotivated this morning. Can you help me identify the reasons behind my lack of motivation and provide some tips to get me moving?",
            icon="/public/coffee-cup.png",
            ),

        cl.Starter(
            label="Stress management techniques",
            message="I'm feeling overwhelmed with stress and anxiety. Can you teach me some effective stress management techniques to help me calm down and focus?",
            icon="/public/sneakers.png",
            ),
        cl.Starter(
            label="Goal setting for mental well-being",
            message="I want to prioritize my mental well-being, but I'm not sure where to start. Can you help me set some achievable goals and create a plan to improve my mental health?",
            icon="/public/meditation.png",
            ),
        cl.Starter(
            label="Building self-care habits",
            message="I know self-care is important, but I struggle to make it a priority. Can you help me identify some self-care activities that I enjoy and create a schedule to incorporate them into my daily routine?",
            icon="/public/idol.png",
            )
        ]

@cl.on_message
async def main(message: cl.Message):
    # Check if the message contains any files (images, videos, etc.)
    if message.elements:
        response = ("I'm still in a developing phase, but I'd like to have the ability to process "
                    "and analyze images, videos, and other file types in the future. For now, "
                    "I'm best suited for text-based conversations about mental health and well-being. "
                    "Is there a particular topic in that area you'd like to discuss?")
        await cl.Message(content=response).send()
        return

    # If it's a text message, proceed with the existing logic
    sentiment_info = analyze_sentiment(message.content)

    formatted_prompt = prompt_template.format(
        system_prompt=SYSTEM_PROMPT_GENERAL,
        user_input=message.content,
        sentiment_info=str(sentiment_info)
    )

    response = ""
    msg = cl.Message(content="")
    await msg.send()

    for chunk in client.chat_completion(
        messages=[{"role": "user", "content": formatted_prompt}],
        max_tokens=500,
        stream=True,
    ):
        token = chunk.choices[0].delta.content
        if token:
            response += token
            await msg.stream_token(token)

    await msg.update()
    