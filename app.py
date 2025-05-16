from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from transformers import pipeline
import json
import os
from datetime import datetime
import logging
import re
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate with Hugging Face (optional for public models)
try:
    login(token="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")
    logger.info("Hugging Face authentication successful")
except Exception as e:
    logger.error(f"Hugging Face authentication failed: {str(e)}")

app = Flask(__name__)

# Configure Google Gemini API
try:
    genai.configure(api_key="AIzaSyCzX_snAhMokJbVSvEovd2rVXPIgP4CeSg")
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    raise

# Initialize Hugging Face models
try:
    # Financial sentiment (FinBERT)
    finbert_sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    # Language detection (for Hinglish/English)
    lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
    # Financial QA (optional, for extracting answers from docs)
    fin_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    logger.info("Hugging Face models loaded successfully")
except Exception as e:
    logger.error(f"Error loading Hugging Face models: {str(e)}")
    raise

# Store user profiles
user_profiles = {}

def detect_language(text):
    # Regex-based quick check for Hinglish
    hinglish_patterns = [
        r'\b(acha|theek|nahi|haan|kya|kaise|kab|kahan|kyun|kaisa|kitna|kaunsa)\b',
        r'\b(mein|se|ka|ki|ke|ko|par|se|tak|aur|ya|lekin|magar|kyunki)\b',
        r'\b(paise|rupee|rupay|savings|investment|profit|loss|market|share|stock)\b'
    ]
    for pattern in hinglish_patterns:
        if re.search(pattern, text.lower()):
            return 'hinglish'
    # Model-based check for Hindi/English
    try:
        result = lang_detector(text)
        if result and result[0]['label'].lower() == 'hindi':
            return 'hinglish'
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
    return 'english'

def format_financial_response(response_text, is_hinglish=False):
    """Format the response to always be in bullet/point form, never paragraphs."""
    try:
        lines = re.split(r'\n|\r|\d+\.|•|-|\u2022|\u2023|\u25E6|\u2043|\u2219|;|\*', response_text)
        formatted_points = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) > 120:
                sentences = re.split(r'(?<=[.!?]) +', line)
                line = ' '.join(sentences[:2])
            bullet = '💡' if is_hinglish else '📌'
            formatted_points.append(f"{bullet} {line}")
        return '\n'.join(formatted_points)
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return response_text

def get_financial_sentiment(text):
    try:
        result = finbert_sentiment(text)
        if result:
            return result[0]['label']
    except Exception as e:
        logger.warning(f"FinBERT sentiment failed: {str(e)}")
    return None

def get_qa_answer(context, question):
    try:
        result = fin_qa(question=question, context=context)
        if result and 'answer' in result:
            return result['answer']
    except Exception as e:
        logger.warning(f"Financial QA failed: {str(e)}")
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/set_profile', methods=['POST'])
def set_profile():
    try:
        data = request.json
        user_id = data.get('user_id')
        user_profiles[user_id] = {
            'income': data.get('income'),
            'expenses': data.get('expenses'),
            'goals': data.get('goals'),
            'experience': data.get('experience'),
            'risk_tolerance': data.get('risk_tolerance')
        }
        logger.info(f"Profile set successfully for user {user_id}")
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error setting profile: {str(e)}")
        return jsonify({"error": "Failed to set profile. Please try again."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id')
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400
        if user_id not in user_profiles:
            return jsonify({"error": "Please set your profile first"}), 400
        # Detect language
        is_hinglish = detect_language(message) == 'hinglish'
        # Get user profile
        profile = user_profiles[user_id]
        # Improved context for Gemini
        context = f"""
        You are a financial advisor AI. Always:
        - Give clear info on each user goal (amount, timeline, type)
        - Reference the user's financial profile (income, expenses, experience, risk)
        - If user writes in Hinglish, reply in Hinglish with practical, actionable advice
        - If user writes in English, reply in English
        - Use bullet points for each goal: [Goal: amount, timeline, type, advice]
        - End with a tailored, actionable next step
        - Be concise but detailed and practical
        
        User Financial Profile:
        Income: {profile['income']}
        Expenses: {profile['expenses']}
        Experience: {profile['experience']}
        Risk Tolerance: {profile['risk_tolerance']}
        
        User Goals:
        {profile['goals']}
        
        User Message:
        {message}
        
        Format:
        [For each goal:]
        - Goal: [describe goal, amount, timeline, type]
        - Advice: [tailored, actionable suggestion]
        [End with:]
        - Next Step: [one practical action user should take]
        """
        try:
            # Generate response using Gemini
            response = model.generate_content(context)
            response_text = response.text
            if not response_text:
                raise Exception("Empty response from Gemini API")
            # Format the response
            formatted_response = format_financial_response(response_text, is_hinglish)
            # Truncate response text if it's too long for sentiment analysis
            truncated_text = formatted_response[:512] if len(formatted_response) > 512 else formatted_response
            # Analyze sentiment of the truncated response (FinBERT for financial, Hinglish for Hinglish)
            sentiment = get_financial_sentiment(truncated_text)
            logger.info(f"Successfully generated response for user {user_id}")
            return jsonify({
                "response": formatted_response,
                "sentiment": sentiment,
                "confidence": 1.0  # Placeholder, as not all models return confidence
            })
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return jsonify({
                "error": "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
            }), 500
    except Exception as e:
        logger.error(f"Unexpected error in chat: {str(e)}")
        return jsonify({
            "error": "An unexpected error occurred. Please try again later."
        }), 500

@app.route('/calculate_investment', methods=['POST'])
def calculate_investment():
    try:
        data = request.json
        amount = float(data.get('amount', 0))
        years = int(data.get('years', 0))
        risk_level = data.get('risk_level', 'medium')
        if amount <= 0 or years <= 0:
            return jsonify({"error": "Amount and years must be greater than 0"}), 400
        # Basic investment calculation
        if risk_level == 'low':
            rate = 0.05  # 5% return
        elif risk_level == 'medium':
            rate = 0.08  # 8% return
        else:
            rate = 0.12  # 12% return
        future_value = amount * (1 + rate) ** years
        logger.info(f"Investment calculated successfully: amount={amount}, years={years}, risk={risk_level}")
        return jsonify({
            "initial_investment": amount,
            "years": years,
            "expected_return": rate * 100,
            "future_value": round(future_value, 2)
        })
    except Exception as e:
        logger.error(f"Error calculating investment: {str(e)}")
        return jsonify({"error": "Failed to calculate investment. Please check your inputs."}), 500

if __name__ == '__main__':
    app.run(debug=True) 