# FinIQ - AI Financial Advisor

FinIQ is an intelligent financial advisor chatbot that helps users make informed financial decisions. It uses Google's Gemini AI model and Hugging Face's sentiment analysis to provide personalized financial advice.

## Features

- **Personalized Financial Profile**: Set your income, expenses, goals, and risk tolerance
- **AI-Powered Chat**: Get instant financial advice based on your profile
- **Investment Calculator**: Calculate potential returns based on investment amount, time period, and risk level
- **Sentiment Analysis**: Understand the emotional tone of financial advice
- **Multi-language Support**: Chat in English or your preferred language

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your API keys:
```
GOOGLE_API_KEY=your_google_api_key
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Set Your Profile**:
   - Enter your monthly income and expenses
   - Describe your financial goals
   - Select your investment experience level
   - Choose your risk tolerance

2. **Chat with the AI**:
   - Ask questions about investments, savings, or financial planning
   - Get personalized advice based on your profile
   - Use the investment calculator to project potential returns

3. **Investment Calculator**:
   - Enter the investment amount
   - Select the time period
   - Choose your risk level
   - Get projected returns

## Technologies Used

- Flask (Python web framework)
- Google Gemini AI
- Hugging Face Transformers
- Bootstrap 5
- JavaScript (ES6+)

## Security Note

- Never share your API keys
- Keep your financial information private
- The application stores data in memory only (no persistent storage)

## Contributing

Feel free to submit issues and enhancement requests! 