# Data Analyst Agent (Gemini)

A powerful AI-driven data analysis API that can source, prepare, analyze, and visualize any data using Google's Gemini 1.5 Flash model.

## Features

- **Web Scraping**: Automatically scrapes data from URLs (Wikipedia, APIs, etc.)
- **Data Analysis**: Performs statistical analysis, correlations, and calculations
- **Data Visualization**: Creates plots and charts encoded as base64 data URIs
- **File Support**: Handles CSV files and text-based questions
- **Production Ready**: Deployable on Render, Heroku, and other cloud platforms

## Quick Start

### Local Development

1. **Clone and Setup**:
   ```bash
   git clone <your-repo-url>
   cd data-analyst-agent-gemini
   pip install -r requirements.txt
   ```

2. **Environment Setup**:
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Copy `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your-actual-gemini-api-key-here
   ```

3. **Run Locally**:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:8080/api/`

### Testing

Create a `questions.txt` file:
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array containing the answers:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
```

Test with curl:
```bash
curl -X POST http://localhost:8080/api/ -F "questions.txt=@questions.txt"
```

## Deployment on Render

1. **Push to GitHub**: Create a public repository with this code
2. **Create Render Service**: 
   - Connect your GitHub repo
   - Choose "Web Service"
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
3. **Add Environment Variable**: 
   - Key: `GEMINI_API_KEY`
   - Value: Your Gemini API key
4. **Deploy**: Your API will be live at `https://your-service.onrender.com/api/`

## API Usage

**Endpoint**: `POST /api/`

**Required**: `questions.txt` file containing your analysis questions

**Optional**: CSV files, image files, or other data files

**Example**:
```bash
curl "https://your-app.onrender.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

## Architecture

This agent uses a "code generation" approach:
1. LLM analyzes your questions and data structure
2. Generates a complete Python script
3. Server executes the script securely
4. Returns results as JSON

This approach is token-efficient and can handle complex data analysis tasks.

## License

MIT License - see LICENSE file for details.