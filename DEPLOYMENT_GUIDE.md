# Data Analyst Agent - Deployment Guide

This guide covers deploying both the OpenAI and Gemini versions of the Data Analyst Agent to Render.

## Project Structure

You now have two complete, identical solutions:

1. **`data-analyst-agent-openai/`** - Uses OpenAI GPT-4o-mini
2. **`data-analyst-agent-gemini/`** - Uses Google Gemini 1.5 Flash

Both projects are production-ready and deployable on Render.

## Pre-Deployment Setup

### For OpenAI Version:
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Update `.env` file: `OPENAI_API_KEY=sk-your-actual-key-here`

### For Gemini Version:
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Update `.env` file: `GEMINI_API_KEY=your-actual-key-here`

## Local Testing

### Test OpenAI Version:
```bash
cd data-analyst-agent-openai
pip install -r requirements.txt
python app.py
```

### Test Gemini Version:
```bash
cd data-analyst-agent-gemini
pip install -r requirements.txt
python app.py
```

### Test with Sample Data:
```bash
# Test web scraping (Wikipedia)
curl -X POST http://localhost:8080/api/ -F "questions.txt=@../test_questions.txt"

# Test CSV analysis
curl -X POST http://localhost:8080/api/ \
  -F "questions.txt=@../test_csv_questions.txt" \
  -F "data.csv=@../test_data.csv"
```

## Render Deployment Steps

### Step 1: Push to GitHub

1. **Create two separate GitHub repositories**:
   - `your-username/data-analyst-agent-openai`
   - `your-username/data-analyst-agent-gemini`

2. **Push each project**:
   ```bash
   # For OpenAI version
   cd data-analyst-agent-openai
   git init
   git add .
   git commit -m "Initial commit - OpenAI Data Analyst Agent"
   git remote add origin https://github.com/your-username/data-analyst-agent-openai.git
   git push -u origin main

   # For Gemini version
   cd ../data-analyst-agent-gemini
   git init
   git add .
   git commit -m "Initial commit - Gemini Data Analyst Agent"
   git remote add origin https://github.com/your-username/data-analyst-agent-gemini.git
   git push -u origin main
   ```

### Step 2: Deploy on Render

#### For Each Project (OpenAI and Gemini):

1. **Create New Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Web Service"
   - Connect your GitHub account
   - Select the repository

2. **Configure Service Settings**:
   - **Name**: `data-analyst-agent-openai` or `data-analyst-agent-gemini`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (sufficient for testing)

3. **Add Environment Variables**:
   - Go to "Environment" tab
   - Click "Add Environment Variable"
   - **For OpenAI**: Key: `OPENAI_API_KEY`, Value: Your OpenAI API key
   - **For Gemini**: Key: `GEMINI_API_KEY`, Value: Your Gemini API key

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete (~5-10 minutes)
   - Your API will be live at: `https://your-service-name.onrender.com/api/`

### Step 3: Test Deployed API

```bash
# Test your deployed API
curl -X POST https://your-service-name.onrender.com/api/ \
  -F "questions.txt=@test_questions.txt"
```

## Performance Optimization

### For Production Use:

1. **Upgrade Render Plan**: Free tier has limitations
   - Consider "Starter" plan for better performance
   - No cold starts with paid plans

2. **Environment Variables**:
   - Set `FLASK_ENV=production`
   - Adjust `PORT` if needed (Render handles this automatically)

3. **Monitoring**:
   - Check Render logs for errors
   - Monitor API response times
   - Set up health check monitoring

## API Endpoints

Both deployed services will have:

- **Health Check**: `GET /` 
- **Analysis Endpoint**: `POST /api/`

## Submission URLs

After successful deployment, you'll have:

1. **GitHub Repository URLs**:
   - OpenAI: `https://github.com/your-username/data-analyst-agent-openai`
   - Gemini: `https://github.com/your-username/data-analyst-agent-gemini`

2. **Live API Endpoints**:
   - OpenAI: `https://your-openai-service.onrender.com/api/`
   - Gemini: `https://your-gemini-service.onrender.com/api/`

Submit these URLs at: https://exam.sanand.workers.dev/tds-data-analyst-agent

## Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check `requirements.txt` syntax
   - Ensure all dependencies are compatible
   - Check build logs in Render dashboard

2. **Runtime Errors**:
   - Verify environment variables are set correctly
   - Check application logs
   - Test API keys locally first

3. **API Key Issues**:
   - Ensure keys are valid and have sufficient credits
   - Check key permissions and quotas
   - Verify environment variable names match exactly

4. **Timeout Issues**:
   - Large data processing may take time
   - Consider optimizing prompts for efficiency
   - Monitor token usage

### Support Resources:

- [Render Documentation](https://render.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google AI Documentation](https://ai.google.dev/docs)

## Security Notes

- Never commit API keys to Git (they're in `.gitignore`)
- Use environment variables for all sensitive data
- Monitor API usage and costs
- Consider rate limiting for production use

Good luck with your deployment!