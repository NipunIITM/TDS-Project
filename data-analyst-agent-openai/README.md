# Data Analyst Agent (OpenAI)

A powerful AI-powered data analysis API that uses OpenAI's GPT-4o-mini to source, prepare, analyze, and visualize any data. This agent can handle complex data analysis tasks including web scraping, statistical analysis, and data visualization.

## Features

- **Intelligent Data Analysis**: Uses OpenAI's GPT-4o-mini to understand and analyze data requests
- **Web Scraping**: Can automatically scrape data from URLs (Wikipedia, APIs, etc.)
- **File Processing**: Supports CSV files and other data formats
- **Data Visualization**: Creates charts and plots encoded as base64 data URIs
- **Statistical Analysis**: Performs correlations, regressions, and other statistical operations
- **Secure Execution**: Safely executes generated Python code in a controlled environment
- **Production Ready**: Optimized for deployment on platforms like Render

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd data-analyst-agent-openai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
PORT=8080
```

### 4. Run Locally

```bash
python app.py
```

The API will be available at `http://localhost:8080`

## API Usage

### Endpoint

```
POST /api/
```

### Request Format

Send a multipart form request with:
- `questions.txt` (required): Contains your data analysis questions
- `data.csv` (optional): Your data file
- Additional files as needed

### Example Request

```bash
curl -X POST http://localhost:8080/api/ \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

### Example Questions File

Create a `questions.txt` file:

```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array containing the answers:
1. How many $2 billion movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 billion?
3. What's the correlation between the Rank and Peak columns?
4. Draw a scatterplot of Rank vs Peak along with a dotted red regression line through it.

Return the plot as a base-64 encoded data URI under 100,000 bytes.
```

### Example Response

```json
[
  1,
  "Titanic",
  0.485782,
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
]
```

## Local Testing

### Test with Sample Data

1. Create a `questions.txt` file with your analysis questions
2. Optionally create a `data.csv` file with your data
3. Run the curl command:

```bash
curl -X POST http://localhost:8080/api/ \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

### Test Web Scraping

Create a `questions.txt` file that asks for data from a URL:

```
Analyze the data from https://example.com/data.csv
1. How many rows are in the dataset?
2. What are the column names?
3. Show summary statistics for numeric columns.
```

## Deployment

### Deploy to Render

1. **Create a GitHub Repository**:
   - Create a new public repository on GitHub
   - Push your code to the repository

2. **Connect to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**:
   - **Name**: `data-analyst-agent-openai`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (or paid for better performance)

4. **Set Environment Variables**:
   - Add `OPENAI_API_KEY` with your API key
   - Add `PORT` with value `10000` (Render's default)

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Your API will be available at `https://your-service-name.onrender.com/api/`

### Deploy to Other Platforms

The application is compatible with:
- **Heroku**: Use the included `requirements.txt` and set environment variables
- **Railway**: Direct deployment from GitHub
- **Google Cloud Run**: Use Docker or direct Python deployment
- **AWS Lambda**: With appropriate adaptations for serverless

## Architecture

The application uses an efficient architecture:

1. **LLM as Planner**: OpenAI generates Python code to solve the analysis task
2. **Secure Execution**: Generated code runs in a controlled environment
3. **Result Capture**: Results are captured from the executed code
4. **Token Efficiency**: Minimizes API costs by using the LLM only for planning

## Security Features

- Sandboxed code execution with limited builtins
- No file system access beyond the execution scope
- Memory management with plot cleanup
- Input validation and error handling

## Supported Analysis Types

- **Statistical Analysis**: Correlations, regressions, descriptive statistics
- **Data Visualization**: Scatter plots, line charts, bar charts, histograms
- **Web Scraping**: Wikipedia, APIs, CSV files from URLs
- **Data Cleaning**: Missing value handling, data type conversions
- **Time Series**: Date parsing, trend analysis
- **Machine Learning**: Basic regression, classification

## Error Handling

The API provides detailed error messages for:
- Missing or invalid files
- Code execution errors
- API key issues
- Network problems during data fetching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check the error messages in the API response
- Verify your OpenAI API key is valid
- Ensure all required files are included in the request
- Check the server logs for detailed error information