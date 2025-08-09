#!/bin/bash

# Local testing script for Data Analyst Agent (Gemini)

echo "🚀 Testing Data Analyst Agent (Gemini)"
echo "======================================"

# Check if server is running
if ! curl -s http://localhost:8080/health > /dev/null; then
    echo "❌ Server is not running. Please start it with: python app.py"
    exit 1
fi

echo "✅ Server is running"

# Test 1: Wikipedia scraping
echo ""
echo "📊 Test 1: Wikipedia Data Scraping"
echo "-----------------------------------"
curl -X POST http://localhost:8080/api/ \
  -F "questions.txt=@test_examples/questions_wikipedia.txt" \
  -H "Accept: application/json" \
  | python3 -m json.tool

echo ""
echo "📊 Test 2: CSV Data Analysis"
echo "-----------------------------"
curl -X POST http://localhost:8080/api/ \
  -F "questions.txt=@test_examples/questions_csv.txt" \
  -F "data.csv=@test_examples/sample_data.csv" \
  -H "Accept: application/json" \
  | python3 -m json.tool

echo ""
echo "🎉 Testing completed!"
echo "Check the JSON responses above for results."