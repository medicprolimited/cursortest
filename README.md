# Semantic Similarity Checker

A web application that computes semantic similarity between two sentences using HuggingFace's all-mpnet-base-v2 model.

## Features

- Modern, responsive UI built with Tailwind CSS
- FastAPI backend with HuggingFace Inference API integration
- Real-time similarity computation
- Cosine similarity scoring

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your HuggingFace API token:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

## Running Locally

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
2. Open your browser and navigate to `http://localhost:8000`

## Deployment

This application is configured for deployment on Vercel. The domain is set to `apps.medicpro.london`.

### Vercel Configuration

1. Connect your GitHub repository to Vercel
2. Set the following environment variables in Vercel:
   - `HUGGINGFACE_TOKEN`: Your HuggingFace API token

## API Endpoints

- `POST /api/similarity`: Compute similarity between two sentences
  - Request body:
    ```json
    {
      "sentence1": "First sentence",
      "sentence2": "Second sentence"
    }
    ```
  - Response:
    ```json
    {
      "similarity": 0.85
    }
    ```

## Tech Stack

- Backend: FastAPI, HuggingFace Inference API
- Frontend: HTML, JavaScript, Tailwind CSS
- Deployment: Vercel 