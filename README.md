# X-ray AI Analysis App

A serverless Flask application for AI-powered X-ray analysis, featuring fracture detection and pneumonia classification with real-time inference.

## Features

- **Fracture Detection**: YOLO-based bone fracture detection
- **Pneumonia Analysis**: Classification and region detection
- **AI Explanations**: Structured medical insights via Google Gemini
- **Real-time Processing**: In-memory image processing (no file persistence)
- **Serverless**: Optimized for Vercel deployment

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshat-gupta-111/x-ray_project.git
   cd x-ray_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run locally**
   ```bash
   python api/main.py
   ```

### Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

3. **Set environment variables in Vercel**
   - Go to your Vercel dashboard
   - Navigate to your project settings
   - Add the following environment variables:
     - `GEMINI_API_KEY`: Your Google Gemini API key
     - `SECRET_KEY`: A secure secret key for the app

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI explanations | Yes |
| `SECRET_KEY` | Flask secret key | Yes |
| `FRACTURE_MODEL_PATH` | Path to fracture detection model | No |
| `PNEUMONIA_CLASSIFIER_PATH` | Path to pneumonia classification model | No |
| `PNEUMONIA_DET_MODEL_PATH` | Path to pneumonia detection model | No |
| `APP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARN, ERROR) | No |
| `ENABLE_VISION_IN_PROMPT` | Enable vision in Gemini prompts | No |

## API Endpoints

- `GET /` - Main interface
- `POST /analyze` - Image analysis endpoint
- `GET /health` - Health check

## Model Requirements

Place your trained YOLO model files in the root directory:
- `best_fracture_yolov8.pt` - Fracture detection model
- `best_classifier.pt` - Pneumonia classification model
- `best_detection.pt` - Pneumonia detection model

## Architecture

- **Frontend**: Vanilla JavaScript with real-time UI updates
- **Backend**: Flask serverless functions
- **AI Models**: YOLO (Ultralytics) + Google Gemini
- **Deployment**: Vercel serverless platform

## Performance Optimizations

- Model caching for reduced cold start times
- In-memory image processing
- Base64 image encoding for frontend display
- Optimized dependencies for serverless

## Security

- No file persistence (images processed in memory)
- Environment variable protection
- Input validation and sanitization
- Error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the GitHub repository.
