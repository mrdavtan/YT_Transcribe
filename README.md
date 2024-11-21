# YT_Transcribe

![symb](https://github.com/mrdavtan/YT_Transcribe/assets/21132073/af830f85-cf54-46ef-8bb5-9cf701e90349)

# YouTube Video Transcription and Analysis CLI Tool

## Description

The YouTube Video Transcription CLI Tool is a Python toolkit that allows you to:
- Download and transcribe YouTube videos
- Generate accurate transcriptions with timestamps
- Perform multi-pass analysis of the content
- Create structured summaries and insights
- Process single videos or batch process multiple videos

The tool uses `yt-dlp` for video downloading, OpenAI's `whisper` for transcription, and the Llama language model for analysis.

## Features

### Transcription Features
- Transcribe single or multiple YouTube videos using Whisper's speech recognition
- Download videos efficiently using yt-dlp
- Preserve natural speech segments with accurate timestamps
- Automatically format and capitalize text for improved readability
- Generate organized output with video title and timestamps
- Save transcriptions in a structured directory format
- Process multiple videos from a text file

### Analysis Features
- Three-pass analysis pipeline for comprehensive understanding:
  1. Detailed summary with preserved timestamps
  2. Structured bullet points with key themes
  3. Analytical report with insights and implications
- Multiple output formats:
  - Executive summary (500 words)
  - Key insights and takeaways
  - Hierarchical bullet points
  - Detailed summary with timestamps
  - Full analytical report
- Organized directory structure for easy navigation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/youtube-transcription-tool.git
```

2. Navigate to the project directory:
```bash
cd youtube-transcription-tool
```

3. Create a Python virtual environment:
```bash
python -m venv .venv
```

4. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Video Processing

The main script `analyze_video.py` handles both transcription and analysis:

```bash
# Process a single video with full analysis
python analyze_video.py -u "https://youtube.com/watch?v=..."

# Process multiple videos from a file
python analyze_video.py -f urls.txt

# Specify custom output directory
python analyze_video.py -u "https://youtube.com/watch?v=..." -o my_analyses
```

### Individual Components

If you prefer to use the components separately:

#### Transcription Only
```bash
python whisper_transcribe_videos.py -u "https://youtube.com/watch?v=..."
# or
python whisper_transcribe_videos.py -f urls.txt
```

#### Analysis Only (for existing transcriptions)
```bash
python advanced_summarizer.py -f transcription.txt
```

### Output Structure

The tool creates an organized directory structure:

```
analysis_pipeline/
└── analysis_20240120_123456/
    ├── index.txt               # Overview of all files
    ├── executive_summary.txt   # 500-word overview
    ├── key_insights.txt        # Core insights
    ├── bullet_points.txt       # Structured summary
    ├── detailed_summary.txt    # Full summary with timestamps
    └── full_analysis.txt       # Complete analysis
```

### URLs File Format

When processing multiple videos, create a text file with one YouTube URL per line:
```text
https://www.youtube.com/watch?v=video1
https://www.youtube.com/watch?v=video2
https://www.youtube.com/watch?v=video3
```

### Output Formats

1. **Executive Summary**: Quick 500-word overview of the main content
2. **Key Insights**: 3-5 most valuable and unique points
3. **Bullet Points**: Hierarchical structure of main themes and supporting points
4. **Detailed Summary**: Comprehensive summary with timestamp references
5. **Full Analysis**: In-depth analysis weaving together all insights

## Technical Details

### Analysis Pipeline Architecture

The analysis pipeline consists of three main stages:

1. **First Pass - Detailed Summary**
   - Processes text in chunks of 1024 tokens
   - Preserves all timestamp references
   - Maintains chronological order
   - Captures detailed information and quotes
   - Output: detailed_summary.txt

2. **Second Pass - Structured Analysis**
   - Takes the detailed summary as input
   - Identifies main themes and patterns
   - Creates hierarchical bullet points
   - Tags key concepts and relationships
   - Output: bullet_points.txt

3. **Third Pass - Synthesis**
   - Combines insights from previous passes
   - Generates executive summary
   - Identifies key insights
   - Creates comprehensive analysis
   - Outputs: executive_summary.txt, key_insights.txt, full_analysis.txt

### Model Configuration

The tool uses the following models with specific configurations:

```python
# Whisper Configuration
model = whisper.load_model("base")
options = {
    "language": "en",
    "task": "transcribe"
}

# Llama Model Configuration
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_config = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
    "use_cache": False
}
```

## Troubleshooting

### Common Issues

1. **Video Download Fails**
   - Check your internet connection
   - Verify the video URL is valid and accessible
   - Ensure you have the latest version of yt-dlp
   ```bash
   pip install --upgrade yt-dlp
   ```

2. **Transcription Errors**
   - Ensure ffmpeg is properly installed
   - Check audio quality of the video
   - Try using a different Whisper model size
   ```bash
   # Modify in whisper_transcribe_videos.py
   model = whisper.load_model("small")  # or "medium", "large"
   ```

3. **Analysis Pipeline Issues**
   - Check available RAM (minimum 16GB recommended)
   - Verify all dependencies are installed
   - Try processing in smaller chunks
   ```bash
   # Modify chunk size in advanced_summarizer.py
   chunk_size = 512  # Default is 1024
   ```

4. **Out of Memory Errors**
   - Reduce batch size
   - Use smaller model variants
   - Process shorter segments
   - Clear GPU memory between runs

### Performance Optimization

1. **For Faster Processing**
   - Use smaller Whisper model
   - Reduce chunk size for analysis
   - Process shorter video segments
   - Enable GPU acceleration if available

2. **For Better Quality**
   - Use larger Whisper model
   - Increase chunk overlap
   - Adjust temperature and top_p settings
   - Increase max_new_tokens for analysis

## Contributing

Contributions are welcome! Here's how you can help:

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Write clear, commented code
4. Add tests for new features
5. Submit a pull request

### Testing

Run the test suite before submitting:
```bash
python -m pytest tests/
```

### Documentation

Help improve the documentation:
- Fix typos or unclear instructions
- Add examples and use cases
- Improve troubleshooting guides
- Update technical documentation

### Bug Reports

When submitting bug reports, include:
- Operating system and version
- Python version
- Full error message
- Steps to reproduce
- Example input causing the error

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for transcription capabilities
- yt-dlp for video downloading
- Meta's Llama model for analysis
- All contributors and users of this tool
