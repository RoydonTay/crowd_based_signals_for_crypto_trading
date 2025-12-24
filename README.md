# NUS Fintech Society: Crowd Based Signals For Algo Crypto Trading

This project was completed by members of the NUS Fintech Society (ML Dept). This repository contains the code for data collection, processing and model training experiments.

## Reddit Data Collection with GitHub Actions (Turned Off)

This repository automatically collects Reddit data from cryptocurrency-related subreddits and uploads the results to Google Drive on a daily schedule.

### Features

- **Automated Data Collection**: Runs hourly via GitHub Actions
- **Multiple Subreddits**: Collects from `btc`, `eth`, and `CryptoMarkets`
- **Google Drive Integration**: Automatically uploads CSV files to your Google Drive
- **Manual Trigger**: Can be triggered manually via GitHub Actions UI

### Setup Instructions

#### 1. Reddit API Credentials

1. Create a Reddit app at https://www.reddit.com/prefs/apps
2. Note your `client_id` and `client_secret`

#### 2. Google Drive API Setup

**Option A: OAuth2 with Refresh Token (Recommended for GitHub Actions)**

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API
4. Create OAuth2 credentials:
   - Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
   - Choose "Desktop application"
   - Download the JSON file
5. Add the JSON content to your `.env` file as `GOOGLE_CREDENTIALS`
6. Run the setup script: `python setup_google_oauth.py`
7. Follow the browser prompts to authorize
8. Copy the generated tokens to your `.env` file

**Option B: Service Account (Limited due to storage quota)**

1. Create a service account and download the JSON key
2. Create a folder in your Google Drive
3. Share the folder with the service account email (found in the JSON)
4. Add the JSON content to your `.env` file as `GOOGLE_CREDENTIALS`

#### 3. GitHub Repository Secrets

**For OAuth2 method (recommended):**
- `REDDIT_CLIENT_ID`: Your Reddit app client ID
- `REDDIT_CLIENT_SECRET`: Your Reddit app client secret  
- `GOOGLE_CLIENT_ID`: From setup script output
- `GOOGLE_CLIENT_SECRET`: From setup script output
- `GOOGLE_REFRESH_TOKEN`: From setup script output

### Workflow Schedule

The GitHub Action runs:
- **Daily at 23:00 SGT** (automatically)
- **On-demand** via the Actions tab in GitHub

### Output Files

Each run generates CSV files named: `reddit_data_{subreddit}_{date}_{time}.csv`

Columns include:
- `title`: Post title
- `selftext`: Post/comment content
- `is_comment`: Boolean indicating if row is a comment
- `score`: Reddit score (upvotes - downvotes)
- `flair`: Post flair text

### Troubleshooting

- **Import errors**: Run `pip install -e .` to ensure all dependencies are installed
- **Google auth issues**: Verify your service account has access to the target folder
- **Reddit API limits**: The script respects rate limits, but heavy usage may require delays
- **Missing data**: Recent posts may have limited comments; consider running multiple times

### Manual Testing

```powershell
# Test locally
python reddit_posts_collection_script.py
```
