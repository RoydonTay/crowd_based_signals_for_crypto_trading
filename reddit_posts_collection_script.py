import praw
import os
import pandas as pd
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from praw.models import MoreComments, Submission
from dotenv import load_dotenv
import google.auth
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

load_dotenv()
client_id = os.getenv('REDDIT_CLIENT_ID')
client_secret = os.getenv('REDDIT_CLIENT_SECRET')
google_credentials_json = os.getenv('GOOGLE_CREDENTIALS')
google_refresh_token = os.getenv('GOOGLE_REFRESH_TOKEN')
google_client_id = os.getenv('GOOGLE_CLIENT_ID')
google_client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
assert client_id, "REDDIT_CLIENT_ID is not set"
assert client_secret, "REDDIT_CLIENT_SECRET is not set"

def get_google_credentials():
    """Get Google credentials from environment variables or default credentials"""
    try:
        # Use refresh token (best for GitHub Actions)
        if google_refresh_token and google_client_id and google_client_secret:
            print("Using OAuth2 refresh token credentials")
            creds = Credentials(
                token=None,
                refresh_token=google_refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=google_client_id,
                client_secret=google_client_secret,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            # Refresh the token if needed
            if not creds.valid:
                creds.refresh(Request())
            return creds
        else:
            # Fall back to default credentials
            creds, _ = google.auth.default(
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            return creds
            
    except Exception as e:
        print(f"Error getting Google credentials: {e}")
        return None

def upload_to_google_drive(file_path, folder_id=None):
    """Upload a file to Google Drive folder and return file ID
    Args: 
        file_path: Local path to the file to upload
        folder_id: Google Drive folder ID (optional)
    Returns: ID of the uploaded file or None if error
    """
    try:
        # Get credentials from environment or default
        creds = get_google_credentials()
        if not creds:
            print(f"No valid Google credentials found. Skipping upload of {file_path}")
            print("To enable Google Drive upload:")
            print("- Set GOOGLE_CREDENTIALS environment variable with service account JSON")
            print("- Or run 'gcloud auth application-default login' for default credentials")
            return None
        
        # Create drive api client
        service = build("drive", "v3", credentials=creds)
        
        # Test folder access if folder_id is provided
        if folder_id:
            try:
                folder_info = service.files().get(fileId=folder_id, fields="name,capabilities").execute()
                print(f"Folder: {folder_info.get('name')}")
                capabilities = folder_info.get('capabilities', {})
                if not capabilities.get('canAddChildren', False):
                    print("Warning: Service account may not have permission to add files to this folder")
            except HttpError as folder_error:
                print(f"Cannot access folder {folder_id}: {folder_error}")
                return None
        
        file_name = os.path.basename(file_path)
        file_metadata = {"name": file_name}
        
        # Add folder parent if specified
        if folder_id:
            file_metadata["parents"] = [folder_id]
        
        # Determine mimetype based on file extension
        if file_path.endswith('.csv'):
            mimetype = 'text/csv'
        else:
            mimetype = 'application/octet-stream'
            
        media = MediaFileUpload(file_path, mimetype=mimetype, resumable=False)
        
        # Upload file
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        
        file_id = file.get("id")
        print(f'Successfully uploaded {file_name} to Google Drive. File ID: {file_id}')
        return file_id
        
    except HttpError as error:
        print(f"Google Drive API error uploading {file_path}: {error}")
        return None
    except Exception as error:
        print(f"Unexpected error uploading {file_path}: {error}")
        return None

# Create a praw.Reddit read-only instance using OAuth2
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent='MyPythonScript/1.0 (Custom User-Agent)',
)

# Obtain a subreddit instance
subreddit_names = ["btc", "eth"]
for subreddit_name in subreddit_names:
    print(f"Collecting data from subreddit: {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)

    start = time.time()

    # Create a pandas dataframe
    df = pd.DataFrame(columns=['title', 'selftext', 'is_comment', "score", 'flair'])

    title = None
    flair = None
    for submission in subreddit.new(limit=100):
        if submission.selftext:
            title = submission.title
            selftext = str(submission.selftext)
            is_comment = False
            score = submission.score
            flair = submission.link_flair_text
            
            # Append the submission data to the dataframe
            submission_df = pd.DataFrame({'title': [title], 'selftext': [selftext], 'is_comment': [is_comment], 'score': [score], 'flair': [flair]})
            df = pd.concat([df, submission_df], axis=0, ignore_index=True)

        if submission.comments:
            # for each submission, iterate through CommentForest and parse top level comments
            for top_level_comment in submission.comments:
                if not title:
                    title = None
                if not flair:
                    flair = None

                if isinstance(top_level_comment, MoreComments):
                    continue
                selftext = str(top_level_comment.body)
                is_comment = True
                score = top_level_comment.score
                
                # Append the comment data to the dataframe
                comment_df = pd.DataFrame({'title': [title], 'selftext': [selftext], 'is_comment': [is_comment], 'score': [score], 'flair': [flair]})
                df = pd.concat([df, comment_df], axis=0, ignore_index=True)

    # Write the dataframe to a CSV file
    # Use Singapore Time (SGT, UTC+8) and include 24-hour time in the filename (YYYY-MM-DD_HHMM)
    now_sgt = datetime.now(ZoneInfo("Asia/Singapore"))
    csv_filename = f'reddit_data_{subreddit_name}_{now_sgt.strftime("%Y-%m-%d_%H%M")}.csv'
    df.to_csv(csv_filename, index=False)

    print(f"Data collection for {subreddit_name} completed in {time.time() - start:.2f} seconds.")
    print(f"Generated {csv_filename} with {len(df)} rows")
    
    # Upload to Google Drive (root) if credentials are available
    print(f"Uploading {csv_filename} to Google Drive...")
    upload_result = upload_to_google_drive(csv_filename, None)
    
    if upload_result:
        print("✓ Google Drive upload successful")
    else:
        print("✗ Google Drive upload failed")
