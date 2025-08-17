#!/usr/bin/env python3
"""
Setup script to obtain Google OAuth2 refresh token for automated uploads.

This script performs the interactive OAuth2 flow once to get a refresh token
that can be used for automated uploads in GitHub Actions.

Usage:
1. Make sure you have OAuth2 client credentials in your .env file as GOOGLE_CREDENTIALS
2. Run: python setup_google_oauth.py
3. Follow the browser prompts to authorize
4. Copy the generated tokens to your .env file and GitHub secrets

Required .env variables:
- GOOGLE_CREDENTIALS: OAuth2 client credentials JSON (the "installed" type)

Generated tokens (add these to .env and GitHub secrets):
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET  
- GOOGLE_REFRESH_TOKEN
"""

import json
import os
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Scopes required for Google Drive file upload
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def main():
    load_dotenv()
    
    google_credentials_json = os.getenv('GOOGLE_CREDENTIALS')
    if not google_credentials_json:
        print("Error: GOOGLE_CREDENTIALS not found in .env file")
        print("Please add your OAuth2 client credentials to .env file")
        return
    
    try:
        creds_dict = json.loads(google_credentials_json)
        
        if 'installed' not in creds_dict:
            print("Error: GOOGLE_CREDENTIALS must contain OAuth2 client credentials (type 'installed')")
            print("Download OAuth2 credentials from Google Cloud Console > APIs & Services > Credentials")
            return
            
        client_config = creds_dict['installed']
        
        # Create the flow using the client credentials
        flow = InstalledAppFlow.from_client_config(
            {"installed": client_config}, 
            SCOPES
        )
        
        # Run the OAuth flow
        print("Starting OAuth2 flow...")
        print("A browser window will open. Please authorize the application.")
        creds = flow.run_local_server(port=0)
        
        # Extract the tokens
        client_id = client_config['client_id']
        client_secret = client_config['client_secret']
        refresh_token = creds.refresh_token
        
        print("\n" + "="*60)
        print("SUCCESS! OAuth2 setup complete.")
        print("="*60)
        print("\nAdd these to your .env file:")
        print(f"GOOGLE_CLIENT_ID={client_id}")
        print(f"GOOGLE_CLIENT_SECRET={client_secret}")
        print(f"GOOGLE_REFRESH_TOKEN={refresh_token}")
        
        print("\nFor GitHub Actions, add these as repository secrets:")
        print(f"GOOGLE_CLIENT_ID: {client_id}")
        print(f"GOOGLE_CLIENT_SECRET: {client_secret}")
        print(f"GOOGLE_REFRESH_TOKEN: {refresh_token}")
        
        print("\nYou can now remove GOOGLE_CREDENTIALS from .env if you want,")
        print("as the refresh token method will be used instead.")
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON in GOOGLE_CREDENTIALS")
    except Exception as e:
        print(f"Error during OAuth setup: {e}")

if __name__ == '__main__':
    main()
