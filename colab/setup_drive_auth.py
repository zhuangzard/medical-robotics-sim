#!/usr/bin/env python3
"""
Setup Google Drive API authentication (one-time)
"""

import os
import pickle
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Scopes - only need to upload files
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def setup_auth():
    """Setup Drive authentication"""
    
    secrets_dir = Path.home() / '.openclaw' / 'secrets'
    secrets_dir.mkdir(parents=True, exist_ok=True)
    
    token_file = secrets_dir / 'gdrive_token.pickle'
    creds_file = secrets_dir / 'gdrive_credentials.json'
    
    creds = None
    
    # Check if we already have a token
    if token_file.exists():
        print("📋 Found existing token")
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not creds_file.exists():
                print("❌ Error: credentials.json not found!")
                print()
                print("📋 To setup:")
                print("1. Go to: https://console.cloud.google.com/")
                print("2. Create new project (or select existing)")
                print("3. Enable Google Drive API")
                print("4. Create OAuth 2.0 credentials (Desktop app)")
                print("5. Download credentials.json")
                print(f"6. Save to: {creds_file}")
                print()
                return False
            
            print("🔐 Opening browser for authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_file), SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
        
        print(f"✅ Token saved to: {token_file}")
    
    print("✅ Drive API authorization complete!")
    return True

if __name__ == '__main__':
    import sys
    
    # Check dependencies
    try:
        import google_auth_oauthlib
        import google.auth
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install -q google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        print("✅ Packages installed! Please run again.")
        sys.exit(0)
    
    success = setup_auth()
    sys.exit(0 if success else 1)
