#!/usr/bin/env python3
"""
Upload notebook to Google Drive automatically
"""

import os
import pickle
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

def upload_notebook(notebook_path: str, folder_name: str = 'Colab Notebooks'):
    """Upload notebook to Drive"""
    
    secrets_dir = Path.home() / '.openclaw' / 'secrets'
    token_file = secrets_dir / 'gdrive_token.pickle'
    
    if not token_file.exists():
        print("❌ Not authenticated! Run setup_drive_auth.py first")
        return None
    
    # Load credentials
    with open(token_file, 'rb') as token:
        creds = pickle.load(token)
    
    # Refresh if needed
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    # Build Drive API client
    service = build('drive', 'v3', credentials=creds)
    
    # Find or create folder
    folder_id = None
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields='files(id, name)'
    ).execute()
    
    folders = results.get('files', [])
    if folders:
        folder_id = folders[0]['id']
        print(f"📁 Found folder: {folder_name}")
    else:
        print(f"📁 Creating folder: {folder_name}")
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        folder_id = folder.get('id')
    
    # Upload file
    file_name = Path(notebook_path).name
    print(f"📤 Uploading: {file_name}")
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(
        notebook_path,
        mimetype='application/x-ipynb+json',
        resumable=True
    )
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    file_id = file.get('id')
    web_link = file.get('webViewLink')
    
    # Generate Colab link
    colab_link = f"https://colab.research.google.com/drive/{file_id}"
    
    print()
    print("✅ Upload complete!")
    print(f"📁 File ID: {file_id}")
    print(f"🔗 Drive link: {web_link}")
    print(f"🚀 Colab link: {colab_link}")
    print()
    print("📋 Next steps:")
    print(f"1. Open: {colab_link}")
    print("2. Runtime → Change runtime type → GPU")
    print("3. Runtime → Run all")
    
    return {
        'file_id': file_id,
        'web_link': web_link,
        'colab_link': colab_link
    }

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 upload_to_drive.py <notebook-path>")
        print()
        print("Example:")
        print("  python3 upload_to_drive.py ../notebooks/week1_training_colab.ipynb")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    
    if not Path(notebook_path).exists():
        print(f"❌ File not found: {notebook_path}")
        sys.exit(1)
    
    result = upload_notebook(notebook_path)
    sys.exit(0 if result else 1)
