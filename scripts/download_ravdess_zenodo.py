import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_and_extract():
    url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    
    root_dir = Path(__file__).resolve().parent.parent
    target_dir = root_dir / "data" / "raw" / "ravdess"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = target_dir / "Audio_Speech_Actors_01-24.zip"
    
    print(f"Downloading RAVDESS from {url}...")
    print(f"Saving to {zip_path}...")
    
    # Custom reporthook to show download progress
    def reporthook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 1e2 / total_size
            s = f"\rProgress: {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB of {total_size / (1024*1024):.1f} MB)"
            sys.stdout.write(s)
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rProgress: {read_so_far / (1024*1024):.1f} MB")
            sys.stdout.flush()
            
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\nDownload complete. Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        print(f"Extraction complete. Files extracted to {target_dir}")
        
        # Clean up zip file
        os.remove(zip_path)
        print("Cleaned up ZIP file.")
        
    except Exception as e:
        print(f"\nError: {e}")
        if zip_path.exists():
            os.remove(zip_path)

if __name__ == "__main__":
    download_and_extract()
