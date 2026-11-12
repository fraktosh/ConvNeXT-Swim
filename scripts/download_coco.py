import os
import sys
import time
import urllib.request
import zipfile

def download_with_retry(url, filepath):
    print(f"Starting resilient download for {url}")
    while True:
        req = urllib.request.Request(url)
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        
        if file_size > 0:
            req.add_header('Range', f'bytes={file_size}-')

        try:
            response = urllib.request.urlopen(req)
            expected_total = int(response.headers.get('content-length', 0)) + file_size
            
            mode = 'ab' if file_size > 0 else 'wb'
            with open(filepath, mode) as f:
                downloaded = file_size
                while True:
                    buffer = response.read(8192 * 4) # 32KB chunks
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    percent = downloaded * 100 / expected_total if expected_total > 0 else 0
                    sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.2f} MB / {expected_total / (1024*1024):.2f} MB ({percent:.1f}%)")
                    sys.stdout.flush()
            
            # If we break out of the loop without exception, the file finished
            print(f"\nSuccessfully finished downloading {filepath}")
            return True
            
        except urllib.error.HTTPError as e:
            if e.code == 416: # Range Not Satisfiable -> already downloaded
                print(f"\n{filepath} is fully downloaded.")
                return True
            print(f"\nHTTP Error: {e.code}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"\nConnection dropped ({e}). Auto-resuming in 5 seconds...")
            time.sleep(5)

def extract_zip(filepath, extract_to):
    print(f"Extracting {filepath}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Finished extracting {filepath}")
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'coco'))
    os.makedirs(base_dir, exist_ok=True)
    
    urls = [
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/zips/train2017.zip"
    ]
    
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(base_dir, filename)
        
        extract_folder = os.path.join(base_dir, filename.split('.')[0])
        if os.path.exists(extract_folder) and len(os.listdir(extract_folder)) > 0:
            print(f"{filename} appears to be already extracted. Skipping.")
            continue
            
        success = download_with_retry(url, filepath)
        if success:
            extract_zip(filepath, base_dir)
            
    print("\nAll downloads and extractions completed successfully!")
