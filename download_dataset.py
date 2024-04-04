import requests
from tqdm import tqdm
import zipfile
import os
import shutil

def extract_zip_with_progress(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get the total number of files in the ZIP archive
        total_files = len(zip_ref.infolist())
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(total=total_files, desc="Extracting", unit="files")
        
        # Extract each file individually and update the progress bar
        for file_info in zip_ref.infolist():
            zip_ref.extract(file_info, extract_to)
            progress_bar.update(1)
        
        progress_bar.close()

def copy_folders(source_dir, destination_dir):
    # Get a list of all folders in the source directory
    folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Copy each folder and its contents to the destination directory
    for folder in folders:
        source_path = os.path.join(source_dir, folder)
        destination_path = os.path.join(destination_dir, folder)
        shutil.copytree(source_path, destination_path)

def download_file(url, destination):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the total file size in bytes
        total_size = int(response.headers.get('content-length', 0))
        # Initialize a progress bar with the total file size
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        # Open the destination file in binary write mode
        with open(destination, 'wb') as f:
            # Iterate over the response content in chunks and write to file
            for data in response.iter_content(chunk_size=1024):
                # Write data to file
                f.write(data)
                # Update the progress bar with the size of the data written
                progress_bar.update(len(data))
        
        # Close the progress bar
        progress_bar.close()
        print("File downloaded successfully")

        extract_zip_with_progress(destination, 'extracted_content')
        copy_folders('./extracted_content/Data', './video')
        shutil.rmtree('./extracted_content')
        os.remove('./data.zip')
    else:
        print("Failed to download file")


url = 'https://storage.googleapis.com/kaggle-data-sets/1487082/2456853/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240403T085228Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b7e19204c46f35356a2045b27af5de0632e158fbda53a925800779bd64895d535e4edde94642435ac7e6f1ca50f8d601d6cb5d3bad513ff221e8e92a2039f2abdfc4ba21a9b506699225ccf2648a51514f0f0219bd2fd1e82c4a3662f4037ab400fbdcf3407a3600d8a7899626646bc8f155b74ffd536507712e58f39e3813334ad240e71ff17d2ef16e0e1afc184f26a5197f989677a57855ea0e7a9cb358a58f6d9bce2585e8dc62037679c746bab0433506e27b4560b69c2652302fa239ae7709780f2caf0535129544d70f31ab8e51d2e18da24faf9fb3f533f348dd79454d5408c3bbfcbebdbab52a80a89faabdf650ce1115ce112b3e4d2718b8667e9a'
destination = 'data.zip'

if os.path.exists('./video'):
    shutil.rmtree('./video')
    os.makedirs('./video')

download_file(url, destination)