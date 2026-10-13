import urllib.request
import tarfile
import os

def download_and_extract():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
    download_path = "reuters21578.tar.gz"
    target_dir = os.path.join("data", "raw")

    print(f"Downloading Reuters-21578 from {url}...")
    urllib.request.urlretrieve(url, download_path)
    print("Download complete.")

    print(f"Extracting .sgm files to {target_dir}...")
    with tarfile.open(download_path, "r:gz") as tar:
        # Extract only the .sgm files
        for member in tar.getmembers():
            if member.name.endswith(".sgm"):
                tar.extract(member, path=target_dir)

    print("Extraction complete. Cleaning up...")
    os.remove(download_path)
    print(f"Success! Your data is ready in the {target_dir} folder.")

if __name__ == "__main__":
    download_and_extract()