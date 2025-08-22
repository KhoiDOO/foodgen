import os
import csv
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import argparse

# Get the directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))
# Path to source.csv
csv_path = os.path.join(base_dir, 'source.csv')
# Path to output folder
output_dir = os.path.join(base_dir, 'src')
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser(description="Download and resize images from URLs in a CSV file.")
parser.add_argument('--num-cores', type=int, default=2, help='Number of parallel processes to use')
args = parser.parse_args()
NUM_CORES = args.num_cores

urls = []
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        url = row.get('image_url')
        if url:
            urls.append(url)

def process_url(url):
    filename = os.path.basename(url)
    out_path = os.path.join(output_dir, filename)
    curl_cmd = f"curl -f -L -o '{out_path}' '{url}' > /dev/null 2>&1"
    ret = os.system(curl_cmd)
    if ret != 0:
        return
    try:
        with Image.open(out_path) as img:
            img = img.resize((256, 256), Image.LANCZOS)
            img.save(out_path)
    except Exception:
        return

if __name__ == "__main__":
    with Pool(NUM_CORES) as pool:
        list(tqdm(pool.imap_unordered(process_url, urls), total=len(urls), desc='Downloading images'))
