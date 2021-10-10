import os
import argparse

from utils import create_onedrive_directdownload, download_rsc

parser = argparse.ArgumentParser()
parser.add_argument("root", default="", type=str)

MODEL_NAME = "resnet152_baseline.pth.tar"
ONEDRIVE_SHARED_LINK = "https://1drv.ms/u/s!Al5BF1i8TRVbiQt2whUMwlzTn69R?e=qMr3XT"

def download(model_path):
    if not os.path.exists(model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_url = create_onedrive_directdownload(ONEDRIVE_SHARED_LINK)
    download_rsc(download_url, model_path)

if __name__ == '__main__':
    args = parser.parse_args()

    model_path = os.join(args.root, MODEL_NAME)

    download(model_path)
