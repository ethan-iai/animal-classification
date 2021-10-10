import os
import base64

from tqdm import tqdm
from urllib.request import urlretrieve


def create_onedrive_directdownload(onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    result_url = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return result_url

def download_rsc(url, filename, show_progress_bar=True):
    '''
    Download resource at ```url``` and save it to ```path```  
    '''
    hook = None if not show_progress_bar else _download_rsc_tqdm_hook(tqdm(unit='B', unit_scale=True))
    urlretrieve(url, filename, hook)

def _download_rsc_tqdm_hook(pbar):
    '''Wrapper for tqdm'''
    downloaded = [0]

    def update(count, block_size, total_size):
        if total_size is not None:
            pbar.total = total_size
        delta = count * block_size - downloaded[0]
        downloaded[0] = count * block_size
        pbar.update()

    return update
