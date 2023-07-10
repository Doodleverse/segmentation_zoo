
# standard imports
import os, json
import asyncio
import platform

# external imports
from glob import glob

# import tqdm
from tqdm.auto import tqdm as auto_tqdm
import tqdm.asyncio
import zipfile
import requests
import aiohttp
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_model_dir(parent_directory: str, dir_name: str) -> str:
    """returns full path to directory named dir_name and if it doesn't exist
    creates new directory dir_name within parent directory

    Args:
        parent_directory (str): directory to create new directory dir_name within
        dir_name (str): name of directory to get full path to

    Returns:
        str: full path to dir_name directory
    """
    new_dir = os.path.join(parent_directory, dir_name)
    if not os.path.isdir(new_dir):
        print(f"Creating {new_dir}")
        os.mkdir(new_dir)
    return new_dir


def request_available_files(zenodo_id: str) -> list:
    """returns list of available downloadable files for zenodo_id

    Args:
        zenodo_id (str): id of zenodo release

    Returns:
        list: list of available files downloadable for zenodo_id
    """
    # Send request to zenodo for selected model by zenodo_id
    root_url = "https://zenodo.org/api/records/" + zenodo_id
    r = requests.get(root_url)
    # get list of all files associated with zenodo id
    js = json.loads(r.text)
    files = js["files"]
    return files


def is_zipped_release(files: list) -> bool:
    """returns True if zenodo id contains 'rgb.zip' file otherwise returns false

    Args:
        files (list): list of available files for download

    Returns:
        bool: returns True if zenodo id contains 'rgb.zip' file otherwise returns false
    """
    zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
    if zipped_model_list == []:
        return False
    return True


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def get_url_dict_to_download(models_json_dict: dict) -> dict:
    """Returns dictionary which contains
    paths to save downloaded files to matched with urls to download files

    each key in returned dictionary contains a full path to a file
    each value in returned dictionary contains url to download file
    ex.
    {'C:\Home\Project\file.json':"https://website/file.json"}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """
    url_dict = {}
    for save_path, link in models_json_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
        json_filepath = save_path.replace("_fullmodel.h5", ".json")
        if not os.path.isfile(json_filepath):
            json_link = link.replace("_fullmodel.h5", ".json")
            url_dict[json_filepath] = json_link

    return url_dict


async def fetch(session: aiohttp.client.ClientSession, url: str, save_path: str) -> None:
    """downloads the file at url to be saved at save_path. Generates tqdm progress bar
    to track download progress of file

    Args:
        session (aiohttp.client.ClientSession): session with server that files are downloaded from
        url (str): url to file to download
        save_path (str): full path where file will be saved
    """
    model_name = url.split("/")[-1]
    chunk_size: int = 2048
    async with session.get(url, timeout=600) as r:
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with auto_tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_name}",
                    initial=0,
                    ascii=False,
                ) as pbar:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(save_path, "wb") as fd:
                async for chunk in r.content.iter_chunked(chunk_size):
                    fd.write(chunk)


async def fetch_all(session: aiohttp.client.ClientSession, url_dict: dict) -> None:
    """concurrently downloads all urls in url_dict within a single provided session

    Args:
        session (aiohttp.client.ClientSession): session with server that files are downloaded from
        url_dict (dict): dictionary with keys as full path to file to download and value being ulr to
        file to download
    """
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(fetch(session, url, save_path))
        tasks.append(task)
    await tqdm.asyncio.asyncio.gather(*tasks)


async def async_download_urls(url_dict: dict) -> None:
    # error raised if downloads dont's complete in 600 seconds (10 mins)
    async with aiohttp.ClientSession(raise_for_status=True, timeout=600) as session:
        await fetch_all(session, url_dict)


def run_async_download(url_dict: dict) -> None:
    """concurrently downloads all thr urls in url_dict

    Args:
        url_dict (dict): dictionary with keys as full path to file to download and value being ulr to
        file to download
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # wait for async downloads to complete
    asyncio.run(async_download_urls(url_dict))


def download_zip(url: str, save_path: str, chunk_size: int = 128) -> None:
    """Downloads the zipped model from the given url to the save_path location.
    Args:
        url (str): url to zip directory  to download
        save_path (str): directory to save model
        chunk_size (int, optional):  Defaults to 128.
    """
    # make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))
        with open(save_path, "wb") as fd:
            with auto_tqdm(
                total=total_length,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading Model",
                initial=0,
                ascii=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
                    pbar.update(len(chunk))


def download_zipped_model(model_direc: str, url: str) -> str:
    """download a zipped model from zenodo located at url

    'rgb.zip' is name of directory containing model online
    this function is used to download older styles of zenodo releases

    Args:
        model_direc (str): full path to directory to save model to
        url (str): url to download model from

    Returns:
        str: full path to unzipped model directory
    """
    # 'rgb.zip' is name of directory containing model online
    filename = "rgb"
    # outfile: full path to directory containing model files
    # example: 'c:/model_name/rgb'
    outfile = model_direc + os.sep + filename
    if os.path.exists(outfile):
        print(f"\n Found model weights directory: {os.path.abspath(outfile)}")
    # if model directory does not exist download zipped model from Zenodo
    if not os.path.exists(outfile):
        print(f"\n Downloading to model weights directory: {os.path.abspath(outfile)}")
        zip_file = filename + ".zip"
        zip_folder = model_direc + os.sep + zip_file
        print(f"Retrieving model {url} ...")
        download_zip(url, zip_folder)
        print(f"Unzipping model to {model_direc} ...")
        with zipfile.ZipFile(zip_folder, "r") as zip_ref:
            zip_ref.extractall(model_direc)
        print(f"Removing {zip_folder}")
        os.remove(zip_folder)

    # set weights dir to sub directory (rgb) containing model files
    model_direc = os.path.join(model_direc, "rgb")

    # Ensure all files are unzipped
    with os.scandir(model_direc) as it:
        for entry in it:
            if entry.name.endswith(".zip"):
                with zipfile.ZipFile(entry, "r") as zip_ref:
                    zip_ref.extractall(model_direc)
                os.remove(entry)
    return model_direc


def download_BEST_model(files: list, model_direc: str) -> None:
    """downloads best model from zenodo.

    Args:
        files (list): list of available files for zenodo release
        model_direc (str): directory of model to download

    Raises:
        FileNotFoundError: if model filename in 'BEST_MODEL.txt' does not
        exist online
    """
    # dictionary to hold urls and full path to save models at
    models_json_dict = {}
    # retrieve best model text file
    best_model_json = [f for f in files if f["key"].strip() == "BEST_MODEL.txt"][0]
    best_model_txt_path = os.path.join(model_direc, "BEST_MODEL.txt")
    # if BEST_MODEL.txt file not exist download it
    if not os.path.isfile(best_model_txt_path):
        download_url(
            best_model_json["links"]["self"],
            best_model_txt_path,
        )

    # read in BEST_MODEL.txt file
    with open(best_model_txt_path) as f:
        best_model_filename = f.read().strip()

    print(f"Best Model filename: {best_model_filename}")
    # check if json and h5 file in BEST_MODEL.txt exist
    model_json = [f for f in files if f["key"].strip() == best_model_filename]
    if model_json == []:
        FILE_NOT_ONLINE_ERROR = f"File {best_model_filename} not found online. Raise an issue on Github"
        raise FileNotFoundError(FILE_NOT_ONLINE_ERROR)
    # path to save model
    outfile = os.path.join(model_direc, best_model_filename)
    # path to save file and json data associated with file saved to dict
    models_json_dict[outfile] = model_json[0]["links"]["self"]
    url_dict = get_url_dict_to_download(models_json_dict)
    # if any files are not found locally download them asynchronous
    if url_dict != {}:
        run_async_download(url_dict)


def download_ENSEMBLE_model(files: list, model_direc: str) -> None:
    """downloads all models from zenodo.

    Args:
        files (list): list of available files for zenodo release
        model_direc (str): directory of model to download
    """
    # dictionary to hold urls and full path to save models at
    models_json_dict = {}
    # list of all models
    all_models = [f for f in files if f["key"].endswith(".h5")]
    # check if all h5 files in files are in model_direc
    for model_json in all_models:
        outfile = model_direc + os.sep + model_json["links"]["self"].split("/")[-1]
        # path to save file and json data associated with file saved to dict
        models_json_dict[outfile] = model_json["links"]["self"]
    url_dict = get_url_dict_to_download(models_json_dict)
    # if any files are not found locally download them asynchronous
    if url_dict != {}:
        run_async_download(url_dict)



def get_weights_list(model_choice: str, weights_direc: str) -> list:
    """Returns of list of full paths to weights files(.h5) within weights_direc

    Args:
        model_choice (str): 'ENSEMBLE' or 'BEST'
        weights_direc (str): full path to directory containing model weights

    Returns:
        list: list of full paths to weights files(.h5) within weights_direc
    """
    if model_choice == "ENSEMBLE":
        return glob(weights_direc + os.sep + "*.h5")
    elif model_choice == "BEST":
        with open(weights_direc + os.sep + "BEST_MODEL.txt") as f:
            w = f.readlines()
        return [weights_direc + os.sep + w[0]]

