from torchvision.datasets.utils import download_and_extract_archive, extract_archive
import os 
import wget
import ssl
import urllib.request

# Disable SSL verification (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context

def download_datasets(root):
    if not os.path.exists(root):
        os.makedirs(root)
    
    print("Downloading waterbirds dataset")
    waterbirds_dir = os.path.join(root, "waterbirds")
    if not os.path.isdir(waterbirds_dir):
        url = (
            "http://worksheets.codalab.org/rest/bundles/"
            "0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"
        )

        download_and_extract_archive(
            url,
            waterbirds_dir,
            filename="waterbirds.tar.gz",
        )

    print("Downloading Multinli dataset")
    multinli_dir = os.path.join(root, "multinli")
    if not os.path.isdir(multinli_dir):
        os.makedirs(multinli_dir)

        url = (
            "https://github.com/kohpangwei/group_DRO/raw/"
            "f7eae929bf4f9b3c381fae6b1b53ab4c6c911a0e/"
            "dataset_metadata/multinli/metadata_random.csv"
        )
        wget.download(url, out=multinli_dir)

        url = "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz"
        wget.download(url, out=multinli_dir)
        extract_archive(os.path.join(multinli_dir, "multinli_bert_features.tar.gz"))

        url = (
            "https://raw.githubusercontent.com/izmailovpavel/"
            "spurious_feature_learning/6d098440c697a1175de6a24"
            "d7a46ddf91786804c/dataset_files/utils_glue.py"
        )
        wget.download(url, out=multinli_dir)

download_datasets("data")