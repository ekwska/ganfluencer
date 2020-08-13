import argparse
import json
import os
import urllib.request

from tqdm import tqdm
from youtube_api import YoutubeDataApi


def download_thumbnails(data_dir, api_key):
    """ Download youtube thumbnails for the top 50 results of each search term.

    Args:
        data_dir: (str) Directory to save images to.
        api_key: (str) Your personal YouTube API key. Follow the steps to get
         one here https://developers.google.com/youtube/v3/getting-started.

    Returns: True if completed successfully.

    """
    with open(api_key, "r") as f:
        api_key = f.read()

    os.makedirs(data_dir, exist_ok=True)

    yt = YoutubeDataApi(api_key)

    search_number = 50

    search_terms = [
        "makeup",
        "grwm",
        "GRWM",
        "makeup haul",
        "plt",
        "pretty little thing",
        "first impressions",
        "beauty",
        "kbeauty",
        "instabaddie",
        "makeup tutorial",
        "everyday makeup",
        "best makeup of 2019",
        "the power of makeup",
        "glam makeup",
        "full face of",
        "eyeshadow palette",
        "beauty products",
        "makeup routine",
        "get ready with me",
        "missguided",
        "iluvsarahii",
        "jeffree star makeup",
        "nikkietutorials",
        "instagram baddie",
        "0-100 makeup transformation",
        "glow up",
        "best makeup of 2018",
        "best makeup of 2017",
        "best makeup transformations 2019",
        "best makeup transformations 2018",
        "best makeup transformations 2017",
        "full face of",
        "makeup i hate",
        "nye grwm",
        "nye glam",
        "smoky eye tutorial",
        "drugstore makeup tutorial",
        "drugstore makeup 2019",
        "drugstore makeup 2018",
        "mmmmitchell",
        "catfish makeup",
        "no makeup makeup",
        "boyfriend choose my makeup",
        "kkw x mario",
        "roxxsaurus makeup revolution",
        "nikita dragun makeup",
        "holiday makeup",
        "makeup hacks",
        "2020 grwm",
        "24hr glow up",
        "full face drugstore makeup",
        "makeup for school",
        "everyday makeup routine 2018",
        "hd brows foundation",
        "grunge makeup",
        "natural soft makeup",
        "autumn makeup",
        "jamie genevieve",
    ]

    if len(search_terms) == 0:
        print("Exiting...")
        return None

    print(
        f"Searching for the top {search_number} results ("
        f"{search_number * len(search_terms)} videos)"
    )

    for search_for in tqdm(search_terms):

        response = yt.search(
            q=search_for, max_results=search_number, parser=None
        )

        if len(response) < 1:
            continue

        for i, item in enumerate(response):
            if "medium" in item["snippet"]["thumbnails"].keys():
                thumbnail = item["snippet"]["thumbnails"]["medium"]["url"]
            elif "default" in item["snippet"]["thumbnails"].keys():
                thumbnail = item["snippet"]["thumbnails"]["default"]["url"]
            else:
                continue

            fpath = os.path.join(
                data_dir,
                f"{str(i)}_{search_for}_" + os.path.basename(thumbnail),
            )

            urllib.request.urlretrieve(thumbnail, fpath)

            with open(fpath.replace("jpg", "json"), "w") as fname:
                json.dump(item, fname)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to scrape beauty YouTube thumbnails from the "
        "YouTube API."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Path to file containing your personal api-key. Should be a text "
        "file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to directory to save YouTube thumbnails and metadata to.",
    )
    args = parser.parse_args()
    download_thumbnails(args.data_dir, args.api_key)
