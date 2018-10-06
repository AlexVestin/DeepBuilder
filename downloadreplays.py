from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from multiprocessing import Pool
import sys

abs_path = "replays"

def download_page(index):
    print("downloading:", str(index))
    resp = urlopen(
        "https://lotv.spawningtool.com/zip/?coop=&pro_only=on&after_time=&tag=1&patch=&before_played_on=&before_time=&order_by=&query=&after_played_on=&p=" + str(
            index))
    zipfile = ZipFile(BytesIO(resp.read()))
    zipfile.extractall(abs_path)
    zipfile.close()

if __name__ == "__main__":
    pages_to_download = sys.argv[1]
    try:
        pages_to_download = int(pages_to_download)
    except ValueError:
        print("That's not an int")
        sys.exit(2)

    if pages_to_download < 0 or pages_to_download > 200:
        print("Bad amount")
        sys.exit(2)

    pool = Pool(4)
    pool.map(download_page, range(2, 200))







