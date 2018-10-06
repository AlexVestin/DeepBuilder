from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from multiprocessing import Pool

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
    pool = Pool(4)
    pool.map(download_page, range(2, 200))







