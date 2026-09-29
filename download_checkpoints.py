"""Download published checkpoints and verify SHA256 before installation."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['all','tokenizer','motionprior','dualtrack'],default='all')
    args=p.parse_args()
    for asset in json.loads((ROOT/'CHECKPOINTS.json').read_text())['assets']:
        if args.model!='all' and args.model!=asset['name']:continue
        destination=ROOT/asset['path']
        if destination.exists():
            if digest(destination)!=asset['sha256']:raise ValueError(f'Existing file has an unexpected hash: {destination}')
            print('Verified',asset['path']);continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.download')
        try:
            request=urllib.request.Request(asset['url'],headers={'User-Agent':'DualTrack-checkpoint-downloader'})
            with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as stream:
                while chunk:=response.read(8*1024*1024):stream.write(chunk)
            if temporary.stat().st_size!=asset['bytes'] or digest(temporary)!=asset['sha256']:
                raise ValueError('Checkpoint integrity check failed: '+asset['name'])
            temporary.replace(destination);print('Downloaded and verified',asset['path'])
        finally:
            temporary.unlink(missing_ok=True)
if __name__=='__main__':main()
