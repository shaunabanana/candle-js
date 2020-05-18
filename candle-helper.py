import os
import math
import json
import torch
import struct
import argparse

from pprint import pprint


parser = argparse.ArgumentParser(description='Load a PyTorch model file, and split it into browser-friendly chunks for the candle.js Loader to read over the internet.')
parser.add_argument('filename', type=str, help='PyTorch model file')
parser.add_argument('--chunksize', type=int, default=20480, help='Bytes per chunk, default to 20480, aka 20KB.')
parser.add_argument('--outpath', default=None, help='Directory to write into. Will create a folder under input directory with the name of the model if not specified.')
args = parser.parse_args()

if args.outpath is None:
    args.outpath = os.path.join(
        os.path.split(args.filename)[0],
        os.path.splitext(args.filename)[0] + '-candle'
    )

try:
    model = torch.load(args.filename)
except RuntimeError:
    model = torch.load(args.filename, map_location=torch.device('cpu'))

start = 0;
end = 0;
metadata = {
    'chunksize': args.chunksize,
    'layers': {},
};
data = b''

for name in model:
    tensor = model[name]
    layer, param = name.split('.')

    if not layer in metadata['layers']:
        metadata['layers'][layer] = {}

    # flatten tensor and convert to list
    tlist = tensor.view(-1).tolist()
    start = len(data)
    buf = struct.pack('%sf' % len(tlist), *tlist);
    end = start + len(buf)
    data += buf

    metadata['layers'][layer][param] = {
        'size': list(tensor.size()),
        'start': int(start / 4),
        'end': int(end / 4)
    }



chunks = math.ceil(len(data) / args.chunksize)
metadata['chunks'] = chunks

os.mkdir(args.outpath)
with open(os.path.join(args.outpath, 'model.json'), 'w') as f:
    f.write(json.dumps(metadata))

for c in range(chunks):
    chunk = data[c * args.chunksize: (c + 1) * args.chunksize]
    with open(os.path.join(args.outpath, '{}.bin'.format(c)), 'wb') as f:
        f.write(chunk)

pprint(metadata)
print('Model files saved to ' + args.outpath + '.')