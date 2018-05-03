# -*- coding: utf-8 -*-
"""
Code for loading CVL and IAM

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

import os
import os.path


def load_cvl_metadata(path, savefile=None):
    cropped_path = "cvl-database-cropped-1-1"
    # List image file names
    filepath = os.path.join(path, cropped_path)
    assert os.path.exists(filepath)
    files = [f for f in os.listdir(filepath) if f[-3:].lower() == 'tif']
    authors = [int(f.split("-")[0]) for f in files]
    filenames = [os.path.join(cropped_path, f) for f in files]
    keys = [f[:f.find("-", f.find("-")+1)] for f in files]
    sets = [int(author > 50) for author in authors]
    return keys, authors, filenames, sets


def load_iam_metadata(path, savefile=None):
    import os.path
#    import numpy as np
#    savefile = os.path.join(path, "cvl_metadata.npz")
#    if os.path.exists(savefile):
#        return np.load(savefile)
    # Set up paths
    forms_path = os.path.join(path, "ascii", "forms.txt")
    lines_path = os.path.join(path, "ascii", "lines.txt")
    form_paths = [os.path.join(path, p) for p in ['formsA-D', 'formsE-H',
                                                  'formsI-Z']]
    # Load data from forms.txt and check for png files
    authors = list()
    filenames = list()
    keys = list()
    assert os.path.exists(forms_path)
    with open(forms_path, 'r') as file:
        for line in file:
            if line[0] != '#':
                splitline = line.split(" ")
                fn = splitline[0] + ".png"
                for p in form_paths:
                    f = os.path.join(p, fn)
                    if os.path.exists(f):
                        authors.append(int(splitline[1]))
                        fsplit = f.split("/")
                        filenames.append(os.path.join(fsplit[-2], fsplit[-1]))
                        keys.append(splitline[0])
                        break

    # Local function for merging bounding boxes
    def merge_bbx(bbx1, bbx2):
        new = [-1]*4
        new[0] = min(bbx1[0], bbx2[0])
        new[1] = min(bbx1[1], bbx2[1])
        new[2] = max(bbx1[0]+bbx1[2], bbx2[0]+bbx2[2]) - new[0]
        new[3] = max(bbx1[1]+bbx1[3], bbx2[1]+bbx2[3]) - new[1]
        return tuple(new)
    # Load data from lines.txt for page segmentation
    bbx_dict = dict()
    assert os.path.exists(lines_path)
    with open(lines_path, 'r') as file:
        for line in file:
            if line[0] != '#':
                splitline = line.split(" ")
                key = splitline[0][:-3]
                bbx = tuple([int(s) for s in splitline[4:8]])
                if key in bbx_dict:
                    bbx_dict[key] = merge_bbx(bbx_dict[key], bbx)
                else:
                    bbx_dict[key] = bbx
    bbx = [bbx_dict[k] for k in keys]
    return keys, filenames, authors, bbx


def load_json(filename):
    assert os.path.exists(filename)
    import gzip
    f = gzip.open(filename, 'r')
    import json
    data = json.loads(f.read().decode('utf-8'))
    f.close()
    return data


def save_json(self, filename, data):
    filename = os.path.join(self.datapath, "metadata.json.gz")
    import gzip
    f = gzip.open(filename, 'w')
    import json
    f.write(json.dumps(data, sort_keys=True, indent=2,
                       separators=(',', ': ')).encode('utf-8'))
    f.close()
