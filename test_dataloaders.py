# -*- coding: utf-8 -*-
"""
@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

from dataloaders import load_iam_metadata, load_cvl_metadata

# Load IAM
path = "/media/fredrik/UB Storage/Images/IAM"
keys1, filenames1, authors1, bbx1 = load_iam_metadata(path)

# Load CVL
path = "/media/fredrik/UB Storage/Images/CVL-Database"
keys2, authors2, filenames2 = load_cvl_metadata(path)
