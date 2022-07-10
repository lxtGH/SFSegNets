"""
# File taken from https://github.com/mcordts/cityscapesScripts/
# License File Available at:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# ----------------------
# The Cityscapes Dataset
# ----------------------
#
#
# License agreement
# -----------------
#
# This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
#
# 1. That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, we (Daimler AG, MPI Informatics, TU Darmstadt) do not accept any responsibility for errors or omissions.
# 2. That you include a reference to the Cityscapes Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the Cityscapes website.
# 3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
# 4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
# 5. That all rights not expressly granted to you are reserved by us (Daimler AG, MPI Informatics, TU Darmstadt).
#
#
# Contact
# -------
#
# Marius Cordts, Mohamed Omran
# www.cityscapes-dataset.net

"""
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        ,
    'id'          ,
    'csId'        ,

    'csTrainId'   ,

    'level4Id'    ,
    'level3Id'    ,
    'category',
    'level2Id'    ,
    'level1Id'    ,

    'hasInstances',
    'ignoreInEval',
    'color'       ,
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!


labels = [
    #       name  id   csId               csTrainId        level4id        level3Id  category           level2Id      level1Id  hasInstances   ignoreInEval   color
    Label('road', 0,  7,                     0, 0, 0, 'drivable', 0, 0, False, False, (128, 64, 128)),
    Label('parking', 1, 9,                  255, 1, 1, 'drivable', 1, 0, False, False, (250, 170, 160)),
    Label('drivable fallback', 2, 255,      255, 2, 1, 'drivable', 1, 0, False, False, (81, 0, 81)),
    Label('sidewalk', 3, 8,                 1, 3, 2, 'non-drivable', 2, 1, False, False, (244, 35, 232)),
    Label('rail track', 4,10,              255, 3, 3, 'non-drivable', 3, 1, False, False, (230, 150, 140)),
    Label('non-drivable fallback', 5, 255,  9, 4, 3, 'non-drivable', 3, 1, False, False, (152, 251, 152)),
    Label('person', 6, 24,                  11, 5, 4, 'living-thing', 4, 2, True, False, (220, 20, 60)),
    Label('animal', 7, 255,                 255, 6, 4, 'living-thing', 4, 2, True, True, (246, 198, 145)),
    Label('rider', 8, 25,                   12, 7, 5, 'living-thing', 5, 2, True, False, (255, 0, 0)),
    Label('motorcycle', 9, 32,              17, 8, 6, '2-wheeler', 6, 3, True, False, (0, 0, 230)),
    Label('bicycle', 10, 33,                18, 9, 7, '2-wheeler', 6, 3, True, False, (119, 11, 32)),
    Label('autorickshaw', 11, 255,         255, 10, 8, 'autorickshaw', 7, 3, True, False, (255, 204, 54)),
    Label('car', 12, 26,                   13, 11, 9, 'car', 7, 3, True, False, (0, 0, 142)),
    Label('truck', 13, 27,                 14, 12, 10, 'large-vehicle', 8, 3, True, False, (0, 0, 70)),
    Label('bus', 14, 28,                   15, 13, 11, 'large-vehicle', 8, 3, True, False, (0, 60, 100)),
    Label('caravan', 15, 29,               255, 14, 12, 'large-vehicle', 8, 3, True, True, (0, 0, 90)),
    Label('trailer', 16, 30,               255, 15, 12, 'large-vehicle', 8, 3, True, True, (0, 0, 110)),
    Label('train', 17, 31,                 16, 15, 12, 'large-vehicle', 8, 3, True, True, (0, 80, 100)),
    Label('vehicle fallback', 18, 355,     255, 15, 12, 'large-vehicle', 8, 3, True, False, (136, 143, 153)),
    Label('curb', 19, 255,                 255, 16, 13, 'barrier', 9, 4, False, False, (220, 190, 40)),
    Label('wall', 20, 12,                  3, 17, 14, 'barrier', 9, 4, False, False, (102, 102, 156)),
    Label('fence', 21, 13,                 4, 18, 15, 'barrier', 10, 4, False, False, (190, 153, 153)),
    Label('guard rail', 22, 14,            255, 19, 16, 'barrier', 10, 4, False, False, (180, 165, 180)),
    Label('billboard', 23, 255,            255, 20, 17, 'structures', 11, 4, False, False, (174, 64, 67)),
    Label('traffic sign', 24, 20,           7, 21, 18, 'structures', 11, 4, False, False, (220, 220, 0)),
    Label('traffic light', 25, 19,          6, 22, 19, 'structures', 11, 4, False, False, (250, 170, 30)),
    Label('pole', 26, 17,                   5, 23, 20, 'structures', 12, 4, False, False, (153, 153, 153)),
    Label('polegroup', 27, 18,               255, 23, 20, 'structures', 12, 4, False, False, (153, 153, 153)),
    Label('obs-str-bar-fallback', 28, 255,  255, 24, 21, 'structures', 12, 4, False, False, (169, 187, 214)),
    Label('building', 29, 11,                2, 25, 22, 'construction', 13, 5, False, False, (70, 70, 70)),
    Label('bridge', 30, 15,                 255, 26, 23, 'construction', 13, 5, False, False, (150, 100, 100)),
    Label('tunnel', 31, 16,                 255, 26, 23, 'construction', 13, 5, False, False, (150, 120, 90)),
    Label('vegetation', 32, 21,             8, 27, 24, 'vegetation', 14, 5, False, False, (107, 142, 35)),
    Label('sky', 33, 23,                    10, 28, 25, 'sky', 15, 6, False, False, (70, 130, 180)),
    Label('fallback background', 34, 255,   255, 29, 25, 'object fallback', 15, 6, False, False, (169, 187, 214)),
    Label('unlabeled', 35, 0,               255, 255, 255, 'void', 255, 255, False, True, (0, 0, 0)),
    Label('ego vehicle', 36, 1,             255, 255, 255, 'void', 255, 255, False, True, (0, 0, 0)),
    Label('rectification border', 37, 2,    255, 255, 255, 'void', 255, 255, False, True, (0, 0, 0)),
    Label('out of roi', 38, 3,              255, 255, 255, 'void', 255, 255, False, True, (0, 0, 0)),
    Label('license plate', 39, 255,         255, 255, 255, 'vehicle', 255, 255, False, True, (0, 0, 142)),

]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.csTrainId : label for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.csTrainId for label in labels   }
# trainId to label object
trainId2name   = { label.csTrainId : label.name for label in labels   }
trainId2color  = { label.csTrainId : label.color for label in labels      }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' )))
    print(("    " + ('-' * 98)))
    for label in labels:
        print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval )))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print(("ID of label '{name}': {id}".format( name=name, id=id )))

    # Map from ID to label
    category = id2label[id].category
    print(("Category of label with ID '{id}': {category}".format( id=id, category=category )))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print(("Name of label with trainID '{id}': {name}".format( id=trainId, name=name )))
