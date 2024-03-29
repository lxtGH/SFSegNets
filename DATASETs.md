
## Here we give Five different datasets example including Mapillary Cityscapes, Camvid, KITTI and BDD.
## Consider Use the soft-link to generate these datasets.

## Mapillary Vistas Dataset

First of all, please request the research edition dataset from [here](https://www.mapillary.com/dataset/vistas/). The downloaded file is named as `mapillary-vistas-dataset_public_v1.1.zip`.

Then simply unzip the file by
```shell
unzip mapillary-vistas-dataset_public_v1.1.zip
```

The folder structure will look like:
```
Mapillary
├── config.json
├── demo.py
├── Mapillary Vistas Research Edition License.pdf
├── README
├── requirements.txt
├── training
│   ├── images
│   ├── instances
│   ├── labels
│   ├── panoptic
├── validation
│   ├── images
│   ├── instances
│   ├── labels
│   ├── panoptic
├── testing
│   ├── images
│   ├── instances
│   ├── labels
│   ├── panoptic
```
Note that, the `instances`, `labels` and `panoptic` folders inside `testing` are empty. 


## Cityscapes Dataset

### Download Dataset
First of all, please request the dataset from [here](https://www.cityscapes-dataset.com/). You need multiple files.
Both Coarse data and Fine data are used. Note that we do not use coarse data for training but we use it for uniform sampling.
```
- leftImg8bit_trainvaltest.zip
- gtFine_trainvaltest.zip
- leftImg8bit_trainextra.zip
- gtCoarse.zip
- leftImg8bit_sequence.zip    # This file is very large, 324G. You only need it if you want to run sdc_aug experiments. 
```

If you prefer to use command lines (e.g., `wget`) to download the dataset,
```
# First step, obtain your login credentials.
Please register an account at https://www.cityscapes-dataset.com/login/.

# Second step, log into cityscapes system, suppose you already have a USERNAME and a PASSWORD.
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=USERNAME&password=PASSWORD&submit=Login' https://www.cityscapes-dataset.com/login/

# Third step, download the zip files you need.
wget -c -t 0 --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

# The corresponding packageID is listed below,
1  -> gtFine_trainvaltest.zip (241MB)            md5sum: 4237c19de34c8a376e9ba46b495d6f66
2  -> gtCoarse.zip (1.3GB)                       md5sum: 1c7b95c84b1d36cc59a9194d8e5b989f
3  -> leftImg8bit_trainvaltest.zip (11GB)        md5sum: 0a6e97e94b616a514066c9e2adb0c97f
4  -> leftImg8bit_trainextra.zip (44GB)          md5sum: 9167a331a158ce3e8989e166c95d56d4
14 -> leftImg8bit_sequence.zip (324GB)           md5sum: 4348961b135d856c1777f7f1098f7266
```

### Prepare Folder Structure

Now unzip those files, the desired folder structure will look like,
```
Cityscapes
├── leftImg8bit_trainvaltest
│   ├── leftImg8bit
│   │   ├── train
│   │   │   ├── aachen
│   │   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   │   ├── aachen_000001_000019_leftImg8bit.png
│   │   │   │   ├── ...
│   │   │   ├── bochum
│   │   │   ├── ...
│   │   ├── val
│   │   ├── test
├── gtFine_trainvaltest
│   ├── gtFine
│   │   ├── train
│   │   │   ├── aachen
│   │   │   │   ├── aachen_000000_000019_gtFine_color.png
│   │   │   │   ├── aachen_000000_000019_gtFine_instanceIds.png
│   │   │   │   ├── aachen_000000_000019_gtFine_labelIds.png
│   │   │   │   ├── aachen_000000_000019_gtFine_polygons.json
│   │   │   │   ├── ...
│   │   │   ├── bochum
│   │   │   ├── ...
│   │   ├── val
│   │   ├── test
├── leftImg8bit_trainextra
│   ├── leftImg8bit
│   │   ├── train_extra
│   │   │   ├── augsburg
│   │   │   ├── bad-honnef
│   │   │   ├── ...
├── gtCoarse
│   ├── gtCoarse
│   │   ├── train
│   │   ├── train_extra
│   │   ├── val
├── leftImg8bit_sequence
│   ├── train
│   ├── val
│   ├── test
```

## CamVid Dataset

Please download and prepare this dataset according to the [tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid). The desired folder structure will look like,
```
CamVid
├── train
├── trainannot
├── val
├── valannot
├── test
├── testannot
```

## KITTI Dataset

Please download this dataset at the KITTI Semantic Segmentation benchmark [webpage](http://www.cvlibs.net/datasets/kitti/eval_semantics.php). 

Now unzip the file, the desired folder structure will look like,
```
KITTI
├── training
│   ├── image_2
│   ├── instance
│   ├── semantic
├── test
│   ├── image_2
```
There is no official training/validation split as the dataset only has `200` training samples. We randomly create three splits at [here](https://github.com/NVIDIA/semantic-segmentation/blob/master/datasets/kitti.py#L41-L44) in order to perform cross-validation. 

## BDD Dataset

Please download this dataset at BDD [webpage](https://bdd-data.berkeley.edu/)

and we use the semantic segmentation parts. Unzip the file, the desired folder structure will look like this.

```
BDD seg
├── images
│   ├── train
│   ├── val
│   ├── test
├── labels
│   ├── train
│   ├── val
```

## IDD Dataset 

Please download the IDD dataset [webpage](https://idd.insaan.iiit.ac.in/)
Also, you should tranform the json file to PNG file. refer to this [link](https://github.com/mseg-dataset/mseg-api/blob/master/download_scripts/mseg_remap_idd.sh#L17)


```
IDD seg
├── leftImg8bit/
│   ├── train
│   ├── val
│   ├── test
├── gtFine/
│   ├── train
│   ├── val
```

## Unified Driving Segmentation dataset (UDS)

*Make Sure all previous steps are correct.*

run mapillary_to_city.py

run idd_to_city.py 

to pre-pare the mapillaray and IDD dataset labels for UDS dataset (generate new labels).

The final dataset structure is: 

```
├── data/
│   ├── bdd
│   ├── cityscapes
│   ├── IDD
│   ├── mapillary
```

#### Soft Link to the data floder 
After that, you can either change the `config.py` or do the soft link according to the default path in config.

For example, 

Suppose you store your dataset at `~/username/data/Cityscapes`, please update the dataset path in `config.py`,
```
__C.DATASET.MAPILLARY_DIR = '~/username/data/Cityscapes'
``` 

For example, 
You can link the data path into current folder.

```
mkdir data 
cd data
ln -s your_cityscapes_root_data_path cityscapes
ln -s your_camvid_root_data_path camvid
```
