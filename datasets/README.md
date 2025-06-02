# Prepare datasets

If you have a dataset directory, you could use os environment variable named `YOLOX_DATADIR`. Under this directory, YOLOX will look for datasets in the structure described below, if needed.
```
$YOLOX_DATADIR/
  COCO/
```
You can set the location for builtin datasets by
```shell
export YOLOX_DATADIR=/path/to/your/datasets
```
If `YOLOX_DATADIR` is not set, the default value of dataset directory is `./datasets` relative to your current working directory.

## Expected dataset structure for [COCO detection](https://cocodataset.org/#download):

```
COCO/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

The "2017" naming convention comes from the COCO dataset structure. COCO (Common Objects in
Context) dataset uses year-based naming for its data splits, with "train2017" and "val2017" being
the standard directory names for the 2017 version of the COCO dataset.

You can use the 2014 version of the dataset as well.


