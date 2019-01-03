# ImageNet-to-TFrecord

This script is a revised version of [TensorFlow-Slim's] (https://github.com/tensorflow/models/tree/master/research/slim) **build_imagenet_data.py** with the difference that this targets the classification task only. Purpose of this script is to convert a set of properly arranged images from Image-Net into TF-Record format.

## Format

The Image-Net images should be in unique synset label name folders, in the following format *(below example is for validation set - 50K images)* :

n01694178  n01843065  n02037110  n02096051  n02107683   ..... n04111531  n04273569  n04456115  n04597913  n07802026

## Usage

For this example the folders mentioned above are inside a folder called "val". To convert the images into TF-Record format just run the script below *(Tested with Python2)* :

```
python build_imagenet_data.py -validation_directory val -output_directory path-of-tf-record-directory
```

To create a TF-Record from ImageNet's training set, replace `-validation_directory` with `-train_directory`.


## Output

```
[thread 0]: Processed 1000 of 50000 images in thread batch.
[thread 0]: Processed 2000 of 50000 images in thread batch.
[thread 0]: Processed 3000 of 50000 images in thread batch.
[thread 0]: Processed 4000 of 50000 images in thread batch.
...
...
[thread 0]: Processed 49000 of 50000 images in thread batch.
[thread 0]: Processed 50000 of 50000 images in thread batch.
```

The tf-record file should be inside `path-of-tf-record-directory/validation-00000-of-00001`.

## Testing

Testing on slim's pre-trained inception_v3 :

```
python eval_image_classifier.py --alsologtostderr --checkpoint_path=/pre-trained_models/inception_v3.ckpt --dataset_dir=/path-of-tf-record-directory/ --dataset_split_name=validation  --model_name=inception_v3
```

###### Top-1 Accuracy = 0.7798	| Top-1 Recall = 0.93942



Testing on slim's pre-trained resnet_v1_50 :

```
python eval_image_classifier.py --alsologtostderr --checkpoint_path=/pre-trained_models/inception_v3.ckpt --dataset_dir=/path-of-tf-record-directory/ --dataset_split_name=validation  --labels_offset=1 --model_name=resnet_v1_50.ckpt
```

###### Top-1 Accuracy = 0.75202	| Top-1 Recall = 0.92194

More information about slim here : https://github.com/tensorflow/models/tree/master/research/slim
