# Downloading datasets

This directory includes instructions and scripts for downloading ImageNet, LSUN bedrooms, and CIFAR-10 for use in this codebase.

## ImageNet-64

To download unconditional ImageNet-64, go to [this page on image-net.org](http://www.image-net.org/small/download.php) and click on "Train (64x64)". Simply download the file and unzip it, and use the resulting directory as the data directory (the `--data_dir` argument for the training script).

## Class-conditional ImageNet

For our class-conditional models, we use the official ILSVRC2012 dataset with manual center cropping and downsampling. To obtain this dataset, navigate to [this page on image-net.org](http://www.image-net.org/challenges/LSVRC/2012/downloads) and sign in (or create an account if you do not already have one). Then click on the link reading "Training images (Task 1 & 2)". This is a 138GB tar file containing 1000 sub-tar files, one per class.

Once the file is downloaded, extract it and look inside. You should see 1000 `.tar` files. You need to extract each of these, which may be impractical to do by hand on your operating system. To automate the process on a Unix-based system, you can `cd` into the directory and run this short shell script:

```
for file in *.tar; do tar xf "$file"; rm "$file"; done
```

This will extract and remove each tar file in turn.

Once all of the images have been extracted, the resulting directory should be usable as a data directory (the `--data_dir` argument for the training script). The filenames should all start with WNID (class ids) followed by underscores, like `n01440764_2708.JPEG`. Conveniently (but not by accident) this is how the automated data-loader expects to discover class labels.

## CIFAR-10

For CIFAR-10, we created a script [cifar10.py](cifar10.py) that creates `cifar_train` and `cifar_test` directories. These directories contain files named like `truck_49997.png`, so that the class name is discernable to the data loader.

The `cifar_train` and `cifar_test` directories can be passed directly to the training scripts via the `--data_dir` argument.

## LSUN bedroom

To download and pre-process LSUN bedroom, clone [fyu/lsun](https://github.com/fyu/lsun) on GitHub and run their download script `python3 download.py bedroom`. The result will be an "lmdb" database named like `bedroom_train_lmdb`. You can pass this to our [lsun_bedroom.py](lsun_bedroom.py) script like so:

```
python lsun_bedroom.py bedroom_train_lmdb lsun_train_output_dir
```

This creates a directory called `lsun_train_output_dir`. This directory can be passed to the training scripts via the `--data_dir` argument.
