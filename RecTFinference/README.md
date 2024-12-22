# Data preprocess and 3D atlas construction

## Basic data preprocess

* step00 - datapreprocessing

Stereo-seq sequencing data were preprocessed using [SAW](https://github.com/STOmics/SAW) to generation spatial gene expression matrices in [GEM format](https://stereopy.readthedocs.io/en/latest/Tutorials/IO.html#GEM).

Image files, which usually in TIFF format, generated during the stereo-seq library construction process should be ready for further process together with the GEM files.

> [!NOTE]
> All data generated in this study were deposited at CNGB Nucleotide Sequence Archive (accession code: STT0000028). Processed data can be interactively explored from our PRISTA4D database (https://db.cngb.org/stomics/prista4d; https://www.bgiocean.com/planarian). 

### GEM and image preprocessing

We provide flexible toolkit to process GEM file and ssDNA file.

#### main usage
```
./GEM3D_toolkit.py -h

Usage : GEM_toolkit.py action [options]

Actions:

---------------------------------------------------------------------

 Format coverting tools:
    gem_to_h5ad                   convert GEM into h5ad by a certain binsize.

 Affine tools:
    affine_gem                    modify the 2D coordinate in GEM(C) by user-defined affine matrix.
    affine_h5ad                   modify the 2D coordinate in GEM(C) by user-defined affine matrix.
    affine_ssdna                  affine the ssdna image by user-defined affine matrix.
    affine_txt                    affine txt like cell.mask by user-defined affine matrix.
    apply_registration        	  use registration result(with/without ROI) to update ssdna/mask/gem ...
    apply_cells                   add cells column to gem based on registered mask file.

 Region of interest(ROI) tools:
    chop_image                    chop region of interests from whole image.
    chop_gem                      chop region of interests from GEM(C).

 Mask tools:
    mask_gem                      mask GEM(C) by mask image.
    mask_h5ad                     mask h5ad data by mask image.

 Visualization tools:
    draw_heatmap                  draw heatmap of expression counts in bin1 resolution with/without cellbin and with/without ssDNA.
    image_blend                   merge image(like heatmap/annotation image) with ssDNA and border image

 Other tools:
    chop_paste                    chop or paste ssDNA image. This tools is useful for ultra-large ssDNA image.
    trakEM2_to_affine             covert trakEM2_matrix to standart affine matrix.
    split_gem                     split gem by x or y coordinate.
    merge_h5ad                    merge files of h5ad.
    gem_xy                        get xmin ymin of gem

    -----------------------------------------------------------------
    -h/--help               show this short usage

```

#### Action usage

if your need action ```xxx```'s usage, please try ```./GEM3D_toolkit.py  xxx -h```

for example:
```
 ./GEM3D_toolkit.py mask_gem -h

Usage : GEM_toolkit.py maskgem -i <input.gem> \
                               -m <mask.png>  \
                               -o <output-folder> \
                               -x [default None, xmin] \
                               -y [default None, ymin]
```


## The WACCA pipeline

* step01 - cellsegmentation: foreach slice, generate cell segmentation via ssDNA image.
* step02 - mirrorregistration: foreach slice, registrate ssDNA to the GEM coordinate system.
* step03 - seamalignment: align multiple slices to one unified 3D coordinate system.
* step04 - 3dmeshreconstruction: generate surface model by the stacked, annotated images. 

