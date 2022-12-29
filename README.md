# SECLA & SECLA-B (WACV 2023)

This is the official implementation of "[Weakly Supervised Face Naming with Symmetry-Enhanced Contrastive Loss](https://arxiv.org/abs/2210.08957)" at WACV 2023

## Preperation

The packages used in our experiments are listed in requirement.txt

    conda create -n facenaming python=3.7
    source activate facenaming
    pip install -r requirements.txt

## Data

### Augmented LFW

We use the augmented LFW dataset as in [Cross-Media Alignment of Names and Faces](https://ieeexplore.ieee.org/document/5332299)

If you cannot find the dataset online, please contact us. The dictionary for reading the dataset is provided in /Berg

### Celebrity Together

You can find Celeberty Together [here](https://www.robots.ox.ac.uk/~vgg/data/celebrity_together/).

You can download the dictionary I made for Celebrity Together [here](https://drive.google.com/drive/folders/1GSZrpFgS9Yv1274kXpGOIA1B4Z3fhnAL?usp=sharing). 

    mkdir CelebrityTo # make the dir and download the dict there

## Train

### SECLA

You can train SECLA model using the command

    bash run_face_align.sh # LFW data
    bash run_face_align_celeb.sh # Celebrity Together data

Remember to replace OUTPUTDIR and DATADIR accordingly.


### SECLA-B

You can train SECLA-B model using the command

    bash run_face_align_incre.sh # LFW data
    bash run_face_align_celeb_incre.sh # Celebrity Together data

Remember to replace OUTPUTDIR and DATADIR accordingly.

### CharacterBERT

If you are interested in using CharacterBERT as backbone, please check [here](https://github.com/helboukkouri/character-bert). You can clone it under /models directory.

## Test

You can calculate metrics using

    bash run_test.sh

## Note

I'll further clean the code when I have time. Thanks for your understanding:)

## Citation

If you are interested in our work, please cite it as:

    @InProceedings{Qu_2023_WACV,
    author    = {Qu, Tingyu and Tuytelaars, Tinne and Moens, Marie-Francine},
    title     = {Weakly Supervised Face Naming With Symmetry-Enhanced Contrastive Loss},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3505-3514}
    }
