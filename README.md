# fibsem_segm
Segmentation of volume EM data

## Using jupyterhub to view data
Go to `jupyterhub.embl.de`
Choose   
**Image Analysis GPU Desktop**

Fiji, ilastik, napari, CellPose. 8 CPU, 16G GPU P100, and 64G RAM


## Set up environment for correcting segmentations

Install miniconda: https://docs.conda.io/en/latest/miniconda.html#linux-installers

```
    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
    bash Miniconda3-py39_4.11.0-Linux-x86_64.sh 
```

Then follow what the installer says.  
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no] - yes


For correction of labels we will need `napari` and `z5py` libraries.



