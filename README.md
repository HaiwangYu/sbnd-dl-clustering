# sbnd-dl-clustering

### ups env:
currently using local builds of wire-cell and larereco
```bash
# in the sl7 container:
source /exp/sbnd/app/users/yuhw/wire-cell-toolkit/setup.sh
```

### python env:
```bash
# gpvm
source /exp/sbnd/app/users/yuhw/dl-clustering/venv/bin/activate
# eaf
source /exp/sbnd/app/users/yuhw/dl-clustering/venv_eaf/bin/activate
```

### prep
setup the two batch scripts then run commands below:
```bash
# ups env
./batch_run_fcl.sh
# python env
./batch_run_labeling.sh
```

### train
```bash
./train.sh
```

### val
```bash
./val.sh
```