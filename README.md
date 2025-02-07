# sbnd-dl-clustering

## quick setup on gpvms
This is based on local build of `wirecell`, `larwirecell` off the `sbndcode v09_91_02`
- wirecell branch: master: https://github.com/WireCell/wire-cell-toolkit/tree/master
- larwirecell branch: https://github.com/HaiwangYu/larwirecell/tree/qlmatch-ls991
- larreco tag v09_25_04 to satisfy the dependencies: https://github.com/LArSoft/larreco/tree/v09_25_04

```bash
git clone https://github.com/HaiwangYu/sbnd-dl-clustering.git dl-clustering
cd dl-clustering
source setup.sh
lar -n 1 --nskip 1 -c celltree_sbnd.fcl -s lynn-sim.root -o tmp.root
./zip-upload.sh
```


## other notes

```bash
lar -n 1 --nskip 1 -c truthdump.fcl -s lynn-sim.root -o tmp.root
```

```bash
lar -n 1 --nskip 1 -c celltree_sbnd.fcl -s lynn-sim.root -o tmp.root
python filter-json.py
./zip-upload.sh
```

Avinay-Jan27
```bash!
/pnfs/sbnd/scratch/users/abhat/v09_91_02_0201/BNB_Cosmics_NCPi0/20250126

# event 23 of
/pnfs/sbnd/scratch/users/abhat/v09_91_02_0201/BNB_Cosmics_NCPi0/20250126/17546024_4/filtered_ncpi0_events_DetSim-20250127T091644_WCLS-20250127T092259_WCLSparallel-20250127T103040_35d58c2d-24db-4fc5-a87f-b8059fb7cbe6.root

# 14
/pnfs/sbnd/scratch/users/abhat/v09_91_02_0201/BNB_Cosmics_NCPi0/20250126/73560087_100/filtered_ncpi0_events_DetSim-20250127T135940_WCLS-20250127T140112_WCLSparallel-20250127T140827_9a3827af-082f-4031-b462-521fe2af3464.root

# 28
/pnfs/sbnd/scratch/users/abhat/v09_91_02_0201/BNB_Cosmics_NCPi0/20250126/73560087_11/filtered_ncpi0_events_DetSim-20250127T113829_WCLS-20250127T114210_WCLSparallel-20250127T120545_0cc205e9-9058-451f-a342-d0a743044ce1.root

# 50
/pnfs/sbnd/scratch/users/abhat/v09_91_02_0201/BNB_Cosmics_NCPi0/20250126/73560087_112/filtered_ncpi0_events_DetSim-20250127T113836_WCLS-20250127T114301_WCLSparallel-20250127T120428_d5942858-5539-403e-adc0-8284e6ab7542.root

```
