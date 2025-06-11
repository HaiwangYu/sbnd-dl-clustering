input=$1
nevents=$2
lar -n $nevents --nskip 0 -c celltree_sbnd.fcl -s $input -o tmp.root | tee log
lar -n $nevents --nskip 0 -c wcls-img-clus.fcl -s $input -o tmp.root | tee -a log
