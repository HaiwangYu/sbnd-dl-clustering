input=$1
nevents=$2
lar -n $nevents --nskip 0 -c wcls-img-clus.fcl -s $input -o tmp.root | tee log
lar -n $nevents --nskip 0 -c celltree_sbnd_apa0.fcl -s $input -o tmp.root | tee -a log
lar -n $nevents --nskip 0 -c celltree_sbnd_apa1.fcl -s $input -o tmp.root | tee -a log