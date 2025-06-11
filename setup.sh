source /cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh
# setup sbndcode v09_91_02 -q e26:prof
# source /exp/sbnd/app/users/yuhw/larsoft/v09_91_02/localProducts_larsoft_v09_91_02_e26_prof/setup
# setup sbndcode v10_04_03 -q e26:prof
# source /exp/sbnd/app/users/yuhw/larsoft/v10_04_03/localProducts_larsoft_v10_04_03_e26_prof/setup
setup sbndcode v10_04_07 -q e26:prof
source /exp/sbnd/app/users/yuhw/larsoft/v10_04_07/localProducts_larsoft_v10_04_07_e26_prof/setup
mrbsetenv
mrbslp


# for compiling larwirecell
# path-prepend /exp/sbnd/app/users/yuhw/opt CMAKE_PREFIX_PATH


path-remove ()
{
    local IFS=':';
    local NEWPATH;
    local DIR;
    local PATHVARIABLE=${2:-PATH};
    for DIR in ${!PATHVARIABLE};
    do
        if [ "$DIR" != "$1" ]; then
            NEWPATH=${NEWPATH:+$NEWPATH:}$DIR;
        fi;
    done;
    export $PATHVARIABLE="$NEWPATH"
}

path-prepend ()
{
    path-remove "$1" "$2";
    local PATHVARIABLE="${2:-PATH}";
    export $PATHVARIABLE="$1${!PATHVARIABLE:+:${!PATHVARIABLE}}"
}

path-append ()
{
    path-remove "$1" "$2";
    local PATHVARIABLE="${2:-PATH}";
    export $PATHVARIABLE="${!PATHVARIABLE:+${!PATHVARIABLE}:}$1"
}

path-prepend /exp/sbnd/app/users/yuhw/opt/lib/ LD_LIBRARY_PATH

path-prepend /exp/sbnd/app/users/yuhw/wire-cell-toolkit/cfg WIRECELL_PATH
path-prepend /exp/sbnd/app/users/yuhw/wire-cell-data WIRECELL_PATH

#rs
#export PS1=(app)$PS1
