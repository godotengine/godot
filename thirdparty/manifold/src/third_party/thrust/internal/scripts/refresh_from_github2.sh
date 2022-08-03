branch="main"

while getopts "hb:c:" opt; do
    case $opt in
        h)
        echo "Usage: $0 [-h] [-b <github_branch_name>] -c <P4_changelist>"
        exit 1
        ;;

        b)
        branch=$OPTARG
        ;;

        c)
        changelist=$OPTARG
        ;;

        /?)
        echo "Invalid option: -$OPTARG" >&2;
        exit 1
        ;;

        :)
        echo "Option -$OPTARG requires an argument";
        exit 1
        ;;
    esac
done

if [ "$changelist" == "" ]; then
    echo "Missing required option -c to specify P4 changelist to put changed files into"
    exit 1
fi

# Cause script to exit on any command that results in an error
set -e

echo "Downloading thrust code from the $branch branch into /tmp/thrust-${branch}"
rm -rf /tmp/thrust-${branch}
git clone -q git://github.com/NVIDIA/thrust.git -b ${branch} /tmp/thrust-${branch}

cd `dirname $0`/../..
echo "Changed current directory to `pwd`"

vulcan_files=`echo *.vlcc *.vlct` 
logdir=`mktemp -d /tmp/tmp.XXXXXXXX`
echo "Logging p4 command outputs to temporary directory $logdir"
for i in *; do
    if [[ "$i" != "internal" && "$i" != "Makefile" ]]; then
        ii="$i";
        if [ -d $i ]; then ii="$i/..."; fi
        echo "Reverting, force syncing, and then removing $ii"
        p4 revert $ii >> $logdir/$i.revert.log 2>&1
        p4 sync -f $ii >> $logdir/$i.sync.log 2>&1
        rm -rf $i
    fi
done

echo "Copying downloaded thrust code to p4 client"
cp -R /tmp/thrust-${branch}/* .
find . -name ".gitignore" | xargs -n 1 rm

echo "Checking if version has been bumped"
new_version=`grep "#define THRUST_VERSION" thrust/version.h | sed -e "s/#define THRUST_VERSION //"`
old_version=`p4 print thrust/version.h | grep "#define THRUST_VERSION" | sed -e "s/#define THRUST_VERSION //"`
if [ "$new_version" != "$old_version" ]; then
    p4 edit internal/test/version.gold
    new_version_print="$(( $new_version / 100000 )).$(( ($new_version / 100) % 1000 )).$(( $new_version % 100 ))"
    sed -e "s/v[0-9\.][0-9\.]*/v${new_version_print}/" internal/test/version.gold > internal/test/version.gold.tmp
    mv internal/test/version.gold.tmp internal/test/version.gold
    echo "Updated version.gold to version $new_version_print"
else
    echo "Version has not changed"
fi

echo "Reconciling changed code into changelist $changelist"
p4 reconcile -c $changelist ... >> $logdir/reconcile.log 2>&1
p4 revert -c $changelist Makefile $vulcan_files internal/... >> $logdir/internal_files_revert.log 2>&1

echo "Looking for examples that were added"
for e in `find examples -name "*.cu"`; do
    if [ ! -e internal/build/`basename $e .cu`.mk ]; then
	echo "ADDED: `basename $e .cu`";
    fi
done

echo "Looking for examples that were deleted or moved"
for e in `find internal/build -name "*.mk"`; do
    ee=`basename $e .mk`
    case "$ee" in
	generic_example | unittester* | warningstester) continue;;
    esac
    if [  "`find examples -name $ee.cu`" == "" ]; then
	echo "DELETED: $ee";
    fi;
done
