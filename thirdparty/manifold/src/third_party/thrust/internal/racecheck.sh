#!/bin/sh
MEMCHECK=/work/nightly/memcheck/bin/x86_64_Linux_release/cuda-memcheck 

#########################

files=`ls thrust.test.*`;
files=`ls thrust.example.*`;

#########################

nfiles=0
for fn in $files; do
  nfiles=$((nfiles + 1))
done
j=1
for fn in $files; do
  echo " ----------------------------------------------------------------------"
  echo "  *** MEMCHECK *** [$j/$nfiles] $fn"
  echo " ----------------------------------------------------------------------"
  $MEMCHECK --tool memcheck ./$fn --verbose
  echo " ----------------------------------------------------------------------"
  echo "  *** RACECHECK *** [$j/$nfiles] $fn"
  echo " ----------------------------------------------------------------------"
  $MEMCHECK --tool racecheck ./$fn --verbose --sizes=small
  j=$((j+1))
done;
