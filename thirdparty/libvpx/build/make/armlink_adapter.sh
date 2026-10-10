#!/bin/sh
##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


verbose=0
set -- $*
for i; do
    if [ "$i" = "-o" ]; then
        on_of=1
    elif [ "$i" = "-v" ]; then
        verbose=1
    elif [ "$i" = "-g" ]; then
        args="${args} --debug"
    elif [ "$on_of" = "1" ]; then
        outfile=$i
        on_of=0
    elif [ -f "$i" ]; then
        infiles="$infiles $i"
    elif [ "${i#-l}" != "$i" ]; then
        libs="$libs ${i#-l}"
    elif [ "${i#-L}" != "$i" ]; then
        libpaths="${libpaths} ${i#-L}"
    else
        args="${args} ${i}"
    fi
    shift
done

# Absolutize library file names
for f in $libs; do
    found=0
    for d in $libpaths; do
        [ -f "$d/$f" ] && infiles="$infiles $d/$f" && found=1 && break
        [ -f "$d/lib${f}.so" ] && infiles="$infiles $d/lib${f}.so" && found=1 && break
        [ -f "$d/lib${f}.a" ] && infiles="$infiles $d/lib${f}.a" && found=1 && break
    done
    [ $found -eq 0 ] && infiles="$infiles $f"
done
for d in $libpaths; do
    [ -n "$libsearchpath" ] && libsearchpath="${libsearchpath},"
    libsearchpath="${libsearchpath}$d"
done

cmd="armlink $args --userlibpath=$libsearchpath --output=$outfile $infiles"
[ $verbose -eq 1 ] && echo $cmd
$cmd
