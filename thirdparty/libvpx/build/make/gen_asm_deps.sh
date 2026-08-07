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


self=$0
show_help() {
    echo "usage: $self [options] <srcfile>"
    echo
    echo "Generate Makefile dependency information from assembly code source"
    echo
    exit 1
}
die_unknown(){
    echo "Unknown option \"$1\"."
    echo "See $0 --help for available options."
    exit 1
}
for opt do
    optval="${opt#*=}"
    case "$opt" in
    --build-pfx=*) pfx="${optval}"
    ;;
    --depfile=*) out="${optval}"
    ;;
    -I*) raw_inc_paths="${raw_inc_paths} ${opt}"
         inc_path="${inc_path} ${opt#-I}"
    ;;
    -h|--help) show_help
    ;;
    *) [ -f "$opt" ] && srcfile="$opt"
    ;;
    esac
done

[ -n "$srcfile" ] || show_help
sfx=${sfx:-asm}
includes=$(LC_ALL=C grep -E -i "include +\"?[a-z0-9_/]+\.${sfx}" $srcfile |
           perl -p -e "s;.*?([a-z0-9_/]+.${sfx}).*;\1;")
#" restore editor state
for inc in ${includes}; do
    found_inc_path=
    for idir in ${inc_path}; do
        [ -f "${idir}/${inc}" ] && found_inc_path="${idir}" && break
    done
    if [ -f `dirname $srcfile`/$inc ]; then
        # Handle include files in the same directory as the source
        $self --build-pfx=$pfx --depfile=$out ${raw_inc_paths} `dirname $srcfile`/$inc
    elif [ -n "${found_inc_path}" ]; then
        # Handle include files on the include path
        $self --build-pfx=$pfx --depfile=$out ${raw_inc_paths} "${found_inc_path}/$inc"
    else
        # Handle generated includes in the build root (which may not exist yet)
        echo ${out} ${out%d}o: "${pfx}${inc}"
    fi
done
echo ${out} ${out%d}o: $srcfile
