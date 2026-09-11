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



for opt in "$@"; do
    optval="${opt#*=}"
    case "$opt" in
    --bare) bare=true ;;
    *) break ;;
    esac
    shift
done
source_path=${1:-.}
out_file=${2}
id=${3:-VERSION_STRING}

git_version_id=""
if [ -e "${source_path}/.git" ]; then
    # Source Path is a git working copy. Check for local modifications.
    # Note that git submodules may have a file as .git, not a directory.
    export GIT_DIR="${source_path}/.git"
    git_version_id=`git describe --match=v[0-9]* 2>/dev/null`
fi

changelog_version=""
for p in "${source_path}" "${source_path}/.."; do
    if [ -z "$git_version_id" -a -f "${p}/CHANGELOG" ]; then
        changelog_version=`head -n1 "${p}/CHANGELOG" | awk '{print $2}'`
        changelog_version="${changelog_version}"
        break
    fi
done
version_str="${changelog_version}${git_version_id}"
bare_version=${version_str#v}
major_version=${bare_version%%.*}
bare_version=${bare_version#*.}
minor_version=${bare_version%%.*}
bare_version=${bare_version#*.}
patch_version=${bare_version%%-*}
bare_version=${bare_version#${patch_version}}
extra_version=${bare_version##-}

#since they'll be used as integers below make sure they are or force to 0
for v in major_version minor_version patch_version; do
    if eval echo \$$v |grep -E -q '[^[:digit:]]'; then
        eval $v=0
    fi
done

if [ ${bare} ]; then
    echo "${changelog_version}${git_version_id}" > $$.tmp
else
    cat<<EOF>$$.tmp
// This file is generated. Do not edit.
#ifndef VPX_VERSION_H_
#define VPX_VERSION_H_
#define VERSION_MAJOR  $major_version
#define VERSION_MINOR  $minor_version
#define VERSION_PATCH  $patch_version
#define VERSION_EXTRA  "$extra_version"
#define VERSION_PACKED ((VERSION_MAJOR<<16)|(VERSION_MINOR<<8)|(VERSION_PATCH))
#define ${id}_NOSP "${version_str}"
#define ${id}      " ${version_str}"
#endif  // VPX_VERSION_H_
EOF
fi
if [ -n "$out_file" ]; then
diff $$.tmp ${out_file} >/dev/null 2>&1 || cat $$.tmp > ${out_file}
else
cat $$.tmp
fi
rm $$.tmp
