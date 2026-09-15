#!/bin/bash
##
##  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

shell_name="$(uname -o 2>/dev/null)"
if [[ "$shell_name" = "Cygwin" || "$shell_name" = "Msys" ]] \
   && cygpath --help >/dev/null 2>&1; then
    FIXPATH='cygpath -m'
else
    FIXPATH='echo_path'
fi

die() {
    echo "${self_basename}: $@" >&2
    exit 1
}

die_unknown(){
    echo "Unknown option \"$1\"." >&2
    echo "See ${self_basename} --help for available options." >&2
    exit 1
}

echo_path() {
    for path; do
        echo "$path"
    done
}

# Output one, possibly changed based on the system, path per line.
fix_path() {
    $FIXPATH "$@"
}

# Corrects the paths in file_list in one pass for efficiency.
# $1 is the name of the array to be modified.
fix_file_list() {
    if [ "${FIXPATH}" = "echo_path" ] ; then
      # When used with echo_path, fix_file_list is a no-op. Avoid warning about
      # unsupported 'declare -n' when it is not important.
      return 0
    elif [ "${BASH_VERSINFO}" -lt 4 ] ; then
      echo "Cygwin path conversion has failed. Please use a version of bash"
      echo "which supports nameref (-n), introduced in bash 4.3"
      return 1
    fi
    declare -n array_ref=$1
    files=$(fix_path "${array_ref[@]}")
    local IFS=$'\n'
    array_ref=($files)
}

generate_uuid() {
    local hex="0123456789ABCDEF"
    local i
    local uuid=""
    local j
    #93995380-89BD-4b04-88EB-625FBE52EBFB
    for ((i=0; i<32; i++)); do
        (( j = $RANDOM % 16 ))
        uuid="${uuid}${hex:$j:1}"
    done
    echo "${uuid:0:8}-${uuid:8:4}-${uuid:12:4}-${uuid:16:4}-${uuid:20:12}"
}

indent1="    "
indent=""
indent_push() {
    indent="${indent}${indent1}"
}
indent_pop() {
    indent="${indent%${indent1}}"
}

tag_attributes() {
    for opt in "$@"; do
        optval="${opt#*=}"
        [ -n "${optval}" ] ||
            die "Missing attribute value in '$opt' while generating $tag tag"
        echo "${indent}${opt%%=*}=\"${optval}\""
    done
}

open_tag() {
    local tag=$1
    shift
    if [ $# -ne 0 ]; then
        echo "${indent}<${tag}"
        indent_push
        tag_attributes "$@"
        echo "${indent}>"
    else
        echo "${indent}<${tag}>"
        indent_push
    fi
}

close_tag() {
    local tag=$1
    indent_pop
    echo "${indent}</${tag}>"
}

tag() {
    local tag=$1
    shift
    if [ $# -ne 0 ]; then
        echo "${indent}<${tag}"
        indent_push
        tag_attributes "$@"
        indent_pop
        echo "${indent}/>"
    else
        echo "${indent}<${tag}/>"
    fi
}

