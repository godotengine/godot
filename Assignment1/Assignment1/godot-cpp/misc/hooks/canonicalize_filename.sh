#!/bin/sh

# Provide the canonicalize filename (physical filename with out any symlinks)
# like the GNU version readlink with the -f option regardless of the version of
# readlink (GNU or BSD).

# This file is part of a set of unofficial pre-commit hooks available
# at github.
# Link:    https://github.com/githubbrowser/Pre-commit-hooks
# Contact: David Martin, david.martin.mailbox@googlemail.com

###########################################################
# There should be no need to change anything below this line.

# Canonicalize by recursively following every symlink in every component of the
# specified filename.  This should reproduce the results of the GNU version of
# readlink with the -f option.
#
# Reference: http://stackoverflow.com/questions/1055671/how-can-i-get-the-behavior-of-gnus-readlink-f-on-a-mac
canonicalize_filename () {
    local target_file="$1"
    local physical_directory=""
    local result=""

    # Need to restore the working directory after work.
    local working_dir="`pwd`"

    cd -- "$(dirname -- "$target_file")"
    target_file="$(basename -- "$target_file")"

    # Iterate down a (possible) chain of symlinks
    while [ -L "$target_file" ]
    do
        target_file="$(readlink -- "$target_file")"
        cd -- "$(dirname -- "$target_file")"
        target_file="$(basename -- "$target_file")"
    done

    # Compute the canonicalized name by finding the physical path
    # for the directory we're in and appending the target file.
    physical_directory="`pwd -P`"
    result="$physical_directory/$target_file"

    # restore the working directory after work.
    cd -- "$working_dir"

    echo "$result"
}
