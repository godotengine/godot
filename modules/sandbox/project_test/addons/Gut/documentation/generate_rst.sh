#!/bin/zsh
# -----------------------------------------------------------------------------
# Generate Class Ref documentation for the GUT plugin.  This generates the xml
# files through the --doctool option, then uses a python script to generate
# rst files from the xml.
#
# This should be run before each release, at the end of the release cycle.
#
# This should be run from the project root directory.  Must have opened the
# the project in the editor or run godot --headless --import.  Maybe the import
# should just be added to this script...just to be safe.  Well, if you don't
# see any xml files, run the import, that's probably the problem.
# -----------------------------------------------------------------------------

# Directory where xml files from `godot --doctool` will be placed.  These are
# used to generate rst files from.  This dir should be in your .gitignore
xmldir='documentation/class_ref_xml'

# Directory where generated rst files are placed.
rstdir='documentation/docs/class_ref'

# The directory where the docker container will place html files (for local
# testing of html generation).  This directory is cleared when this is run.
htmldir='documentation/docs/_build/html'


function printdir(){
    # echo "-- $1"
    # ls -1 $1
    # echo "-------"
}


function generate_xml(){
    the_dir=$1
    scripts_dir=$2

    mkdir -p $the_dir
    echo "Clearing $the_dir xml files"
    rm "$the_dir"/*.xml

    # The command hangs forever, always.  It looks like this will be fixed in
    # soon (fixed merged after 4.3).  So we wait 2 seconds +1 seconds (-k 1s)
    # using gtimeout (which is mac version of timeout from coreutils) and then
    # kill it (-k 1s).
    case $OSTYPE in
        "darwin"*)
            gtimeout -k 1s 2s $GODOT --doctool $the_dir --no-docbase --gdscript-docs $scripts_dir
        ;;
        *)
            timeout -k 1s 2s $GODOT --doctool $the_dir --no-docbase --gdscript-docs $scripts_dir
        ;;
    esac

    printdir $the_dir
}


function generate_rst(){
    input_dir=$1
    output_dir=$2
    flags=${3-""}

    echo "Clearing $output_dir rst files"
    rm "$output_dir"/*.rst

    python3 documentation/class_ref/godot_make_rst.py $input_dir --filter $input_dir -o $output_dir  $flags

    printdir $output_dir
}


# If you want to look at it locally, you have to do this.  This step does not
# generate any checked-in files though.
function generate_html(){
    the_dir=$1

    rm -r "$the_dir"/*
    docker-compose -f documentation/docker/compose.yml up

    # tree $the_dir
}


function main(){
    echo "--- Generating XML files ---"
    generate_xml $xmldir "res://addons/gut"

    echo "\n\n"
    echo "--- Generating RST files ---"
    generate_rst $xmldir $rstdir

    echo "\n\n"
    echo "--- Generating All HTML ---"
    generate_html $htmldir
}

main
# generate_rst $xmldir $rstdir
