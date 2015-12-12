#! /bin/bash
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
godotHome=$(dirname "$here")
docTarget=${here}/html/class_list

throw() {
    echo "$@" >&2
    exit 1
}

[ -d "$docTarget" ] || mkdir -p "$docTarget" || throw "Could not create doc target $docTarget"

cd "$docTarget"
python ${here}/makehtml.py -multipage ${here}/base/classes.xml
cd "$here"

