#!/bin/sh
SKIP_LIST="./.*,./**/.*,./bin,./thirdparty,*.desktop,*.gen.*,*.po,*.pot,*.rc,./AUTHORS.md,./COPYRIGHT.txt,./DONORS.md,"
SKIP_LIST+="./core/input/gamecontrollerdb.txt,./core/string/locales.h,./editor/renames_map_3_to_4.cpp,./misc/scripts/codespell.sh,"
SKIP_LIST+="./platform/android/java/lib/src/com,./platform/web/node_modules,./platform/web/package-lock.json,"

IGNORE_LIST="curvelinear,doubleclick,expct,findn,gird,hel,inout,lod,mis,nd,numer,ot,requestor,te,vai"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}" --builtin "clear,rare,en-GB_to_en-US"
