#!/bin/sh
SKIP_LIST="./.*,./bin,./thirdparty,*.desktop,*.gen.*,*.po,*.pot,*.rc,./AUTHORS.md,./COPYRIGHT.txt,./DONORS.md,"
SKIP_LIST+="./core/string/locales.h,./editor/project_converter_3_to_4.cpp,./misc/scripts/codespell.sh,"
SKIP_LIST+="./platform/android/java/lib/src/com,./platform/web/node_modules,./platform/web/package-lock.json,"

IGNORE_LIST="alo,ba,complies,curvelinear,doubleclick,expct,fave,findn,gird,gud,inout,lod,nd,numer,ois,readded,ro,sav,statics,te,varius,varn,wan"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}" --builtin "clear,rare,en-GB_to_en-US"
