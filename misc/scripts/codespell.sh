#!/bin/sh
SKIP_LIST="./.*,./bin,.platform/web/node_modules,./platform/android/java/lib/src/com,./thirdparty,*.gen.*,*.po,*.pot,*.rc,package-lock.json,./core/string/locales.h,./AUTHORS.md,./COPYRIGHT.txt,./DONORS.md,./misc/dist/linux/org.godotengine.Godot.desktop,./misc/scripts/codespell.sh"
IGNORE_LIST="alo,ba,childs,complies,curvelinear,doubleclick,expct,fave,findn,gird,gud,inout,lod,nd,numer,ois,readded,ro,sav,statics,te,varius,varn,wan"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}" --builtin "clear,rare,en-GB_to_en-US"
