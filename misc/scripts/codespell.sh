#!/bin/sh
SKIP_LIST="./.git,./bin,./thirdparty,*.gen.*,*.po,*.pot,package-lock.json,./core/string/locales.h,./DONORS.md,./misc/dist/linux/org.godotengine.Godot.desktop,./misc/scripts/codespell.sh"
IGNORE_LIST="alo,ba,childs,complies,curvelinear,doubleclick,expct,fave,findn,gird,gud,inout,lod,nd,numer,ois,readded,ro,sav,statics,te,varius,varn,wan"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}"
