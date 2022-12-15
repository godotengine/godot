#!/bin/sh
SKIP_LIST="./.git,./bin,./thirdparty,*.gen.*,*.po,*.pot,package-lock.json,./core/string/locales.h,./DONORS.md,./misc/dist/linux/org.godotengine.Godot.desktop,./misc/scripts/codespell.sh"
IGNORE_LIST="ba,childs,commiting,complies,curvelinear,doubleclick,expct,fave,findn,gird,inout,leapyear,lod,nd,numer,ois,readded,ro,statics,switchs,te,varius,varn"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}"
