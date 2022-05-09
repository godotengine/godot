#!/bin/sh
SKIP_LIST="./thirdparty,*.gen.*,*.po,*.pot,package-lock.json,./core/string/locales.h,./DONORS.md,./misc/dist/linux/org.godotengine.Godot.desktop,./misc/scripts/codespell.sh"
IGNORE_LIST="ba,childs,complies,curvelinear,expct,fave,findn,gird,inout,lod,nd,numer,ois,ro,statics,te,varius,varn"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}"
