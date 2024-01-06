#!/bin/sh
SKIP_LIST="./external,./.git,*.desktop,*.gen.*,*.po,*.pot,*.rc"

IGNORE_LIST="curvelinear,doubleclick,expct,findn,gird,hel,inout,lod,nd,numer,ot,te,vai"

codespell -w -q 3 -S "${SKIP_LIST}" -L "${IGNORE_LIST}" --builtin "clear,rare,en-GB_to_en-US"