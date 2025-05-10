#!/bin/sh
#
# Check to make sure 8BitDo controller configurations are correct

echo "Expected output:"
cat <<__EOF__
    "050000003512000020ab000000780f00,8BitDo SNES30 Gamepad,a:b20,b:b21,back:b30,dpdown:+a1,dpleft:-a0,dpright:+a0,dpup:-a1,leftshoulder:b26,rightshoulder:b27,start:b31,x:b23,y:b24,hint:SDL_GAMECONTROLLER_USE_BUTTON_LABELS:=1,",
    "050000003512000020ab000000780f00,8BitDo SNES30 Gamepad,a:b21,b:b20,back:b30,dpdown:+a1,dpleft:-a0,dpright:+a0,dpup:-a1,leftshoulder:b26,rightshoulder:b27,start:b31,x:b24,y:b23,hint:!SDL_GAMECONTROLLER_USE_BUTTON_LABELS:=1,",

__EOF__

echo "Actual output:"
${FGREP:-grep -F} 8BitDo SDL_gamepad_db.h | ${FGREP:-grep -F} -v hint
${EGREP:-grep -E} "hint:SDL_GAMECONTROLLER_USE_BUTTON_LABELS:=1" SDL_gamepad_db.h  | ${FGREP:-grep -F} -i 8bit | ${FGREP:-grep -F} -v x:b2,y:b3 | ${FGREP:-grep -F} -v x:b3,y:b4
${EGREP:-grep -E} "hint:.SDL_GAMECONTROLLER_USE_BUTTON_LABELS:=1" SDL_gamepad_db.h  | ${FGREP:-grep -F} -i 8bit | ${FGREP:-grep -F} -v x:b3,y:b2 | ${FGREP:-grep -F} -v x:b4,y:b3
