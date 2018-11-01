/*************************************************************************/
/*  key_mapping_x11.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "key_mapping_xkb.h"

/***** SCAN CODE CONVERSION ******/

struct XKB_GODOT_Pair {

	xkb_keysym_t keysym;
	unsigned int keycode;
};

static XKB_GODOT_Pair _xkb_keysym_to_keycode[] = {

	// misc keys
	{ XKB_KEY_Escape, KEY_ESCAPE },
	{ XKB_KEY_Tab, KEY_TAB },
	{ XKB_KEY_ISO_Left_Tab, KEY_BACKTAB },
	{ XKB_KEY_BackSpace, KEY_BACKSPACE },
	{ XKB_KEY_Return, KEY_ENTER },
	{ XKB_KEY_Insert, KEY_INSERT },
	{ XKB_KEY_Delete, KEY_DELETE },
	{ XKB_KEY_Clear, KEY_DELETE },
	{ XKB_KEY_Pause, KEY_PAUSE },
	{ XKB_KEY_Print, KEY_PRINT },
	{ XKB_KEY_Home, KEY_HOME },
	{ XKB_KEY_End, KEY_END },
	{ XKB_KEY_Left, KEY_LEFT },
	{ XKB_KEY_Up, KEY_UP },
	{ XKB_KEY_Right, KEY_RIGHT },
	{ XKB_KEY_Down, KEY_DOWN },
	{ XKB_KEY_Prior, KEY_PAGEUP },
	{ XKB_KEY_Next, KEY_PAGEDOWN },
	{ XKB_KEY_Shift_L, KEY_SHIFT },
	{ XKB_KEY_Shift_R, KEY_SHIFT },
	{ XKB_KEY_Shift_Lock, KEY_SHIFT },
	{ XKB_KEY_Control_L, KEY_CONTROL },
	{ XKB_KEY_Control_R, KEY_CONTROL },
	{ XKB_KEY_Meta_L, KEY_META },
	{ XKB_KEY_Meta_R, KEY_META },
	{ XKB_KEY_Alt_L, KEY_ALT },
	{ XKB_KEY_Alt_R, KEY_ALT },
	{ XKB_KEY_Caps_Lock, KEY_CAPSLOCK },
	{ XKB_KEY_Num_Lock, KEY_NUMLOCK },
	{ XKB_KEY_Scroll_Lock, KEY_SCROLLLOCK },
	{ XKB_KEY_Super_L, KEY_SUPER_L },
	{ XKB_KEY_Super_R, KEY_SUPER_R },
	{ XKB_KEY_Menu, KEY_MENU },
	{ XKB_KEY_Hyper_L, KEY_HYPER_L },
	{ XKB_KEY_Hyper_R, KEY_HYPER_R },
	{ XKB_KEY_Help, KEY_HELP },
	{ XKB_KEY_KP_Space, KEY_SPACE },
	{ XKB_KEY_KP_Tab, KEY_TAB },
	{ XKB_KEY_KP_Enter, KEY_KP_ENTER },
	{ XKB_KEY_Home, KEY_HOME },
	{ XKB_KEY_Left, KEY_LEFT },
	{ XKB_KEY_Up, KEY_UP },
	{ XKB_KEY_Right, KEY_RIGHT },
	{ XKB_KEY_Down, KEY_DOWN },
	{ XKB_KEY_Prior, KEY_PAGEUP },
	{ XKB_KEY_Next, KEY_PAGEDOWN },
	{ XKB_KEY_End, KEY_END },
	{ XKB_KEY_Begin, KEY_CLEAR },
	{ XKB_KEY_Insert, KEY_INSERT },
	{ XKB_KEY_Delete, KEY_DELETE },
	//{ XKB_KEY_KP_Equal,                KEY_EQUAL   },
	//{ XKB_KEY_KP_Separator,            KEY_COMMA   },
	{ XKB_KEY_KP_Decimal, KEY_KP_PERIOD },
	{ XKB_KEY_KP_Delete, KEY_KP_PERIOD },
	{ XKB_KEY_KP_Multiply, KEY_KP_MULTIPLY },
	{ XKB_KEY_KP_Divide, KEY_KP_DIVIDE },
	{ XKB_KEY_KP_Subtract, KEY_KP_SUBTRACT },
	{ XKB_KEY_KP_Add, KEY_KP_ADD },
	{ XKB_KEY_KP_0, KEY_KP_0 },
	{ XKB_KEY_KP_1, KEY_KP_1 },
	{ XKB_KEY_KP_2, KEY_KP_2 },
	{ XKB_KEY_KP_3, KEY_KP_3 },
	{ XKB_KEY_KP_4, KEY_KP_4 },
	{ XKB_KEY_KP_5, KEY_KP_5 },
	{ XKB_KEY_KP_6, KEY_KP_6 },
	{ XKB_KEY_KP_7, KEY_KP_7 },
	{ XKB_KEY_KP_8, KEY_KP_8 },
	{ XKB_KEY_KP_9, KEY_KP_9 },

	// same but with numlock
	{ XKB_KEY_KP_Insert, KEY_KP_0 },
	{ XKB_KEY_KP_End, KEY_KP_1 },
	{ XKB_KEY_KP_Down, KEY_KP_2 },
	{ XKB_KEY_KP_Page_Down, KEY_KP_3 },
	{ XKB_KEY_KP_Left, KEY_KP_4 },
	{ XKB_KEY_KP_Begin, KEY_KP_5 },
	{ XKB_KEY_KP_Right, KEY_KP_6 },
	{ XKB_KEY_KP_Home, KEY_KP_7 },
	{ XKB_KEY_KP_Up, KEY_KP_8 },
	{ XKB_KEY_KP_Page_Up, KEY_KP_9 },
	{ XKB_KEY_F1, KEY_F1 },
	{ XKB_KEY_F2, KEY_F2 },
	{ XKB_KEY_F3, KEY_F3 },
	{ XKB_KEY_F4, KEY_F4 },
	{ XKB_KEY_F5, KEY_F5 },
	{ XKB_KEY_F6, KEY_F6 },
	{ XKB_KEY_F7, KEY_F7 },
	{ XKB_KEY_F8, KEY_F8 },
	{ XKB_KEY_F9, KEY_F9 },
	{ XKB_KEY_F10, KEY_F10 },
	{ XKB_KEY_F11, KEY_F11 },
	{ XKB_KEY_F12, KEY_F12 },
	{ XKB_KEY_F13, KEY_F13 },
	{ XKB_KEY_F14, KEY_F14 },
	{ XKB_KEY_F15, KEY_F15 },
	{ XKB_KEY_F16, KEY_F16 },

	// media keys
	{ XKB_KEY_XF86Back, KEY_BACK },
	{ XKB_KEY_XF86Forward, KEY_FORWARD },
	{ XKB_KEY_XF86Stop, KEY_STOP },
	{ XKB_KEY_XF86Refresh, KEY_REFRESH },
	{ XKB_KEY_XF86Favorites, KEY_FAVORITES },
	{ XKB_KEY_XF86AudioMedia, KEY_LAUNCHMEDIA },
	{ XKB_KEY_XF86OpenURL, KEY_OPENURL },
	{ XKB_KEY_XF86HomePage, KEY_HOMEPAGE },
	{ XKB_KEY_XF86Search, KEY_SEARCH },
	{ XKB_KEY_XF86AudioLowerVolume, KEY_VOLUMEDOWN },
	{ XKB_KEY_XF86AudioMute, KEY_VOLUMEMUTE },
	{ XKB_KEY_XF86AudioRaiseVolume, KEY_VOLUMEUP },
	{ XKB_KEY_XF86AudioPlay, KEY_MEDIAPLAY },
	{ XKB_KEY_XF86AudioStop, KEY_MEDIASTOP },
	{ XKB_KEY_XF86AudioPrev, KEY_MEDIAPREVIOUS },
	{ XKB_KEY_XF86AudioNext, KEY_MEDIANEXT },
	{ XKB_KEY_XF86AudioRecord, KEY_MEDIARECORD },

	// launch keys
	{ XKB_KEY_XF86Mail, KEY_LAUNCHMAIL },
	{ XKB_KEY_XF86MyComputer, KEY_LAUNCH0 },
	{ XKB_KEY_XF86Calculator, KEY_LAUNCH1 },
	{ XKB_KEY_XF86Standby, KEY_STANDBY },

	{ XKB_KEY_XF86Launch0, KEY_LAUNCH2 },
	{ XKB_KEY_XF86Launch1, KEY_LAUNCH3 },
	{ XKB_KEY_XF86Launch2, KEY_LAUNCH4 },
	{ XKB_KEY_XF86Launch3, KEY_LAUNCH5 },
	{ XKB_KEY_XF86Launch4, KEY_LAUNCH6 },
	{ XKB_KEY_XF86Launch5, KEY_LAUNCH7 },
	{ XKB_KEY_XF86Launch6, KEY_LAUNCH8 },
	{ XKB_KEY_XF86Launch7, KEY_LAUNCH9 },
	{ XKB_KEY_XF86Launch8, KEY_LAUNCHA },
	{ XKB_KEY_XF86Launch9, KEY_LAUNCHB },
	{ XKB_KEY_XF86LaunchA, KEY_LAUNCHC },
	{ XKB_KEY_XF86LaunchB, KEY_LAUNCHD },
	{ XKB_KEY_XF86LaunchC, KEY_LAUNCHE },
	{ XKB_KEY_XF86LaunchD, KEY_LAUNCHF },

	{ 0, 0 }
};

unsigned int KeyMappingXKB::get_keycode(xkb_keysym_t p_keysym) {

	// kinda bruteforce.. could optimize.

	if (p_keysym < 0x100) // Latin 1, maps 1-1
		return p_keysym;

	// look for special key
	for (int idx = 0; _xkb_keysym_to_keycode[idx].keysym != 0; idx++) {

		if (_xkb_keysym_to_keycode[idx].keysym == p_keysym)
			return _xkb_keysym_to_keycode[idx].keycode;
	}

	return KEY_UNKNOWN;
}
