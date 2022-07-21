/*************************************************************************/
/*  key_mapping_macos.mm                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "key_mapping_macos.h"

#import <Carbon/Carbon.h>
#import <Cocoa/Cocoa.h>

bool KeyMappingMacOS::is_numpad_key(unsigned int key) {
	static const unsigned int table[] = {
		0x41, /* kVK_ANSI_KeypadDecimal */
		0x43, /* kVK_ANSI_KeypadMultiply */
		0x45, /* kVK_ANSI_KeypadPlus */
		0x47, /* kVK_ANSI_KeypadClear */
		0x4b, /* kVK_ANSI_KeypadDivide */
		0x4c, /* kVK_ANSI_KeypadEnter */
		0x4e, /* kVK_ANSI_KeypadMinus */
		0x51, /* kVK_ANSI_KeypadEquals */
		0x52, /* kVK_ANSI_Keypad0 */
		0x53, /* kVK_ANSI_Keypad1 */
		0x54, /* kVK_ANSI_Keypad2 */
		0x55, /* kVK_ANSI_Keypad3 */
		0x56, /* kVK_ANSI_Keypad4 */
		0x57, /* kVK_ANSI_Keypad5 */
		0x58, /* kVK_ANSI_Keypad6 */
		0x59, /* kVK_ANSI_Keypad7 */
		0x5b, /* kVK_ANSI_Keypad8 */
		0x5c, /* kVK_ANSI_Keypad9 */
		0x5f, /* kVK_JIS_KeypadComma */
		0x00
	};
	for (int i = 0; table[i] != 0; i++) {
		if (key == table[i]) {
			return true;
		}
	}
	return false;
}

// Keyboard symbol translation table.
static const Key _macos_to_godot_table[128] = {
	/* 00 */ Key::A,
	/* 01 */ Key::S,
	/* 02 */ Key::D,
	/* 03 */ Key::F,
	/* 04 */ Key::H,
	/* 05 */ Key::G,
	/* 06 */ Key::Z,
	/* 07 */ Key::X,
	/* 08 */ Key::C,
	/* 09 */ Key::V,
	/* 0a */ Key::SECTION, /* ISO Section */
	/* 0b */ Key::B,
	/* 0c */ Key::Q,
	/* 0d */ Key::W,
	/* 0e */ Key::E,
	/* 0f */ Key::R,
	/* 10 */ Key::Y,
	/* 11 */ Key::T,
	/* 12 */ Key::KEY_1,
	/* 13 */ Key::KEY_2,
	/* 14 */ Key::KEY_3,
	/* 15 */ Key::KEY_4,
	/* 16 */ Key::KEY_6,
	/* 17 */ Key::KEY_5,
	/* 18 */ Key::EQUAL,
	/* 19 */ Key::KEY_9,
	/* 1a */ Key::KEY_7,
	/* 1b */ Key::MINUS,
	/* 1c */ Key::KEY_8,
	/* 1d */ Key::KEY_0,
	/* 1e */ Key::BRACERIGHT,
	/* 1f */ Key::O,
	/* 20 */ Key::U,
	/* 21 */ Key::BRACELEFT,
	/* 22 */ Key::I,
	/* 23 */ Key::P,
	/* 24 */ Key::ENTER,
	/* 25 */ Key::L,
	/* 26 */ Key::J,
	/* 27 */ Key::APOSTROPHE,
	/* 28 */ Key::K,
	/* 29 */ Key::SEMICOLON,
	/* 2a */ Key::BACKSLASH,
	/* 2b */ Key::COMMA,
	/* 2c */ Key::SLASH,
	/* 2d */ Key::N,
	/* 2e */ Key::M,
	/* 2f */ Key::PERIOD,
	/* 30 */ Key::TAB,
	/* 31 */ Key::SPACE,
	/* 32 */ Key::QUOTELEFT,
	/* 33 */ Key::BACKSPACE,
	/* 34 */ Key::UNKNOWN,
	/* 35 */ Key::ESCAPE,
	/* 36 */ Key::META,
	/* 37 */ Key::META,
	/* 38 */ Key::SHIFT,
	/* 39 */ Key::CAPSLOCK,
	/* 3a */ Key::ALT,
	/* 3b */ Key::CTRL,
	/* 3c */ Key::SHIFT,
	/* 3d */ Key::ALT,
	/* 3e */ Key::CTRL,
	/* 3f */ Key::UNKNOWN, /* Function */
	/* 40 */ Key::F17,
	/* 41 */ Key::KP_PERIOD,
	/* 42 */ Key::UNKNOWN,
	/* 43 */ Key::KP_MULTIPLY,
	/* 44 */ Key::UNKNOWN,
	/* 45 */ Key::KP_ADD,
	/* 46 */ Key::UNKNOWN,
	/* 47 */ Key::NUMLOCK, /* Really KeypadClear... */
	/* 48 */ Key::VOLUMEUP, /* VolumeUp */
	/* 49 */ Key::VOLUMEDOWN, /* VolumeDown */
	/* 4a */ Key::VOLUMEMUTE, /* Mute */
	/* 4b */ Key::KP_DIVIDE,
	/* 4c */ Key::KP_ENTER,
	/* 4d */ Key::UNKNOWN,
	/* 4e */ Key::KP_SUBTRACT,
	/* 4f */ Key::F18,
	/* 50 */ Key::F19,
	/* 51 */ Key::EQUAL, /* KeypadEqual */
	/* 52 */ Key::KP_0,
	/* 53 */ Key::KP_1,
	/* 54 */ Key::KP_2,
	/* 55 */ Key::KP_3,
	/* 56 */ Key::KP_4,
	/* 57 */ Key::KP_5,
	/* 58 */ Key::KP_6,
	/* 59 */ Key::KP_7,
	/* 5a */ Key::F20,
	/* 5b */ Key::KP_8,
	/* 5c */ Key::KP_9,
	/* 5d */ Key::YEN, /* JIS Yen */
	/* 5e */ Key::UNDERSCORE, /* JIS Underscore */
	/* 5f */ Key::COMMA, /* JIS KeypadComma */
	/* 60 */ Key::F5,
	/* 61 */ Key::F6,
	/* 62 */ Key::F7,
	/* 63 */ Key::F3,
	/* 64 */ Key::F8,
	/* 65 */ Key::F9,
	/* 66 */ Key::UNKNOWN, /* JIS Eisu */
	/* 67 */ Key::F11,
	/* 68 */ Key::UNKNOWN, /* JIS Kana */
	/* 69 */ Key::F13,
	/* 6a */ Key::F16,
	/* 6b */ Key::F14,
	/* 6c */ Key::UNKNOWN,
	/* 6d */ Key::F10,
	/* 6e */ Key::MENU,
	/* 6f */ Key::F12,
	/* 70 */ Key::UNKNOWN,
	/* 71 */ Key::F15,
	/* 72 */ Key::INSERT, /* Really Help... */
	/* 73 */ Key::HOME,
	/* 74 */ Key::PAGEUP,
	/* 75 */ Key::KEY_DELETE,
	/* 76 */ Key::F4,
	/* 77 */ Key::END,
	/* 78 */ Key::F2,
	/* 79 */ Key::PAGEDOWN,
	/* 7a */ Key::F1,
	/* 7b */ Key::LEFT,
	/* 7c */ Key::RIGHT,
	/* 7d */ Key::DOWN,
	/* 7e */ Key::UP,
	/* 7f */ Key::UNKNOWN,
};

// Translates a OS X keycode to a Godot keycode.
Key KeyMappingMacOS::translate_key(unsigned int key) {
	if (key >= 128) {
		return Key::UNKNOWN;
	}

	return _macos_to_godot_table[key];
}

// Translates a Godot keycode back to a macOS keycode.
unsigned int KeyMappingMacOS::unmap_key(Key key) {
	for (int i = 0; i <= 126; i++) {
		if (_macos_to_godot_table[i] == key) {
			return i;
		}
	}
	return 127;
}

struct _KeyCodeMap {
	UniChar kchar;
	Key kcode;
};

static const _KeyCodeMap _keycodes[55] = {
	{ '`', Key::QUOTELEFT },
	{ '~', Key::ASCIITILDE },
	{ '0', Key::KEY_0 },
	{ '1', Key::KEY_1 },
	{ '2', Key::KEY_2 },
	{ '3', Key::KEY_3 },
	{ '4', Key::KEY_4 },
	{ '5', Key::KEY_5 },
	{ '6', Key::KEY_6 },
	{ '7', Key::KEY_7 },
	{ '8', Key::KEY_8 },
	{ '9', Key::KEY_9 },
	{ '-', Key::MINUS },
	{ '_', Key::UNDERSCORE },
	{ '=', Key::EQUAL },
	{ '+', Key::PLUS },
	{ 'q', Key::Q },
	{ 'w', Key::W },
	{ 'e', Key::E },
	{ 'r', Key::R },
	{ 't', Key::T },
	{ 'y', Key::Y },
	{ 'u', Key::U },
	{ 'i', Key::I },
	{ 'o', Key::O },
	{ 'p', Key::P },
	{ '[', Key::BRACELEFT },
	{ ']', Key::BRACERIGHT },
	{ '{', Key::BRACELEFT },
	{ '}', Key::BRACERIGHT },
	{ 'a', Key::A },
	{ 's', Key::S },
	{ 'd', Key::D },
	{ 'f', Key::F },
	{ 'g', Key::G },
	{ 'h', Key::H },
	{ 'j', Key::J },
	{ 'k', Key::K },
	{ 'l', Key::L },
	{ ';', Key::SEMICOLON },
	{ ':', Key::COLON },
	{ '\'', Key::APOSTROPHE },
	{ '\"', Key::QUOTEDBL },
	{ '\\', Key::BACKSLASH },
	{ '#', Key::NUMBERSIGN },
	{ 'z', Key::Z },
	{ 'x', Key::X },
	{ 'c', Key::C },
	{ 'v', Key::V },
	{ 'b', Key::B },
	{ 'n', Key::N },
	{ 'm', Key::M },
	{ ',', Key::COMMA },
	{ '.', Key::PERIOD },
	{ '/', Key::SLASH }
};

// Remap key according to current keyboard layout.
Key KeyMappingMacOS::remap_key(unsigned int key, unsigned int state) {
	if (is_numpad_key(key)) {
		return translate_key(key);
	}

	TISInputSourceRef current_keyboard = TISCopyCurrentKeyboardInputSource();
	if (!current_keyboard) {
		return translate_key(key);
	}

	CFDataRef layout_data = (CFDataRef)TISGetInputSourceProperty(current_keyboard, kTISPropertyUnicodeKeyLayoutData);
	if (!layout_data) {
		return translate_key(key);
	}

	const UCKeyboardLayout *keyboard_layout = (const UCKeyboardLayout *)CFDataGetBytePtr(layout_data);

	UInt32 keys_down = 0;
	UniChar chars[4];
	UniCharCount real_length;

	OSStatus err = UCKeyTranslate(keyboard_layout,
			key,
			kUCKeyActionDisplay,
			(state >> 8) & 0xFF,
			LMGetKbdType(),
			kUCKeyTranslateNoDeadKeysBit,
			&keys_down,
			sizeof(chars) / sizeof(chars[0]),
			&real_length,
			chars);

	if (err != noErr) {
		return translate_key(key);
	}

	for (unsigned int i = 0; i < 55; i++) {
		if (_keycodes[i].kchar == chars[0]) {
			return _keycodes[i].kcode;
		}
	}
	return translate_key(key);
}

struct _KeyCodeText {
	Key code;
	char32_t text;
};

static const _KeyCodeText _native_keycodes[] = {
	/* clang-format off */
		{Key::ESCAPE                        ,0x001B},
		{Key::TAB                           ,0x0009},
		{Key::BACKTAB                       ,0x007F},
		{Key::BACKSPACE                     ,0x0008},
		{Key::ENTER                         ,0x000D},
		{Key::INSERT                        ,NSInsertFunctionKey},
		{Key::KEY_DELETE                    ,0x007F},
		{Key::PAUSE                         ,NSPauseFunctionKey},
		{Key::PRINT                         ,NSPrintScreenFunctionKey},
		{Key::SYSREQ                        ,NSSysReqFunctionKey},
		{Key::CLEAR                         ,NSClearLineFunctionKey},
		{Key::HOME                          ,0x2196},
		{Key::END                           ,0x2198},
		{Key::LEFT                          ,0x001C},
		{Key::UP                            ,0x001E},
		{Key::RIGHT                         ,0x001D},
		{Key::DOWN                          ,0x001F},
		{Key::PAGEUP                        ,0x21DE},
		{Key::PAGEDOWN                      ,0x21DF},
		{Key::NUMLOCK                       ,NSClearLineFunctionKey},
		{Key::SCROLLLOCK                    ,NSScrollLockFunctionKey},
		{Key::F1                            ,NSF1FunctionKey},
		{Key::F2                            ,NSF2FunctionKey},
		{Key::F3                            ,NSF3FunctionKey},
		{Key::F4                            ,NSF4FunctionKey},
		{Key::F5                            ,NSF5FunctionKey},
		{Key::F6                            ,NSF6FunctionKey},
		{Key::F7                            ,NSF7FunctionKey},
		{Key::F8                            ,NSF8FunctionKey},
		{Key::F9                            ,NSF9FunctionKey},
		{Key::F10                           ,NSF10FunctionKey},
		{Key::F11                           ,NSF11FunctionKey},
		{Key::F12                           ,NSF12FunctionKey},
		{Key::F13                           ,NSF13FunctionKey},
		{Key::F14                           ,NSF14FunctionKey},
		{Key::F15                           ,NSF15FunctionKey},
		{Key::F16                           ,NSF16FunctionKey},
		{Key::F17                           ,NSF17FunctionKey},
		{Key::F18                           ,NSF18FunctionKey},
		{Key::F19                           ,NSF19FunctionKey},
		{Key::F20                           ,NSF20FunctionKey},
		{Key::F21                           ,NSF21FunctionKey},
		{Key::F22                           ,NSF22FunctionKey},
		{Key::F23                           ,NSF23FunctionKey},
		{Key::F24                           ,NSF24FunctionKey},
		{Key::F25                           ,NSF25FunctionKey},
		{Key::F26                           ,NSF26FunctionKey},
		{Key::F27                           ,NSF27FunctionKey},
		{Key::F28                           ,NSF28FunctionKey},
		{Key::F29                           ,NSF29FunctionKey},
		{Key::F30                           ,NSF30FunctionKey},
		{Key::F31                           ,NSF31FunctionKey},
		{Key::F32                           ,NSF32FunctionKey},
		{Key::F33                           ,NSF33FunctionKey},
		{Key::F34                           ,NSF34FunctionKey},
		{Key::F35                           ,NSF35FunctionKey},
		{Key::MENU                          ,NSMenuFunctionKey},
		{Key::HELP                          ,NSHelpFunctionKey},
		{Key::STOP                          ,NSStopFunctionKey},
		{Key::LAUNCH0                       ,NSUserFunctionKey},
		{Key::SPACE                         ,0x0020},
		{Key::EXCLAM                        ,'!'},
		{Key::QUOTEDBL                      ,'\"'},
		{Key::NUMBERSIGN                    ,'#'},
		{Key::DOLLAR                        ,'$'},
		{Key::PERCENT                       ,'\%'},
		{Key::AMPERSAND                     ,'&'},
		{Key::APOSTROPHE                    ,'\''},
		{Key::PARENLEFT                     ,'('},
		{Key::PARENRIGHT                    ,')'},
		{Key::ASTERISK                      ,'*'},
		{Key::PLUS                          ,'+'},
		{Key::COMMA                         ,','},
		{Key::MINUS                         ,'-'},
		{Key::PERIOD                        ,'.'},
		{Key::SLASH                         ,'/'},
		{Key::KEY_0                         ,'0'},
		{Key::KEY_1                         ,'1'},
		{Key::KEY_2                         ,'2'},
		{Key::KEY_3                         ,'3'},
		{Key::KEY_4                         ,'4'},
		{Key::KEY_5                         ,'5'},
		{Key::KEY_6                         ,'6'},
		{Key::KEY_7                         ,'7'},
		{Key::KEY_8                         ,'8'},
		{Key::KEY_9                         ,'9'},
		{Key::COLON                         ,':'},
		{Key::SEMICOLON                     ,';'},
		{Key::LESS                          ,'<'},
		{Key::EQUAL                         ,'='},
		{Key::GREATER                       ,'>'},
		{Key::QUESTION                      ,'?'},
		{Key::AT                            ,'@'},
		{Key::A                             ,'a'},
		{Key::B                             ,'b'},
		{Key::C                             ,'c'},
		{Key::D                             ,'d'},
		{Key::E                             ,'e'},
		{Key::F                             ,'f'},
		{Key::G                             ,'g'},
		{Key::H                             ,'h'},
		{Key::I                             ,'i'},
		{Key::J                             ,'j'},
		{Key::K                             ,'k'},
		{Key::L                             ,'l'},
		{Key::M                             ,'m'},
		{Key::N                             ,'n'},
		{Key::O                             ,'o'},
		{Key::P                             ,'p'},
		{Key::Q                             ,'q'},
		{Key::R                             ,'r'},
		{Key::S                             ,'s'},
		{Key::T                             ,'t'},
		{Key::U                             ,'u'},
		{Key::V                             ,'v'},
		{Key::W                             ,'w'},
		{Key::X                             ,'x'},
		{Key::Y                             ,'y'},
		{Key::Z                             ,'z'},
		{Key::BRACKETLEFT                   ,'['},
		{Key::BACKSLASH                     ,'\\'},
		{Key::BRACKETRIGHT                  ,']'},
		{Key::ASCIICIRCUM                   ,'^'},
		{Key::UNDERSCORE                    ,'_'},
		{Key::QUOTELEFT                     ,'`'},
		{Key::BRACELEFT                     ,'{'},
		{Key::BAR                           ,'|'},
		{Key::BRACERIGHT                    ,'}'},
		{Key::ASCIITILDE                    ,'~'},
		{Key::NONE                          ,0x0000}
	/* clang-format on */
};

String KeyMappingMacOS::keycode_get_native_string(Key p_keycode) {
	const _KeyCodeText *kct = &_native_keycodes[0];

	while (kct->text) {
		if (kct->code == p_keycode) {
			return String::chr(kct->text);
		}
		kct++;
	}
	return String();
}

unsigned int KeyMappingMacOS::keycode_get_native_mask(Key p_keycode) {
	unsigned int mask = 0;
	if ((p_keycode & KeyModifierMask::CTRL) != Key::NONE) {
		mask |= NSEventModifierFlagControl;
	}
	if ((p_keycode & KeyModifierMask::ALT) != Key::NONE) {
		mask |= NSEventModifierFlagOption;
	}
	if ((p_keycode & KeyModifierMask::SHIFT) != Key::NONE) {
		mask |= NSEventModifierFlagShift;
	}
	if ((p_keycode & KeyModifierMask::META) != Key::NONE) {
		mask |= NSEventModifierFlagCommand;
	}
	if ((p_keycode & KeyModifierMask::KPAD) != Key::NONE) {
		mask |= NSEventModifierFlagNumericPad;
	}
	return mask;
}
