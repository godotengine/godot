/**************************************************************************/
/*  keyboard.cpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "keyboard.h"

#include "core/os/os.h"

struct _KeyCodeText {
	Key code;
	const char *text;
};

static const _KeyCodeText _keycodes[] = {
	/* clang-format off */
	{Key::ESCAPE                ,"Escape"},
	{Key::TAB                   ,"Tab"},
	{Key::BACKTAB               ,"Backtab"},
	{Key::BACKSPACE             ,"Backspace"},
	{Key::ENTER                 ,"Enter"},
	{Key::KP_ENTER              ,"Kp Enter"},
	{Key::INSERT                ,"Insert"},
	{Key::KEY_DELETE            ,"Delete"},
	{Key::PAUSE                 ,"Pause"},
	{Key::PRINT                 ,"Print"},
	{Key::SYSREQ                ,"SysReq"},
	{Key::CLEAR                 ,"Clear"},
	{Key::HOME                  ,"Home"},
	{Key::END                   ,"End"},
	{Key::LEFT                  ,"Left"},
	{Key::UP                    ,"Up"},
	{Key::RIGHT                 ,"Right"},
	{Key::DOWN                  ,"Down"},
	{Key::PAGEUP                ,"PageUp"},
	{Key::PAGEDOWN              ,"PageDown"},
	{Key::SHIFT                 ,"Shift"},
	{Key::CTRL                  ,"Ctrl"},
#if defined(MACOS_ENABLED)
	{Key::META                  ,"Command"},
	{Key::CMD_OR_CTRL           ,"Command"},
	{Key::ALT                   ,"Option"},
#elif defined(WINDOWS_ENABLED)
	{Key::META                  ,"Windows"},
	{Key::CMD_OR_CTRL           ,"Ctrl"},
	{Key::ALT                   ,"Alt"},
#else
	{Key::META                  ,"Meta"},
	{Key::CMD_OR_CTRL           ,"Ctrl"},
	{Key::ALT                   ,"Alt"},
#endif
	{Key::CAPSLOCK              ,"CapsLock"},
	{Key::NUMLOCK               ,"NumLock"},
	{Key::SCROLLLOCK            ,"ScrollLock"},
	{Key::F1                    ,"F1"},
	{Key::F2                    ,"F2"},
	{Key::F3                    ,"F3"},
	{Key::F4                    ,"F4"},
	{Key::F5                    ,"F5"},
	{Key::F6                    ,"F6"},
	{Key::F7                    ,"F7"},
	{Key::F8                    ,"F8"},
	{Key::F9                    ,"F9"},
	{Key::F10                   ,"F10"},
	{Key::F11                   ,"F11"},
	{Key::F12                   ,"F12"},
	{Key::F13                   ,"F13"},
	{Key::F14                   ,"F14"},
	{Key::F15                   ,"F15"},
	{Key::F16                   ,"F16"},
	{Key::F17                   ,"F17"},
	{Key::F18                   ,"F18"},
	{Key::F19                   ,"F19"},
	{Key::F20                   ,"F20"},
	{Key::F21                   ,"F21"},
	{Key::F22                   ,"F22"},
	{Key::F23                   ,"F23"},
	{Key::F24                   ,"F24"},
	{Key::F25                   ,"F25"},
	{Key::F26                   ,"F26"},
	{Key::F27                   ,"F27"},
	{Key::F28                   ,"F28"},
	{Key::F29                   ,"F29"},
	{Key::F30                   ,"F30"},
	{Key::F31                   ,"F31"},
	{Key::F32                   ,"F32"},
	{Key::F33                   ,"F33"},
	{Key::F34                   ,"F34"},
	{Key::F35                   ,"F35"},
	{Key::KP_MULTIPLY           ,"Kp Multiply"},
	{Key::KP_DIVIDE             ,"Kp Divide"},
	{Key::KP_SUBTRACT           ,"Kp Subtract"},
	{Key::KP_PERIOD             ,"Kp Period"},
	{Key::KP_ADD                ,"Kp Add"},
	{Key::KP_0                  ,"Kp 0"},
	{Key::KP_1                  ,"Kp 1"},
	{Key::KP_2                  ,"Kp 2"},
	{Key::KP_3                  ,"Kp 3"},
	{Key::KP_4                  ,"Kp 4"},
	{Key::KP_5                  ,"Kp 5"},
	{Key::KP_6                  ,"Kp 6"},
	{Key::KP_7                  ,"Kp 7"},
	{Key::KP_8                  ,"Kp 8"},
	{Key::KP_9                  ,"Kp 9"},
	{Key::MENU                  ,"Menu"},
	{Key::HYPER                 ,"Hyper"},
	{Key::HELP                  ,"Help"},
	{Key::BACK                  ,"Back"},
	{Key::FORWARD               ,"Forward"},
	{Key::STOP                  ,"Stop"},
	{Key::REFRESH               ,"Refresh"},
	{Key::VOLUMEDOWN            ,"VolumeDown"},
	{Key::VOLUMEMUTE            ,"VolumeMute"},
	{Key::VOLUMEUP              ,"VolumeUp"},
	{Key::MEDIAPLAY             ,"MediaPlay"},
	{Key::MEDIASTOP             ,"MediaStop"},
	{Key::MEDIAPREVIOUS         ,"MediaPrevious"},
	{Key::MEDIANEXT             ,"MediaNext"},
	{Key::MEDIARECORD           ,"MediaRecord"},
	{Key::HOMEPAGE              ,"HomePage"},
	{Key::FAVORITES             ,"Favorites"},
	{Key::SEARCH                ,"Search"},
	{Key::STANDBY               ,"StandBy"},
	{Key::OPENURL               ,"OpenURL"},
	{Key::LAUNCHMAIL            ,"LaunchMail"},
	{Key::LAUNCHMEDIA           ,"LaunchMedia"},
	{Key::LAUNCH0               ,"Launch0"},
	{Key::LAUNCH1               ,"Launch1"},
	{Key::LAUNCH2               ,"Launch2"},
	{Key::LAUNCH3               ,"Launch3"},
	{Key::LAUNCH4               ,"Launch4"},
	{Key::LAUNCH5               ,"Launch5"},
	{Key::LAUNCH6               ,"Launch6"},
	{Key::LAUNCH7               ,"Launch7"},
	{Key::LAUNCH8               ,"Launch8"},
	{Key::LAUNCH9               ,"Launch9"},
	{Key::LAUNCHA               ,"LaunchA"},
	{Key::LAUNCHB               ,"LaunchB"},
	{Key::LAUNCHC               ,"LaunchC"},
	{Key::LAUNCHD               ,"LaunchD"},
	{Key::LAUNCHE               ,"LaunchE"},
	{Key::LAUNCHF               ,"LaunchF"},
	{Key::GLOBE                 ,"Globe"},
	{Key::KEYBOARD              ,"On-screen keyboard"},
	{Key::JIS_EISU              ,"JIS Eisu"},
	{Key::JIS_KANA              ,"JIS Kana"},
	{Key::UNKNOWN               ,"Unknown"},
	{Key::SPACE                 ,"Space"},
	{Key::EXCLAM                ,"Exclam"},
	{Key::QUOTEDBL              ,"QuoteDbl"},
	{Key::NUMBERSIGN            ,"NumberSign"},
	{Key::DOLLAR                ,"Dollar"},
	{Key::PERCENT               ,"Percent"},
	{Key::AMPERSAND             ,"Ampersand"},
	{Key::APOSTROPHE            ,"Apostrophe"},
	{Key::PARENLEFT             ,"ParenLeft"},
	{Key::PARENRIGHT            ,"ParenRight"},
	{Key::ASTERISK              ,"Asterisk"},
	{Key::PLUS                  ,"Plus"},
	{Key::COMMA                 ,"Comma"},
	{Key::MINUS                 ,"Minus"},
	{Key::PERIOD                ,"Period"},
	{Key::SLASH                 ,"Slash"},
	{Key::KEY_0                 ,"0"},
	{Key::KEY_1                 ,"1"},
	{Key::KEY_2                 ,"2"},
	{Key::KEY_3                 ,"3"},
	{Key::KEY_4                 ,"4"},
	{Key::KEY_5                 ,"5"},
	{Key::KEY_6                 ,"6"},
	{Key::KEY_7                 ,"7"},
	{Key::KEY_8                 ,"8"},
	{Key::KEY_9                 ,"9"},
	{Key::COLON                 ,"Colon"},
	{Key::SEMICOLON             ,"Semicolon"},
	{Key::LESS                  ,"Less"},
	{Key::EQUAL                 ,"Equal"},
	{Key::GREATER               ,"Greater"},
	{Key::QUESTION              ,"Question"},
	{Key::AT                    ,"At"},
	{Key::A                     ,"A"},
	{Key::B                     ,"B"},
	{Key::C                     ,"C"},
	{Key::D                     ,"D"},
	{Key::E                     ,"E"},
	{Key::F                     ,"F"},
	{Key::G                     ,"G"},
	{Key::H                     ,"H"},
	{Key::I                     ,"I"},
	{Key::J                     ,"J"},
	{Key::K                     ,"K"},
	{Key::L                     ,"L"},
	{Key::M                     ,"M"},
	{Key::N                     ,"N"},
	{Key::O                     ,"O"},
	{Key::P                     ,"P"},
	{Key::Q                     ,"Q"},
	{Key::R                     ,"R"},
	{Key::S                     ,"S"},
	{Key::T                     ,"T"},
	{Key::U                     ,"U"},
	{Key::V                     ,"V"},
	{Key::W                     ,"W"},
	{Key::X                     ,"X"},
	{Key::Y                     ,"Y"},
	{Key::Z                     ,"Z"},
	{Key::BRACKETLEFT           ,"BracketLeft"},
	{Key::BACKSLASH             ,"BackSlash"},
	{Key::BRACKETRIGHT          ,"BracketRight"},
	{Key::ASCIICIRCUM           ,"AsciiCircum"},
	{Key::UNDERSCORE            ,"UnderScore"},
	{Key::QUOTELEFT             ,"QuoteLeft"},
	{Key::BRACELEFT             ,"BraceLeft"},
	{Key::BAR                   ,"Bar"},
	{Key::BRACERIGHT            ,"BraceRight"},
	{Key::ASCIITILDE            ,"AsciiTilde"},
	{Key::YEN                   ,"Yen"},
	{Key::SECTION               ,"Section"},
	{Key::NONE                  ,nullptr}
	/* clang-format on */
};

bool keycode_has_unicode(Key p_keycode) {
	switch (p_keycode) {
		case Key::ESCAPE:
		case Key::TAB:
		case Key::BACKTAB:
		case Key::BACKSPACE:
		case Key::ENTER:
		case Key::KP_ENTER:
		case Key::INSERT:
		case Key::KEY_DELETE:
		case Key::PAUSE:
		case Key::PRINT:
		case Key::SYSREQ:
		case Key::CLEAR:
		case Key::HOME:
		case Key::END:
		case Key::LEFT:
		case Key::UP:
		case Key::RIGHT:
		case Key::DOWN:
		case Key::PAGEUP:
		case Key::PAGEDOWN:
		case Key::SHIFT:
		case Key::CTRL:
		case Key::META:
		case Key::ALT:
		case Key::CAPSLOCK:
		case Key::NUMLOCK:
		case Key::SCROLLLOCK:
		case Key::F1:
		case Key::F2:
		case Key::F3:
		case Key::F4:
		case Key::F5:
		case Key::F6:
		case Key::F7:
		case Key::F8:
		case Key::F9:
		case Key::F10:
		case Key::F11:
		case Key::F12:
		case Key::F13:
		case Key::F14:
		case Key::F15:
		case Key::F16:
		case Key::F17:
		case Key::F18:
		case Key::F19:
		case Key::F20:
		case Key::F21:
		case Key::F22:
		case Key::F23:
		case Key::F24:
		case Key::F25:
		case Key::F26:
		case Key::F27:
		case Key::F28:
		case Key::F29:
		case Key::F30:
		case Key::F31:
		case Key::F32:
		case Key::F33:
		case Key::F34:
		case Key::F35:
		case Key::MENU:
		case Key::HYPER:
		case Key::HELP:
		case Key::BACK:
		case Key::FORWARD:
		case Key::STOP:
		case Key::REFRESH:
		case Key::VOLUMEDOWN:
		case Key::VOLUMEMUTE:
		case Key::VOLUMEUP:
		case Key::MEDIAPLAY:
		case Key::MEDIASTOP:
		case Key::MEDIAPREVIOUS:
		case Key::MEDIANEXT:
		case Key::MEDIARECORD:
		case Key::HOMEPAGE:
		case Key::FAVORITES:
		case Key::SEARCH:
		case Key::STANDBY:
		case Key::OPENURL:
		case Key::LAUNCHMAIL:
		case Key::LAUNCHMEDIA:
		case Key::LAUNCH0:
		case Key::LAUNCH1:
		case Key::LAUNCH2:
		case Key::LAUNCH3:
		case Key::LAUNCH4:
		case Key::LAUNCH5:
		case Key::LAUNCH6:
		case Key::LAUNCH7:
		case Key::LAUNCH8:
		case Key::LAUNCH9:
		case Key::LAUNCHA:
		case Key::LAUNCHB:
		case Key::LAUNCHC:
		case Key::LAUNCHD:
		case Key::LAUNCHE:
		case Key::LAUNCHF:
		case Key::GLOBE:
		case Key::KEYBOARD:
		case Key::JIS_EISU:
		case Key::JIS_KANA:
			return false;
		default: {
		}
	}

	return true;
}

String keycode_get_string(Key p_code) {
	String codestr;
	if ((p_code & KeyModifierMask::SHIFT) != Key::NONE) {
		codestr += find_keycode_name(Key::SHIFT);
		codestr += "+";
	}
	if ((p_code & KeyModifierMask::ALT) != Key::NONE) {
		codestr += find_keycode_name(Key::ALT);
		codestr += "+";
	}
	if ((p_code & KeyModifierMask::CMD_OR_CTRL) != Key::NONE) {
		if (OS::get_singleton()->has_feature("macos") || OS::get_singleton()->has_feature("web_macos") || OS::get_singleton()->has_feature("web_ios")) {
			codestr += find_keycode_name(Key::META);
		} else {
			codestr += find_keycode_name(Key::CTRL);
		}
		codestr += "+";
	}
	if ((p_code & KeyModifierMask::CTRL) != Key::NONE) {
		codestr += find_keycode_name(Key::CTRL);
		codestr += "+";
	}
	if ((p_code & KeyModifierMask::META) != Key::NONE) {
		codestr += find_keycode_name(Key::META);
		codestr += "+";
	}

	p_code &= KeyModifierMask::CODE_MASK;

	const _KeyCodeText *kct = &_keycodes[0];

	while (kct->text) {
		if (kct->code == p_code) {
			codestr += kct->text;
			return codestr;
		}
		kct++;
	}

	codestr += String::chr((char32_t)p_code);

	return codestr;
}

Key find_keycode(const String &p_codestr) {
	Key keycode = Key::NONE;
	Vector<String> code_parts = p_codestr.split("+");
	if (code_parts.size() < 1) {
		return keycode;
	}

	const String &last_part = code_parts[code_parts.size() - 1];
	const _KeyCodeText *kct = &_keycodes[0];

	while (kct->text) {
		if (last_part.nocasecmp_to(kct->text) == 0) {
			keycode = kct->code;
			break;
		}
		kct++;
	}

	for (int part = 0; part < code_parts.size() - 1; part++) {
		const String &code_part = code_parts[part];
		if (code_part.nocasecmp_to(find_keycode_name(Key::SHIFT)) == 0) {
			keycode |= KeyModifierMask::SHIFT;
		} else if (code_part.nocasecmp_to(find_keycode_name(Key::CTRL)) == 0) {
			keycode |= KeyModifierMask::CTRL;
		} else if (code_part.nocasecmp_to(find_keycode_name(Key::META)) == 0) {
			keycode |= KeyModifierMask::META;
		} else if (code_part.nocasecmp_to(find_keycode_name(Key::ALT)) == 0) {
			keycode |= KeyModifierMask::ALT;
		}
	}

	return keycode;
}

const char *find_keycode_name(Key p_keycode) {
	const _KeyCodeText *kct = &_keycodes[0];

	while (kct->text) {
		if (kct->code == p_keycode) {
			return kct->text;
		}
		kct++;
	}

	return "";
}

int keycode_get_count() {
	const _KeyCodeText *kct = &_keycodes[0];

	int count = 0;
	while (kct->text) {
		count++;
		kct++;
	}
	return count;
}

int keycode_get_value_by_index(int p_index) {
	return (int)_keycodes[p_index].code;
}

const char *keycode_get_name_by_index(int p_index) {
	return _keycodes[p_index].text;
}

char32_t fix_unicode(char32_t p_char) {
	if (p_char >= 0x20 && p_char != 0x7F) {
		return p_char;
	}
	return 0;
}

Key fix_keycode(char32_t p_char, Key p_key) {
	if (p_char >= 0x20 && p_char <= 0x7E) {
		return (Key)String::char_uppercase(p_char);
	}
	return p_key;
}

Key fix_key_label(char32_t p_char, Key p_key) {
	if (p_char >= 0x20 && p_char != 0x7F) {
		return (Key)String::char_uppercase(p_char);
	}
	return p_key;
}
