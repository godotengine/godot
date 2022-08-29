/*************************************************************************/
/*  dom_keys.inc                                                         */
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

#include "core/os/keyboard.h"

// See https://w3c.github.io/uievents-code/#code-value-tables
Key dom_code2godot_scancode(EM_UTF8 const p_code[32], EM_UTF8 const p_key[32], bool p_physical) {
#define DOM2GODOT(p_str, p_godot_code)                                         \
	if (memcmp((const void *)p_str, (void *)p_code, strlen(p_str) + 1) == 0) { \
		return Key::p_godot_code;                                              \
	}

	// Numpad section.
	DOM2GODOT("NumLock", NUMLOCK);
	DOM2GODOT("Numpad0", KP_0);
	DOM2GODOT("Numpad1", KP_1);
	DOM2GODOT("Numpad2", KP_2);
	DOM2GODOT("Numpad3", KP_3);
	DOM2GODOT("Numpad4", KP_4);
	DOM2GODOT("Numpad5", KP_5);
	DOM2GODOT("Numpad6", KP_6);
	DOM2GODOT("Numpad7", KP_7);
	DOM2GODOT("Numpad8", KP_8);
	DOM2GODOT("Numpad9", KP_9);
	DOM2GODOT("NumpadAdd", KP_ADD);
	DOM2GODOT("NumpadBackspace", BACKSPACE);
	DOM2GODOT("NumpadClear", CLEAR);
	DOM2GODOT("NumpadClearEntry", CLEAR);
	//DOM2GODOT("NumpadComma", UNKNOWN);
	DOM2GODOT("NumpadDecimal", KP_PERIOD);
	DOM2GODOT("NumpadDivide", KP_DIVIDE);
	DOM2GODOT("NumpadEnter", KP_ENTER);
	DOM2GODOT("NumpadEqual", EQUAL);
	//DOM2GODOT("NumpadHash", UNKNOWN);
	//DOM2GODOT("NumpadMemoryAdd", UNKNOWN);
	//DOM2GODOT("NumpadMemoryClear", UNKNOWN);
	//DOM2GODOT("NumpadMemoryRecall", UNKNOWN);
	//DOM2GODOT("NumpadMemoryStore", UNKNOWN);
	//DOM2GODOT("NumpadMemorySubtract", UNKNOWN);
	DOM2GODOT("NumpadMultiply", KP_MULTIPLY);
	DOM2GODOT("NumpadParenLeft", PARENLEFT);
	DOM2GODOT("NumpadParenRight", PARENRIGHT);
	DOM2GODOT("NumpadStar", KP_MULTIPLY); // or ASTERISK ?
	DOM2GODOT("NumpadSubtract", KP_SUBTRACT);

	// Printable ASCII.
	if (!p_physical) {
		uint8_t b0 = (uint8_t)p_key[0];
		uint8_t b1 = (uint8_t)p_key[1];
		uint8_t b2 = (uint8_t)p_key[2];
		if (b1 == 0 && b0 > 0x1F && b0 < 0x7F) { // ASCII.
			if (b0 > 0x60 && b0 < 0x7B) { // Lowercase ASCII.
				b0 -= 32;
			}
			return (Key)b0;
		}

#define _U_2BYTES_MASK 0xE0
#define _U_2BYTES 0xC0
		// Latin-1 codes.
		if (b2 == 0 && (b0 & _U_2BYTES_MASK) == _U_2BYTES) { // 2-bytes utf8, only known latin.
			uint32_t key = ((b0 & ~_U_2BYTES_MASK) << 6) | (b1 & 0x3F);
			if (key >= 0xA0 && key <= 0xDF) {
				return (Key)key;
			}
			if (key >= 0xE0 && key <= 0xFF) { // Lowercase known latin.
				key -= 0x20;
				return (Key)key;
			}
		}
#undef _U_2BYTES_MASK
#undef _U_2BYTES
	}

	// Alphanumeric section.
	DOM2GODOT("Backquote", QUOTELEFT);
	DOM2GODOT("Backslash", BACKSLASH);
	DOM2GODOT("BracketLeft", BRACKETLEFT);
	DOM2GODOT("BracketRight", BRACKETRIGHT);
	DOM2GODOT("Comma", COMMA);
	DOM2GODOT("Digit0", KEY_0);
	DOM2GODOT("Digit1", KEY_1);
	DOM2GODOT("Digit2", KEY_2);
	DOM2GODOT("Digit3", KEY_3);
	DOM2GODOT("Digit4", KEY_4);
	DOM2GODOT("Digit5", KEY_5);
	DOM2GODOT("Digit6", KEY_6);
	DOM2GODOT("Digit7", KEY_7);
	DOM2GODOT("Digit8", KEY_8);
	DOM2GODOT("Digit9", KEY_9);
	DOM2GODOT("Equal", EQUAL);
	DOM2GODOT("IntlBackslash", BACKSLASH);
	//DOM2GODOT("IntlRo", UNKNOWN);
	DOM2GODOT("IntlYen", YEN);

	DOM2GODOT("KeyA", A);
	DOM2GODOT("KeyB", B);
	DOM2GODOT("KeyC", C);
	DOM2GODOT("KeyD", D);
	DOM2GODOT("KeyE", E);
	DOM2GODOT("KeyF", F);
	DOM2GODOT("KeyG", G);
	DOM2GODOT("KeyH", H);
	DOM2GODOT("KeyI", I);
	DOM2GODOT("KeyJ", J);
	DOM2GODOT("KeyK", K);
	DOM2GODOT("KeyL", L);
	DOM2GODOT("KeyM", M);
	DOM2GODOT("KeyN", N);
	DOM2GODOT("KeyO", O);
	DOM2GODOT("KeyP", P);
	DOM2GODOT("KeyQ", Q);
	DOM2GODOT("KeyR", R);
	DOM2GODOT("KeyS", S);
	DOM2GODOT("KeyT", T);
	DOM2GODOT("KeyU", U);
	DOM2GODOT("KeyV", V);
	DOM2GODOT("KeyW", W);
	DOM2GODOT("KeyX", X);
	DOM2GODOT("KeyY", Y);
	DOM2GODOT("KeyZ", Z);

	DOM2GODOT("Minus", MINUS);
	DOM2GODOT("Period", PERIOD);
	DOM2GODOT("Quote", APOSTROPHE);
	DOM2GODOT("Semicolon", SEMICOLON);
	DOM2GODOT("Slash", SLASH);

	// Functional keys in the Alphanumeric section.
	DOM2GODOT("AltLeft", ALT);
	DOM2GODOT("AltRight", ALT);
	DOM2GODOT("Backspace", BACKSPACE);
	DOM2GODOT("CapsLock", CAPSLOCK);
	DOM2GODOT("ContextMenu", MENU);
	DOM2GODOT("ControlLeft", CTRL);
	DOM2GODOT("ControlRight", CTRL);
	DOM2GODOT("Enter", ENTER);
	DOM2GODOT("MetaLeft", SUPER_L);
	DOM2GODOT("MetaRight", SUPER_R);
	DOM2GODOT("ShiftLeft", SHIFT);
	DOM2GODOT("ShiftRight", SHIFT);
	DOM2GODOT("Space", SPACE);
	DOM2GODOT("Tab", TAB);

	// ControlPad section.
	DOM2GODOT("Delete", KEY_DELETE);
	DOM2GODOT("End", END);
	DOM2GODOT("Help", HELP);
	DOM2GODOT("Home", HOME);
	DOM2GODOT("Insert", INSERT);
	DOM2GODOT("PageDown", PAGEDOWN);
	DOM2GODOT("PageUp", PAGEUP);

	// ArrowPad section.
	DOM2GODOT("ArrowDown", DOWN);
	DOM2GODOT("ArrowLeft", LEFT);
	DOM2GODOT("ArrowRight", RIGHT);
	DOM2GODOT("ArrowUp", UP);

	// Function section.
	DOM2GODOT("Escape", ESCAPE);
	DOM2GODOT("F1", F1);
	DOM2GODOT("F2", F2);
	DOM2GODOT("F3", F3);
	DOM2GODOT("F4", F4);
	DOM2GODOT("F5", F5);
	DOM2GODOT("F6", F6);
	DOM2GODOT("F7", F7);
	DOM2GODOT("F8", F8);
	DOM2GODOT("F9", F9);
	DOM2GODOT("F10", F10);
	DOM2GODOT("F11", F11);
	DOM2GODOT("F12", F12);
	//DOM2GODOT("Fn", UNKNOWN); // never actually fired, but included in the standard draft.
	//DOM2GODOT("FnLock", UNKNOWN);
	DOM2GODOT("PrintScreen", PRINT);
	DOM2GODOT("ScrollLock", SCROLLLOCK);
	DOM2GODOT("Pause", PAUSE);

	// Media keys section.
	DOM2GODOT("BrowserBack", BACK);
	DOM2GODOT("BrowserFavorites", FAVORITES);
	DOM2GODOT("BrowserForward", FORWARD);
	DOM2GODOT("BrowserHome", OPENURL);
	DOM2GODOT("BrowserRefresh", REFRESH);
	DOM2GODOT("BrowserSearch", SEARCH);
	DOM2GODOT("BrowserStop", STOP);
	//DOM2GODOT("Eject", UNKNOWN);
	DOM2GODOT("LaunchApp1", LAUNCH0);
	DOM2GODOT("LaunchApp2", LAUNCH1);
	DOM2GODOT("LaunchMail", LAUNCHMAIL);
	DOM2GODOT("MediaPlayPause", MEDIAPLAY);
	DOM2GODOT("MediaSelect", LAUNCHMEDIA);
	DOM2GODOT("MediaStop", MEDIASTOP);
	DOM2GODOT("MediaTrackNext", MEDIANEXT);
	DOM2GODOT("MediaTrackPrevious", MEDIAPREVIOUS);
	//DOM2GODOT("Power", UNKNOWN);
	//DOM2GODOT("Sleep", UNKNOWN);
	DOM2GODOT("AudioVolumeDown", VOLUMEDOWN);
	DOM2GODOT("AudioVolumeMute", VOLUMEMUTE);
	DOM2GODOT("AudioVolumeUp", VOLUMEUP);
	//DOM2GODOT("WakeUp", UNKNOWN);
	return Key::UNKNOWN;
#undef DOM2GODOT
}
