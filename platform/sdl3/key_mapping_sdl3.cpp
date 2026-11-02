/**************************************************************************/
/*  key_mapping_sdl3.cpp                                                  */
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

#include "key_mapping_sdl3.h"

#include "core/templates/hash_map.h"

#include <SDL3/SDL_scancode.h>

// SDL scancodes are USB HID usage codes, so this table is the same physical-key
// mapping the Apple embedded port uses.
static HashMap<int, Key> keymap;
static HashMap<int, KeyLocation> location;

void KeyMappingSDL3::initialize() {
	if (!keymap.is_empty()) {
		return;
	}

	keymap[SDL_SCANCODE_A] = Key::A;
	keymap[SDL_SCANCODE_B] = Key::B;
	keymap[SDL_SCANCODE_C] = Key::C;
	keymap[SDL_SCANCODE_D] = Key::D;
	keymap[SDL_SCANCODE_E] = Key::E;
	keymap[SDL_SCANCODE_F] = Key::F;
	keymap[SDL_SCANCODE_G] = Key::G;
	keymap[SDL_SCANCODE_H] = Key::H;
	keymap[SDL_SCANCODE_I] = Key::I;
	keymap[SDL_SCANCODE_J] = Key::J;
	keymap[SDL_SCANCODE_K] = Key::K;
	keymap[SDL_SCANCODE_L] = Key::L;
	keymap[SDL_SCANCODE_M] = Key::M;
	keymap[SDL_SCANCODE_N] = Key::N;
	keymap[SDL_SCANCODE_O] = Key::O;
	keymap[SDL_SCANCODE_P] = Key::P;
	keymap[SDL_SCANCODE_Q] = Key::Q;
	keymap[SDL_SCANCODE_R] = Key::R;
	keymap[SDL_SCANCODE_S] = Key::S;
	keymap[SDL_SCANCODE_T] = Key::T;
	keymap[SDL_SCANCODE_U] = Key::U;
	keymap[SDL_SCANCODE_V] = Key::V;
	keymap[SDL_SCANCODE_W] = Key::W;
	keymap[SDL_SCANCODE_X] = Key::X;
	keymap[SDL_SCANCODE_Y] = Key::Y;
	keymap[SDL_SCANCODE_Z] = Key::Z;
	keymap[SDL_SCANCODE_0] = Key::KEY_0;
	keymap[SDL_SCANCODE_1] = Key::KEY_1;
	keymap[SDL_SCANCODE_2] = Key::KEY_2;
	keymap[SDL_SCANCODE_3] = Key::KEY_3;
	keymap[SDL_SCANCODE_4] = Key::KEY_4;
	keymap[SDL_SCANCODE_5] = Key::KEY_5;
	keymap[SDL_SCANCODE_6] = Key::KEY_6;
	keymap[SDL_SCANCODE_7] = Key::KEY_7;
	keymap[SDL_SCANCODE_8] = Key::KEY_8;
	keymap[SDL_SCANCODE_9] = Key::KEY_9;
	keymap[SDL_SCANCODE_BACKSLASH] = Key::BACKSLASH;
	keymap[SDL_SCANCODE_RIGHTBRACKET] = Key::BRACKETRIGHT;
	keymap[SDL_SCANCODE_COMMA] = Key::COMMA;
	keymap[SDL_SCANCODE_EQUALS] = Key::EQUAL;
	keymap[SDL_SCANCODE_MINUS] = Key::MINUS;
	keymap[SDL_SCANCODE_NONUSBACKSLASH] = Key::BAR;
	keymap[SDL_SCANCODE_LEFTBRACKET] = Key::BRACKETLEFT;
	keymap[SDL_SCANCODE_PERIOD] = Key::PERIOD;
	keymap[SDL_SCANCODE_APOSTROPHE] = Key::QUOTEDBL;
	keymap[SDL_SCANCODE_SEMICOLON] = Key::SEMICOLON;
	keymap[SDL_SCANCODE_SEPARATOR] = Key::SECTION;
	keymap[SDL_SCANCODE_SLASH] = Key::SLASH;
	keymap[SDL_SCANCODE_SPACE] = Key::SPACE;
	keymap[SDL_SCANCODE_CAPSLOCK] = Key::CAPSLOCK;
	keymap[SDL_SCANCODE_LALT] = Key::ALT;
	keymap[SDL_SCANCODE_LCTRL] = Key::CTRL;
	keymap[SDL_SCANCODE_LSHIFT] = Key::SHIFT;
	keymap[SDL_SCANCODE_RALT] = Key::ALT;
	keymap[SDL_SCANCODE_RCTRL] = Key::CTRL;
	keymap[SDL_SCANCODE_RSHIFT] = Key::SHIFT;
	keymap[SDL_SCANCODE_SCROLLLOCK] = Key::SCROLLLOCK;
	keymap[SDL_SCANCODE_LEFT] = Key::LEFT;
	keymap[SDL_SCANCODE_RIGHT] = Key::RIGHT;
	keymap[SDL_SCANCODE_UP] = Key::UP;
	keymap[SDL_SCANCODE_DOWN] = Key::DOWN;
	keymap[SDL_SCANCODE_PAGEUP] = Key::PAGEUP;
	keymap[SDL_SCANCODE_PAGEDOWN] = Key::PAGEDOWN;
	keymap[SDL_SCANCODE_HOME] = Key::HOME;
	keymap[SDL_SCANCODE_END] = Key::END;
	keymap[SDL_SCANCODE_DELETE] = Key::KEY_DELETE;
	keymap[SDL_SCANCODE_BACKSPACE] = Key::BACKSPACE;
	keymap[SDL_SCANCODE_ESCAPE] = Key::ESCAPE;
	keymap[SDL_SCANCODE_INSERT] = Key::INSERT;
	keymap[SDL_SCANCODE_RETURN2] = Key::ENTER;
	keymap[SDL_SCANCODE_TAB] = Key::TAB;
	keymap[SDL_SCANCODE_F1] = Key::F1;
	keymap[SDL_SCANCODE_F2] = Key::F2;
	keymap[SDL_SCANCODE_F3] = Key::F3;
	keymap[SDL_SCANCODE_F4] = Key::F4;
	keymap[SDL_SCANCODE_F5] = Key::F5;
	keymap[SDL_SCANCODE_F6] = Key::F6;
	keymap[SDL_SCANCODE_F7] = Key::F7;
	keymap[SDL_SCANCODE_F8] = Key::F8;
	keymap[SDL_SCANCODE_F9] = Key::F9;
	keymap[SDL_SCANCODE_F10] = Key::F10;
	keymap[SDL_SCANCODE_F11] = Key::F11;
	keymap[SDL_SCANCODE_F12] = Key::F12;
	keymap[SDL_SCANCODE_F13] = Key::F13;
	keymap[SDL_SCANCODE_F14] = Key::F14;
	keymap[SDL_SCANCODE_F15] = Key::F15;
	keymap[SDL_SCANCODE_F16] = Key::F16;
	keymap[SDL_SCANCODE_F17] = Key::F17;
	keymap[SDL_SCANCODE_F18] = Key::F18;
	keymap[SDL_SCANCODE_F19] = Key::F19;
	keymap[SDL_SCANCODE_F20] = Key::F20;
	keymap[SDL_SCANCODE_F21] = Key::F21;
	keymap[SDL_SCANCODE_F22] = Key::F22;
	keymap[SDL_SCANCODE_F23] = Key::F23;
	keymap[SDL_SCANCODE_F24] = Key::F24;
	keymap[SDL_SCANCODE_KP_0] = Key::KP_0;
	keymap[SDL_SCANCODE_KP_1] = Key::KP_1;
	keymap[SDL_SCANCODE_KP_2] = Key::KP_2;
	keymap[SDL_SCANCODE_KP_3] = Key::KP_3;
	keymap[SDL_SCANCODE_KP_4] = Key::KP_4;
	keymap[SDL_SCANCODE_KP_5] = Key::KP_5;
	keymap[SDL_SCANCODE_KP_6] = Key::KP_6;
	keymap[SDL_SCANCODE_KP_7] = Key::KP_7;
	keymap[SDL_SCANCODE_KP_8] = Key::KP_8;
	keymap[SDL_SCANCODE_KP_9] = Key::KP_9;
	keymap[SDL_SCANCODE_KP_MULTIPLY] = Key::KP_MULTIPLY;
	keymap[SDL_SCANCODE_GRAVE] = Key::QUOTELEFT;
	keymap[SDL_SCANCODE_KP_ENTER] = Key::KP_ENTER;
	keymap[SDL_SCANCODE_KP_MINUS] = Key::KP_SUBTRACT;
	keymap[SDL_SCANCODE_NUMLOCKCLEAR] = Key::NUMLOCK;
	keymap[SDL_SCANCODE_KP_PERIOD] = Key::KP_PERIOD;
	keymap[SDL_SCANCODE_KP_PLUS] = Key::KP_ADD;
	keymap[SDL_SCANCODE_KP_DIVIDE] = Key::KP_DIVIDE;
	keymap[SDL_SCANCODE_PAUSE] = Key::PAUSE;
	keymap[SDL_SCANCODE_STOP] = Key::STOP;
	keymap[SDL_SCANCODE_MUTE] = Key::VOLUMEMUTE;
	keymap[SDL_SCANCODE_VOLUMEUP] = Key::VOLUMEUP;
	keymap[SDL_SCANCODE_VOLUMEDOWN] = Key::VOLUMEDOWN;
	keymap[SDL_SCANCODE_FIND] = Key::SEARCH;
	keymap[SDL_SCANCODE_HELP] = Key::HELP;
	keymap[SDL_SCANCODE_LGUI] = Key::META;
	keymap[SDL_SCANCODE_RGUI] = Key::META;
	keymap[SDL_SCANCODE_MENU] = Key::MENU;
	keymap[SDL_SCANCODE_PRINTSCREEN] = Key::PRINT;
	keymap[SDL_SCANCODE_RETURN] = Key::ENTER;
	keymap[SDL_SCANCODE_SYSREQ] = Key::SYSREQ;
	keymap[SDL_SCANCODE_LANG1] = Key::JIS_EISU;
	keymap[SDL_SCANCODE_LANG2] = Key::JIS_KANA;
	location[SDL_SCANCODE_LALT] = KeyLocation::LEFT;
	location[SDL_SCANCODE_RALT] = KeyLocation::RIGHT;
	location[SDL_SCANCODE_LCTRL] = KeyLocation::LEFT;
	location[SDL_SCANCODE_RCTRL] = KeyLocation::RIGHT;
	location[SDL_SCANCODE_LSHIFT] = KeyLocation::LEFT;
	location[SDL_SCANCODE_RSHIFT] = KeyLocation::RIGHT;
	location[SDL_SCANCODE_LGUI] = KeyLocation::LEFT;
	location[SDL_SCANCODE_RGUI] = KeyLocation::RIGHT;
}

Key KeyMappingSDL3::remap_key(unsigned int p_scancode) {
	const Key *key = keymap.getptr((int)p_scancode);
	return key != nullptr ? *key : Key::NONE;
}

KeyLocation KeyMappingSDL3::key_location(unsigned int p_scancode) {
	const KeyLocation *loc = location.getptr((int)p_scancode);
	return loc != nullptr ? *loc : KeyLocation::UNSPECIFIED;
}
