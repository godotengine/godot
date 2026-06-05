/**************************************************************************/
/*  gc_keyboard_handler.mm                                               */
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

#import "gc_keyboard_handler.h"

#include "core/os/keyboard.h"
#import "drivers/apple_embedded/display_server_apple_embedded.h"
#import "drivers/apple_embedded/key_mapping_apple_embedded.h"

#import <GameController/GameController.h>

// Derive a basic US-layout unshifted character from a GCKeyCode.
// Used for key label and unicode generation when UIPress is not available.
// GCKeyCode constants are extern CFIndex globals, so an if/else chain is required (not switch).
static char32_t unshifted_char_from_keycode(GCKeyCode p_keycode) {
	// Letter and digit blocks are guaranteed contiguous by the HID usage table.
	if (p_keycode >= GCKeyCodeKeyA && p_keycode <= GCKeyCodeKeyZ) {
		return 'a' + (p_keycode - GCKeyCodeKeyA);
	}
	if (p_keycode >= GCKeyCodeOne && p_keycode <= GCKeyCodeNine) {
		return '1' + (p_keycode - GCKeyCodeOne);
	}
	if (p_keycode == GCKeyCodeZero) {
		return '0';
	}
	if (p_keycode == GCKeyCodeReturnOrEnter) {
		return '\r';
	}
	if (p_keycode == GCKeyCodeTab) {
		return '\t';
	}
	if (p_keycode == GCKeyCodeSpacebar) {
		return ' ';
	}
	if (p_keycode == GCKeyCodeHyphen) {
		return '-';
	}
	if (p_keycode == GCKeyCodeEqualSign) {
		return '=';
	}
	if (p_keycode == GCKeyCodeOpenBracket) {
		return '[';
	}
	if (p_keycode == GCKeyCodeCloseBracket) {
		return ']';
	}
	if (p_keycode == GCKeyCodeBackslash) {
		return '\\';
	}
	if (p_keycode == GCKeyCodeSemicolon) {
		return ';';
	}
	if (p_keycode == GCKeyCodeQuote) {
		return '\'';
	}
	if (p_keycode == GCKeyCodeGraveAccentAndTilde) {
		return '`';
	}
	if (p_keycode == GCKeyCodeComma) {
		return ',';
	}
	if (p_keycode == GCKeyCodePeriod) {
		return '.';
	}
	if (p_keycode == GCKeyCodeSlash) {
		return '/';
	}
	return 0;
}

// Derive a shifted character for US layout.
static char32_t shifted_char_from_keycode(GCKeyCode p_keycode) {
	if (p_keycode >= GCKeyCodeKeyA && p_keycode <= GCKeyCodeKeyZ) {
		return 'A' + (p_keycode - GCKeyCodeKeyA);
	}
	if (p_keycode == GCKeyCodeOne) {
		return '!';
	}
	if (p_keycode == GCKeyCodeTwo) {
		return '@';
	}
	if (p_keycode == GCKeyCodeThree) {
		return '#';
	}
	if (p_keycode == GCKeyCodeFour) {
		return '$';
	}
	if (p_keycode == GCKeyCodeFive) {
		return '%';
	}
	if (p_keycode == GCKeyCodeSix) {
		return '^';
	}
	if (p_keycode == GCKeyCodeSeven) {
		return '&';
	}
	if (p_keycode == GCKeyCodeEight) {
		return '*';
	}
	if (p_keycode == GCKeyCodeNine) {
		return '(';
	}
	if (p_keycode == GCKeyCodeZero) {
		return ')';
	}
	if (p_keycode == GCKeyCodeHyphen) {
		return '_';
	}
	if (p_keycode == GCKeyCodeEqualSign) {
		return '+';
	}
	if (p_keycode == GCKeyCodeOpenBracket) {
		return '{';
	}
	if (p_keycode == GCKeyCodeCloseBracket) {
		return '}';
	}
	if (p_keycode == GCKeyCodeBackslash) {
		return '|';
	}
	if (p_keycode == GCKeyCodeSemicolon) {
		return ':';
	}
	if (p_keycode == GCKeyCodeQuote) {
		return '"';
	}
	if (p_keycode == GCKeyCodeGraveAccentAndTilde) {
		return '~';
	}
	if (p_keycode == GCKeyCodeComma) {
		return '<';
	}
	if (p_keycode == GCKeyCodePeriod) {
		return '>';
	}
	if (p_keycode == GCKeyCodeSlash) {
		return '?';
	}
	return unshifted_char_from_keycode(p_keycode);
}

GCKeyboardHandler::GCKeyboardHandler() {
}

GCKeyboardHandler::~GCKeyboardHandler() {
	stop();
}

void GCKeyboardHandler::setup_keyboard_handler() {
	GCKeyboard *keyboard = [GCKeyboard coalescedKeyboard];
	if (!keyboard || !keyboard.keyboardInput) {
		return;
	}

	keyboard.keyboardInput.keyChangedHandler = ^(GCKeyboardInput *kbInput, GCDeviceButtonInput *key, GCKeyCode keyCode, BOOL pressed) {
		DisplayServerAppleEmbedded *ds = DisplayServerAppleEmbedded::get_singleton();
		if (!ds) {
			return;
		}

		// Skip if the virtual keyboard is active (soft keyboard for text input).
		if (ds->is_keyboard_active()) {
			return;
		}

		Key godot_key = KeyMappingAppleEmbedded::remap_key(keyCode);
		if (godot_key == Key::NONE) {
			return;
		}

		KeyLocation location = KeyMappingAppleEmbedded::key_location(keyCode);

		// Build modifier flags compatible with UIKeyModifier constants.
		NSInteger modifier_flags = 0;
		if ([kbInput buttonForKeyCode:GCKeyCodeLeftShift].pressed || [kbInput buttonForKeyCode:GCKeyCodeRightShift].pressed) {
			modifier_flags |= UIKeyModifierShift;
		}
		if ([kbInput buttonForKeyCode:GCKeyCodeLeftControl].pressed || [kbInput buttonForKeyCode:GCKeyCodeRightControl].pressed) {
			modifier_flags |= UIKeyModifierControl;
		}
		if ([kbInput buttonForKeyCode:GCKeyCodeLeftAlt].pressed || [kbInput buttonForKeyCode:GCKeyCodeRightAlt].pressed) {
			modifier_flags |= UIKeyModifierAlternate;
		}
		if ([kbInput buttonForKeyCode:GCKeyCodeLeftGUI].pressed || [kbInput buttonForKeyCode:GCKeyCodeRightGUI].pressed) {
			modifier_flags |= UIKeyModifierCommand;
		}

		// Derive character from key code (US layout approximation).
		char32_t us = unshifted_char_from_keycode(keyCode);
		char32_t typed_char = 0;
		if (pressed) {
			if (modifier_flags & UIKeyModifierShift) {
				typed_char = shifted_char_from_keycode(keyCode);
			} else {
				typed_char = us;
			}
		}

		ds->key(fix_keycode(us, godot_key), typed_char, fix_key_label(us, godot_key), godot_key, modifier_flags, pressed, location);
	};
}

void GCKeyboardHandler::remove_keyboard_handler() {
	GCKeyboard *keyboard = [GCKeyboard coalescedKeyboard];
	if (keyboard && keyboard.keyboardInput) {
		keyboard.keyboardInput.keyChangedHandler = nil;
	}
}

void GCKeyboardHandler::start() {
	if (active) {
		return;
	}
	active = true;

	// Set up handler for already-connected keyboard.
	setup_keyboard_handler();

	// Observe keyboard connect/disconnect.
	keyboard_connect_observer = [[NSNotificationCenter defaultCenter]
			addObserverForName:GCKeyboardDidConnectNotification
						object:nil
						 queue:nil
					usingBlock:^(NSNotification *note) {
						setup_keyboard_handler();
					}];

	keyboard_disconnect_observer = [[NSNotificationCenter defaultCenter]
			addObserverForName:GCKeyboardDidDisconnectNotification
						object:nil
						 queue:nil
					usingBlock:^(NSNotification *note){
							// Keyboard disconnected; handler is automatically invalidated.
					}];
}

void GCKeyboardHandler::stop() {
	if (!active) {
		return;
	}
	active = false;

	remove_keyboard_handler();

	if (keyboard_connect_observer) {
		[[NSNotificationCenter defaultCenter] removeObserver:keyboard_connect_observer];
		keyboard_connect_observer = nil;
	}
	if (keyboard_disconnect_observer) {
		[[NSNotificationCenter defaultCenter] removeObserver:keyboard_disconnect_observer];
		keyboard_disconnect_observer = nil;
	}
}
