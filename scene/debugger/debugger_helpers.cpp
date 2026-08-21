/**************************************************************************/
/*  debugger_helpers.cpp                                                  */
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

#ifdef DEBUG_ENABLED
#include "debugger_helpers.h"

#include "core/input/input.h"
#include "core/input/shortcut.h"

bool DebuggerHelpers::is_shortcut_pressed(const int p_idx, const HashMap<int, Ref<Shortcut>> &p_shortcuts, bool p_true_if_empty) {
	ERR_FAIL_INDEX_V(p_idx, (int)p_shortcuts.size(), p_true_if_empty);
	if (is_shortcut_empty(p_idx, p_shortcuts)) {
		return p_true_if_empty;
	}

	Ref<Shortcut> shortcut = p_shortcuts[p_idx];
	for (Ref<InputEventKey> k : shortcut->get_events()) {
		if (k.is_null()) {
			continue;
		}

		if (k->get_physical_keycode() == Key::NONE && Input::get_singleton()->is_key_pressed(k->get_keycode())) {
			return true;
		} else if (Input::get_singleton()->is_physical_key_pressed(k->get_physical_keycode())) {
			return true;
		}
	}

	return false;
}

bool DebuggerHelpers::is_shortcut_empty(const int p_idx, const HashMap<int, Ref<Shortcut>> &p_shortcuts) {
	ERR_FAIL_INDEX_V(p_idx, (int)p_shortcuts.size(), true);
	Ref<Shortcut> shortcut = p_shortcuts[p_idx];
	return shortcut.is_null() || shortcut->get_events().is_empty();
}

#endif // DEBUG_ENABLED
