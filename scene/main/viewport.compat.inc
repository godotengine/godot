/**************************************************************************/
/*  viewport.compat.inc                                                   */
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

#ifndef DISABLE_DEPRECATED

void Viewport::_gui_remove_focus_for_window_bind_compat_102412(Node *p_window) {
	_gui_remove_focus_for_window(p_window, PlayerId::P1);
}

void Viewport::_push_text_input_bind_compat_102412(const String &p_text) {
	push_text_input(p_text, PlayerId::P1);
}

void Viewport::_gui_release_focus_bind_compat_102412() {
	gui_release_focus(PlayerId::P1);
}

Control *Viewport::_gui_get_focus_owner_bind_compat_102412() const {
	return gui_get_focus_owner(PlayerId::P1);
}

void Viewport::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("_gui_remove_focus_for_window"), &Viewport::_gui_remove_focus_for_window_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("push_text_input", "text"), &Viewport::_push_text_input_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("gui_release_focus"), &Viewport::_gui_release_focus_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("gui_get_focus_owner"), &Viewport::_gui_get_focus_owner_bind_compat_102412);
}

#endif // DISABLE_DEPRECATED
