/**************************************************************************/
/*  control.compat.inc                                                    */
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

bool Control::_has_focus_bind_compat_102412() const {
	return has_focus(PlayerId::P1);
}

void Control::_grab_focus_bind_compat_102412() {
	grab_focus(PlayerId::P1);
}

void Control::_release_focus_bind_compat_102412() {
	release_focus(PlayerId::P1);
}

Control *Control::_find_next_valid_focus_bind_compat_102412() const {
	return find_next_valid_focus(PlayerId::P1);
}

Control *Control::_find_prev_valid_focus_bind_compat_102412() const {
	return find_prev_valid_focus(PlayerId::P1);
}

Control *Control::_find_valid_focus_neighbor_bind_compat_102412(Side p_size) const {
	return find_valid_focus_neighbor(p_size, PlayerId::P1);
}

void Control::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("has_focus"), &Control::_has_focus_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("grab_focus"), &Control::_grab_focus_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("release_focus"), &Control::_release_focus_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("find_next_valid_focus"), &Control::_find_next_valid_focus_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("find_prev_valid_focus"), &Control::_find_prev_valid_focus_bind_compat_102412);
	ClassDB::bind_compatibility_method(D_METHOD("find_valid_focus_neighbor", "side"), &Control::_find_valid_focus_neighbor_bind_compat_102412);
}

#endif // DISABLE_DEPRECATED
