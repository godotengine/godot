/**************************************************************************/
/*  spx_input_mgr.cpp                                                     */
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

#include "spx_input_mgr.h"
#include "gdextension_spx_ext.h"
#include "scene/main/window.h"

void SpxInputMgr::on_start() {
	SpxBaseMgr::on_start();
	input_proxy = memnew(SpxInputProxy);
	input_proxy->set_name( "input_proxy");
	get_spx_root()->add_child(input_proxy);
	input_proxy->ready();
}

// input
GdVec2 SpxInputMgr::get_mouse_pos() {
	auto pos = Input::get_singleton()->get_mouse_position();
	auto size  = DisplayServer::get_singleton()->window_get_size();
	return GdVec2(pos.x - size.x*0.5, size.y*0.5- pos.y);
}

GdBool SpxInputMgr::get_mouse_state(GdInt mouse_id) {
	if (mouse_id < (int)MouseButton::LEFT && (int)MouseButton::RIGHT < mouse_id) {
		print_error("unknown mouse id " + itos(mouse_id));
	}
	return Input::get_singleton()->is_mouse_button_pressed((MouseButton)mouse_id);
}
GdBool SpxInputMgr::get_key(GdInt key) {
	return Input::get_singleton()->is_key_pressed((Key)key);
}
GdInt SpxInputMgr::get_key_state(GdInt key) {
	return get_key(key) ? 1 : 0;
}

GdFloat SpxInputMgr::get_axis(GdString neg_action, GdString pos_action) {
	return Input::get_singleton()->get_axis(SpxStr(neg_action),SpxStr(pos_action));
}

GdBool SpxInputMgr::is_action_pressed(GdString action) {
	return Input::get_singleton()->is_action_pressed(SpxStr(action));
}

GdBool SpxInputMgr::is_action_just_pressed(GdString action) {
	return Input::get_singleton()->is_action_just_pressed(SpxStr(action));
}

GdBool SpxInputMgr::is_action_just_released(GdString action) {
	return Input::get_singleton()->is_action_just_released(SpxStr(action));
}
