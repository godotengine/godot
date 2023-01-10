/**************************************************************************/
/*  main_loop.cpp                                                         */
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

#include "main_loop.h"

#include "core/script_language.h"

void MainLoop::_bind_methods() {
	ClassDB::bind_method(D_METHOD("input_event", "event"), &MainLoop::input_event);
	ClassDB::bind_method(D_METHOD("input_text", "text"), &MainLoop::input_text);
	ClassDB::bind_method(D_METHOD("init"), &MainLoop::init);
	ClassDB::bind_method(D_METHOD("iteration", "delta"), &MainLoop::iteration);
	ClassDB::bind_method(D_METHOD("idle", "delta"), &MainLoop::idle);
	ClassDB::bind_method(D_METHOD("finish"), &MainLoop::finish);

	BIND_VMETHOD(MethodInfo("_input_event", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	BIND_VMETHOD(MethodInfo("_input_text", PropertyInfo(Variant::STRING, "text")));
	BIND_VMETHOD(MethodInfo("_initialize"));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_iteration", PropertyInfo(Variant::REAL, "delta")));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_idle", PropertyInfo(Variant::REAL, "delta")));
	BIND_VMETHOD(MethodInfo("_drop_files", PropertyInfo(Variant::POOL_STRING_ARRAY, "files"), PropertyInfo(Variant::INT, "from_screen")));
	BIND_VMETHOD(MethodInfo("_finalize"));

	BIND_VMETHOD(MethodInfo("_global_menu_action", PropertyInfo(Variant::NIL, "id"), PropertyInfo(Variant::NIL, "meta")));

	BIND_CONSTANT(NOTIFICATION_WM_MOUSE_ENTER);
	BIND_CONSTANT(NOTIFICATION_WM_MOUSE_EXIT);
	BIND_CONSTANT(NOTIFICATION_WM_FOCUS_IN);
	BIND_CONSTANT(NOTIFICATION_WM_FOCUS_OUT);
	BIND_CONSTANT(NOTIFICATION_WM_QUIT_REQUEST);
	BIND_CONSTANT(NOTIFICATION_WM_GO_BACK_REQUEST);
	BIND_CONSTANT(NOTIFICATION_WM_UNFOCUS_REQUEST);
	BIND_CONSTANT(NOTIFICATION_OS_MEMORY_WARNING);
	BIND_CONSTANT(NOTIFICATION_TRANSLATION_CHANGED);
	BIND_CONSTANT(NOTIFICATION_WM_ABOUT);
	BIND_CONSTANT(NOTIFICATION_CRASH);
	BIND_CONSTANT(NOTIFICATION_OS_IME_UPDATE);
	BIND_CONSTANT(NOTIFICATION_APP_RESUMED);
	BIND_CONSTANT(NOTIFICATION_APP_PAUSED);

	ADD_SIGNAL(MethodInfo("on_request_permissions_result", PropertyInfo(Variant::STRING, "permission"), PropertyInfo(Variant::BOOL, "granted")));
};

void MainLoop::set_init_script(const Ref<Script> &p_init_script) {
	init_script = p_init_script;
}

MainLoop::MainLoop() {
}

MainLoop::~MainLoop() {
}

void MainLoop::input_text(const String &p_text) {
	if (get_script_instance()) {
		get_script_instance()->call("_input_text", p_text);
	}
}

void MainLoop::input_event(const Ref<InputEvent> &p_event) {
	if (get_script_instance()) {
		get_script_instance()->call("_input_event", p_event);
	}
}

void MainLoop::init() {
	if (init_script.is_valid()) {
		set_script(init_script.get_ref_ptr());
	}

	if (get_script_instance()) {
		get_script_instance()->call("_initialize");
	}
}
bool MainLoop::iteration(float p_time) {
	if (get_script_instance()) {
		return get_script_instance()->call("_iteration", p_time);
	}

	return false;
}
bool MainLoop::idle(float p_time) {
	if (get_script_instance()) {
		return get_script_instance()->call("_idle", p_time);
	}

	return false;
}

void MainLoop::drop_files(const Vector<String> &p_files, int p_from_screen) {
	if (get_script_instance()) {
		get_script_instance()->call("_drop_files", p_files, p_from_screen);
	}
}

void MainLoop::global_menu_action(const Variant &p_id, const Variant &p_meta) {
	if (get_script_instance()) {
		get_script_instance()->call("_global_menu_action", p_id, p_meta);
	}
}

void MainLoop::finish() {
	if (get_script_instance()) {
		get_script_instance()->call("_finalize");
		set_script(RefPtr()); //clear script
	}
}
