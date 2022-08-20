/*************************************************************************/
/*  editor_debugger_plugin.cpp                                           */
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

#include "editor_debugger_plugin.h"

#include "editor/debugger/script_editor_debugger.h"

void EditorDebuggerPlugin::_breaked(bool p_really_did, bool p_can_debug, String p_message, bool p_has_stackdump) {
	if (p_really_did) {
		emit_signal(SNAME("breaked"), p_can_debug);
	} else {
		emit_signal(SNAME("continued"));
	}
}

void EditorDebuggerPlugin::_started() {
	emit_signal(SNAME("started"));
}

void EditorDebuggerPlugin::_stopped() {
	emit_signal(SNAME("stopped"));
}

void EditorDebuggerPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("send_message", "message", "data"), &EditorDebuggerPlugin::send_message);
	ClassDB::bind_method(D_METHOD("register_message_capture", "name", "callable"), &EditorDebuggerPlugin::register_message_capture);
	ClassDB::bind_method(D_METHOD("unregister_message_capture", "name"), &EditorDebuggerPlugin::unregister_message_capture);
	ClassDB::bind_method(D_METHOD("has_capture", "name"), &EditorDebuggerPlugin::has_capture);
	ClassDB::bind_method(D_METHOD("is_breaked"), &EditorDebuggerPlugin::is_breaked);
	ClassDB::bind_method(D_METHOD("is_debuggable"), &EditorDebuggerPlugin::is_debuggable);
	ClassDB::bind_method(D_METHOD("is_session_active"), &EditorDebuggerPlugin::is_session_active);

	ADD_SIGNAL(MethodInfo("started"));
	ADD_SIGNAL(MethodInfo("stopped"));
	ADD_SIGNAL(MethodInfo("breaked", PropertyInfo(Variant::BOOL, "can_debug")));
	ADD_SIGNAL(MethodInfo("continued"));
}

void EditorDebuggerPlugin::attach_debugger(ScriptEditorDebugger *p_debugger) {
	debugger = p_debugger;
	if (debugger) {
		debugger->connect("started", callable_mp(this, &EditorDebuggerPlugin::_started));
		debugger->connect("stopped", callable_mp(this, &EditorDebuggerPlugin::_stopped));
		debugger->connect("breaked", callable_mp(this, &EditorDebuggerPlugin::_breaked));
	}
}

void EditorDebuggerPlugin::detach_debugger(bool p_call_debugger) {
	if (debugger) {
		debugger->disconnect("started", callable_mp(this, &EditorDebuggerPlugin::_started));
		debugger->disconnect("stopped", callable_mp(this, &EditorDebuggerPlugin::_stopped));
		debugger->disconnect("breaked", callable_mp(this, &EditorDebuggerPlugin::_breaked));
		if (p_call_debugger && get_script_instance()) {
			debugger->remove_debugger_plugin(get_script_instance()->get_script());
		}
		debugger = nullptr;
	}
}

void EditorDebuggerPlugin::send_message(const String &p_message, const Array &p_args) {
	ERR_FAIL_COND_MSG(!debugger, "Plugin is not attached to debugger");
	debugger->send_message(p_message, p_args);
}

void EditorDebuggerPlugin::register_message_capture(const StringName &p_name, const Callable &p_callable) {
	ERR_FAIL_COND_MSG(!debugger, "Plugin is not attached to debugger");
	debugger->register_message_capture(p_name, p_callable);
}

void EditorDebuggerPlugin::unregister_message_capture(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!debugger, "Plugin is not attached to debugger");
	debugger->unregister_message_capture(p_name);
}

bool EditorDebuggerPlugin::has_capture(const StringName &p_name) {
	ERR_FAIL_COND_V_MSG(!debugger, false, "Plugin is not attached to debugger");
	return debugger->has_capture(p_name);
}

bool EditorDebuggerPlugin::is_breaked() {
	ERR_FAIL_COND_V_MSG(!debugger, false, "Plugin is not attached to debugger");
	return debugger->is_breaked();
}

bool EditorDebuggerPlugin::is_debuggable() {
	ERR_FAIL_COND_V_MSG(!debugger, false, "Plugin is not attached to debugger");
	return debugger->is_debuggable();
}

bool EditorDebuggerPlugin::is_session_active() {
	ERR_FAIL_COND_V_MSG(!debugger, false, "Plugin is not attached to debugger");
	return debugger->is_session_active();
}

EditorDebuggerPlugin::~EditorDebuggerPlugin() {
	detach_debugger(true);
}
