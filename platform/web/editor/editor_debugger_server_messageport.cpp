/**************************************************************************/
/*  editor_debugger_server_messageport.cpp                                */
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

#include "editor_debugger_server_messageport.h"

#include "editor/editor_log.h"
#include "editor/editor_node.h"

extern "C" {
bool godot_js_editor_debugger_active();
void godot_js_editor_debugger_cb(void (*p_callback)(int p_id));
}

EditorDebuggerServerMessagePort *EditorDebuggerServerMessagePort::singleton = nullptr;

void EditorDebuggerServerMessagePort::_add_session(int p_session) {
	ERR_FAIL_NULL(singleton);
	singleton->pending.push_back(p_session);
}

void EditorDebuggerServerMessagePort::initialize() {
	EditorDebuggerServer::register_protocol_handler("messageport://", EditorDebuggerServerMessagePort::create);
}

void EditorDebuggerServerMessagePort::poll() {
}

String EditorDebuggerServerMessagePort::get_uri() const {
	return "messageport://";
}

Error EditorDebuggerServerMessagePort::start(const String &p_uri) {
	godot_js_editor_debugger_cb(&_add_session);
	return OK;
}

void EditorDebuggerServerMessagePort::stop() {
	godot_js_editor_debugger_cb(nullptr);
	pending.clear();
}

bool EditorDebuggerServerMessagePort::is_active() const {
	return godot_js_editor_debugger_active();
}

bool EditorDebuggerServerMessagePort::is_connection_available() const {
	return pending.size();
}

Ref<RemoteDebuggerPeer> EditorDebuggerServerMessagePort::take_connection() {
	ERR_FAIL_COND_V(!is_connection_available(), Ref<RemoteDebuggerPeer>());
	Ref<RemoteDebuggerPeerMessagePort> peer = memnew(RemoteDebuggerPeerMessagePort(pending.front()->get()));
	pending.pop_front();
	return peer;
}

EditorDebuggerServerMessagePort::EditorDebuggerServerMessagePort() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

EditorDebuggerServerMessagePort::~EditorDebuggerServerMessagePort() {
	stop();
	singleton = nullptr;
}

Ref<EditorDebuggerServer> EditorDebuggerServerMessagePort::create(const String &p_protocol) {
	ERR_FAIL_COND_V(p_protocol != "messageport://", nullptr);
	return memnew(EditorDebuggerServerMessagePort);
}
