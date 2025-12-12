/**************************************************************************/
/*  gdscript_language_server.cpp                                          */
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

#include "gdscript_language_server.h"

#include "core/os/os.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/settings/editor_settings.h"

int GDScriptLanguageServer::port_override = -1;
bool GDScriptLanguageServer::use_stdio = false;

GDScriptLanguageServer::GDScriptLanguageServer() {
	// TODO: Move to editor_settings.cpp
	_EDITOR_DEF("network/language_server/remote_host", host);
	_EDITOR_DEF("network/language_server/remote_port", port);
	_EDITOR_DEF("network/language_server/enable_smart_resolve", true);
	_EDITOR_DEF("network/language_server/show_native_symbols_in_editor", false);
	_EDITOR_DEF("network/language_server/use_thread", use_thread);
	_EDITOR_DEF("network/language_server/poll_limit_usec", poll_limit_usec);

	set_process_internal(true);
}

void GDScriptLanguageServer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_EXIT_TREE: {
			stop();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (!start_attempted && EditorNode::get_singleton()->is_editor_ready()) {
				start_attempted = true;
				start();
			}

			if (started && !use_thread) {
				protocol.poll(poll_limit_usec);
			}
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (!EditorSettings::get_singleton()->check_changed_settings_in_group("network/language_server")) {
				break;
			}

			String remote_host = String(_EDITOR_GET("network/language_server/remote_host"));
			int remote_port = (GDScriptLanguageServer::port_override > -1) ? GDScriptLanguageServer::port_override : (int)_EDITOR_GET("network/language_server/remote_port");
			bool remote_use_thread = (bool)_EDITOR_GET("network/language_server/use_thread");
			int remote_poll_limit = (int)_EDITOR_GET("network/language_server/poll_limit_usec");
			if (remote_host != host || remote_port != port || remote_use_thread != use_thread || remote_poll_limit != poll_limit_usec) {
				stop();
				start();
			}
		} break;
	}
}

void GDScriptLanguageServer::thread_main(void *p_userdata) {
	set_current_thread_safe_for_nodes(true);
	GDScriptLanguageServer *self = static_cast<GDScriptLanguageServer *>(p_userdata);
	while (self->thread_running) {
		// Poll 20 times per second
		self->protocol.poll(self->poll_limit_usec);
		OS::get_singleton()->delay_usec(50000);
	}
}

void GDScriptLanguageServer::start() {
	use_thread = (bool)_EDITOR_GET("network/language_server/use_thread");
	poll_limit_usec = (int)_EDITOR_GET("network/language_server/poll_limit_usec");

	Error err;
	if (GDScriptLanguageServer::use_stdio) {
		err = protocol.start_stdio();
		if (err == OK) {
			OS::get_singleton()->print("--- GDScript language server started in stdio mode ---\n");
		}
	} else {
		host = String(_EDITOR_GET("network/language_server/remote_host"));
		port = (GDScriptLanguageServer::port_override > -1) ? GDScriptLanguageServer::port_override : (int)_EDITOR_GET("network/language_server/remote_port");
		err = protocol.start(port, IPAddress(host));
		if (err == OK) {
			EditorNode::get_log()->add_message("--- GDScript language server started on port " + itos(port) + " ---", EditorLog::MSG_TYPE_EDITOR);
		}
	}

	if (err == OK) {
		if (use_thread) {
			thread_running = true;
			thread.start(GDScriptLanguageServer::thread_main, this);
		}
		set_process_internal(!use_thread);
		started = true;
	}
}

void GDScriptLanguageServer::stop() {
	if (use_thread) {
		ERR_FAIL_COND(!thread.is_started());
		thread_running = false;
		thread.wait_to_finish();
	}
	protocol.stop();
	started = false;
	if (GDScriptLanguageServer::use_stdio) {
		OS::get_singleton()->print("--- GDScript language server stopped ---\n");
	} else {
		EditorNode::get_log()->add_message("--- GDScript language server stopped ---", EditorLog::MSG_TYPE_EDITOR);
	}
}

void register_lsp_types() {
	GDREGISTER_CLASS(GDScriptLanguageProtocol);
	GDREGISTER_CLASS(GDScriptTextDocument);
	GDREGISTER_CLASS(GDScriptWorkspace);
}
