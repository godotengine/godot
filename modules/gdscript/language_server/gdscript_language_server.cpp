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

#include "gdscript_language_protocol.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "scene/main/node.h"

int GDScriptLanguageServer::port_override = -1;

GDScriptLanguageServer::GDScriptLanguageServer() {
	set_process_internal(true);
}

void survey_folder(const String &p_dir, int &r_total, int &r_self, int &r_class) {
	Error err = OK;
	Ref<DirAccess> dir = DirAccess::open(p_dir, &err);

	if (err != OK) {
		return;
	}

	String path = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	while (!next.is_empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == "..") {
				next = dir->get_next();
				continue;
			}
			survey_folder(path.path_join(next), r_total, r_self, r_class);
		} else if (next.ends_with(".gd")) {
			Ref<GDScript> script = ResourceLoader::load(ProjectSettings::get_singleton()->localize_path(path.path_join(next)));

			HashMap<StringName, GDScriptFunction *> m_funcs = HashMap<StringName, GDScriptFunction *>(script->get_member_functions());
			m_funcs["@implicit_new"] = const_cast<GDScriptFunction *>(script->get_implicit_initializer());
			m_funcs["@implicit_ready"] = const_cast<GDScriptFunction *>(script->get_implicit_ready());
			m_funcs["@static_initializer"] = const_cast<GDScriptFunction *>(script->get_static_initializer());

			for (auto fn : script->get_member_functions()) {
				r_total += 1;
				if (fn.value->_self_used) {
					r_self += 1;
				}
				if (fn.value->_class_used) {
					r_class += 1;
				}
			}
		}
		next = dir->get_next();
	}
}

static void survey() {
	int total_fn = 0;
	int self_used = 0;
	int class_used = 0;
	survey_folder("res://", total_fn, self_used, class_used);
	print_line("total: ", total_fn, " self: ", self_used, " class: ", class_used);
}

void GDScriptLanguageServer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			EditorCommandPalette::get_singleton()->add_command("GDScript Survey: Stack Use", "gdscript/survey/stack_use", callable_mp_static(&survey));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			stop();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (!start_attempted && EditorNode::get_singleton()->is_editor_ready()) {
				start_attempted = true;
				start();
			}

			if (started && !use_thread) {
				GDScriptLanguageProtocol::get_singleton()->poll(poll_limit_usec);
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
		GDScriptLanguageProtocol::get_singleton()->poll(self->poll_limit_usec);
		OS::get_singleton()->delay_usec(50000);
	}
}

void GDScriptLanguageServer::start() {
	host = String(_EDITOR_GET("network/language_server/remote_host"));
	port = (GDScriptLanguageServer::port_override > -1) ? GDScriptLanguageServer::port_override : (int)_EDITOR_GET("network/language_server/remote_port");
	use_thread = (bool)_EDITOR_GET("network/language_server/use_thread");
	poll_limit_usec = (int)_EDITOR_GET("network/language_server/poll_limit_usec");
	const Error status = GDScriptLanguageProtocol::get_singleton()->start(port, IPAddress(host));
	if (status != OK) {
		EditorNode::get_log()->add_message("--- Failed to start GDScript language server on port " + itos(port) + ": " + error_names[status] + " ---", EditorLog::MSG_TYPE_EDITOR);
		return;
	}
	EditorNode::get_log()->add_message("--- GDScript language server started on port " + itos(port) + " ---", EditorLog::MSG_TYPE_EDITOR);
	if (use_thread) {
		thread_running = true;
		thread.start(GDScriptLanguageServer::thread_main, this);
	}
	set_process_internal(!use_thread);
	started = true;
}

void GDScriptLanguageServer::stop() {
	if (use_thread) {
		ERR_FAIL_COND(!thread.is_started());
		thread_running = false;
		thread.wait_to_finish();
	}
	GDScriptLanguageProtocol::get_singleton()->stop();
	started = false;
	EditorNode::get_log()->add_message("--- GDScript language server stopped ---", EditorLog::MSG_TYPE_EDITOR);
}

void register_lsp_types() {
	GDREGISTER_CLASS(GDScriptLanguageProtocol);
	GDREGISTER_CLASS(GDScriptTextDocument);
	GDREGISTER_CLASS(GDScriptWorkspace);
}
