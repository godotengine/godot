/*************************************************************************/
/*  gdscript_language_server.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "gdscript_language_server.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "editor/editor_node.h"

GDScriptLanguageServer::GDScriptLanguageServer() {
	thread = NULL;
	thread_exit = false;
	_EDITOR_DEF("network/language_server/remote_port", 6008);
	_EDITOR_DEF("network/language_server/enable_smart_resolve", false);
	_EDITOR_DEF("network/language_server/show_native_symbols_in_editor", false);
}

void GDScriptLanguageServer::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			start();
			break;
		case NOTIFICATION_EXIT_TREE:
			stop();
			break;
	}
}

void GDScriptLanguageServer::thread_main(void *p_userdata) {
	GDScriptLanguageServer *self = static_cast<GDScriptLanguageServer *>(p_userdata);
	while (!self->thread_exit) {
		self->protocol.poll();
		OS::get_singleton()->delay_usec(10);
	}
}

void GDScriptLanguageServer::start() {
	int port = (int)_EDITOR_GET("network/language_server/remote_port");
	if (protocol.start(port) == OK) {
		EditorNode::get_log()->add_message("--- GDScript language server started ---", EditorLog::MSG_TYPE_EDITOR);
		ERR_FAIL_COND(thread != NULL || thread_exit);
		thread_exit = false;
		thread = Thread::create(GDScriptLanguageServer::thread_main, this);
	}
}

void GDScriptLanguageServer::stop() {
	ERR_FAIL_COND(NULL == thread || thread_exit);
	thread_exit = true;
	Thread::wait_to_finish(thread);
	memdelete(thread);
	thread = NULL;
	protocol.stop();
	EditorNode::get_log()->add_message("--- GDScript language server stopped ---", EditorLog::MSG_TYPE_EDITOR);
}

void register_lsp_types() {
	ClassDB::register_class<GDScriptLanguageProtocol>();
	ClassDB::register_class<GDScriptTextDocument>();
	ClassDB::register_class<GDScriptWorkspace>();
}
