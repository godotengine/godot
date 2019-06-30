/*************************************************************************/
/*  gdscript_language_protocol.h                                         */
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

#ifndef GDSCRIPT_PROTOCAL_SERVER_H
#define GDSCRIPT_PROTOCAL_SERVER_H

#include "gdscript_text_document.h"
#include "gdscript_workspace.h"
#include "lsp.hpp"
#include "modules/jsonrpc/jsonrpc.h"
#include "modules/websocket/websocket_peer.h"
#include "modules/websocket/websocket_server.h"

class GDScriptLanguageProtocol : public JSONRPC {
	GDCLASS(GDScriptLanguageProtocol, JSONRPC)

	enum LSPErrorCode {
		RequestCancelled = -32800,
		ContentModified = -32801,
	};

	static GDScriptLanguageProtocol *singleton;

	HashMap<int, Ref<WebSocketPeer> > clients;
	WebSocketServer *server;
	int lastest_client_id;

	Ref<GDScriptTextDocument> text_document;
	Ref<GDScriptWorkspace> workspace;

	void on_data_received(int p_id);
	void on_client_connected(int p_id, const String &p_protocal);
	void on_client_disconnected(int p_id, bool p_was_clean_close);

	String process_message(const String &p_text);
	String format_output(const String &p_text);

	bool _initialized;

protected:
	static void _bind_methods();

	Dictionary initialize(const Dictionary &p_params);
	void initialized(const Variant &p_params);

public:
	_FORCE_INLINE_ static GDScriptLanguageProtocol *get_singleton() { return singleton; }
	_FORCE_INLINE_ Ref<GDScriptWorkspace> get_workspace() { return workspace; }
	_FORCE_INLINE_ Ref<GDScriptTextDocument> get_text_document() { return text_document; }
	_FORCE_INLINE_ bool is_initialized() const { return _initialized; }

	void poll();
	Error start(int p_port);
	void stop();

	void notify_all_clients(const String &p_method, const Variant &p_params = Variant());
	void notify_client(const String &p_method, const Variant &p_params = Variant(), int p_client = -1);

	bool is_smart_resolve_enabled() const;
	bool is_goto_native_symbols_enabled() const;

	GDScriptLanguageProtocol();
	~GDScriptLanguageProtocol();
};

#endif
