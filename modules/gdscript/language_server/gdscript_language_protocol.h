/**************************************************************************/
/*  gdscript_language_protocol.h                                          */
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

#pragma once

#include "core/io/resource_loader.h"
#include "core/templates/local_vector.h"
#include "gdscript_text_document.h"
#include "gdscript_workspace.h"

#include "core/io/stream_peer_tcp.h"
#include "core/io/tcp_server.h"
#include "editor/file_system/editor_file_system.h"

#include "modules/jsonrpc/jsonrpc.h"
#include "scene/resources/packed_scene.h"

#define LSP_MAX_BUFFER_SIZE 4194304
#define LSP_MAX_CLIENTS 8

/**
 * Used to load and cache scenes for autocompletion.
 * */
class SceneCache {
private:
	friend class GDScriptLanguageProtocol;

	struct ScriptQueueForLoad {
		String current_loaded_owner = "";
		ResourceLoader::ThreadLoadStatus current_load_status = ResourceLoader::THREAD_LOAD_INVALID_RESOURCE;
		LocalVector<String> script_path_queue;
		bool is_empty() const { return script_path_queue.size() == 0; }
		bool has_script_path(String p_script_path) const { return script_path_queue.has(p_script_path); }
		bool has_current_load() const { return (not current_loaded_owner.is_empty() && (current_load_status == ResourceLoader::THREAD_LOAD_IN_PROGRESS || current_load_status == ResourceLoader::THREAD_LOAD_LOADED)); }

		void clear() {
			get_current_loaded_owner_res();
			script_path_queue.clear();
		}

		void try_owners_for_load(LocalVector<String> &owners) {
			Error r_error = Error::FAILED;
			while (r_error != Error::OK && not owners.is_empty()) {
				String owner_path = owners[0];
				owners.remove_at(0);
				r_error = ResourceLoader::load_threaded_request(owner_path);
				if (r_error == Error::OK) {
					current_loaded_owner = owner_path;
					update_load_status();
					LOG_LSP("Scene load started for:", get_front(), current_loaded_owner);
				}
			}
		}

		void update_load_status() {
			if (not current_loaded_owner.is_empty()) {
				current_load_status = ResourceLoader::load_threaded_get_status(current_loaded_owner);
			} else {
				current_load_status = ResourceLoader::THREAD_LOAD_INVALID_RESOURCE;
			}
		}

		// can also be used to clean up currently loaded owner
		// ResourceLoader has no way of dropping the load of a specific requested resource
		Ref<PackedScene> get_current_loaded_owner_res() {
			Ref<PackedScene> res;
			if (has_current_load()) {
				res = ResourceLoader::load_threaded_get(current_loaded_owner);
				current_loaded_owner = "";
				current_load_status = ResourceLoader::THREAD_LOAD_INVALID_RESOURCE;
			}
			return res;
		}

		void push_back(String p_script_path) { script_path_queue.push_back(p_script_path); }

		String get_front() { return script_path_queue.size() > 0 ? script_path_queue[0] : ""; }

		void push_front(String p_script_path) {
			if (script_path_queue.size() > 0 && script_path_queue.has(p_script_path)) {
				script_path_queue.erase(p_script_path);
			}
			script_path_queue.insert(0, p_script_path);
		}

		void remove_front() {
			if (script_path_queue.size() == 0) {
				return;
			}
			script_path_queue.remove_at(0);
			get_current_loaded_owner_res();
		}

		void erase(String p_script_path) {
			if (script_path_queue.size() == 0) {
				return;
			}
			if (p_script_path == script_path_queue[0]) {
				get_current_loaded_owner_res();
			}
			script_path_queue.erase(p_script_path);
		}
	};

	HashMap<String, Node *> cache;
	ScriptQueueForLoad pending_script_queue;

	void _get_owner_paths(EditorFileSystemDirectory *p_dir, const String &p_path, LocalVector<String> &r_owner_paths);
	void _try_scene_load_for_next_script();
	void _enqueue_script_for_scene_load(String p_path);
	void _poll_scene_load();
	void _finalize_scene_load();

public:
	Node *get(const String &p_path);
	void request_scene_load_for_script(const String &p_path);
	void unload_scene_for_script(const String &p_path);
	void clear();
};

class GDScriptLanguageProtocol : public JSONRPC {
	GDCLASS(GDScriptLanguageProtocol, JSONRPC)

private:
	struct LSPeer : RefCounted {
		Ref<StreamPeerTCP> connection;

		uint8_t req_buf[LSP_MAX_BUFFER_SIZE];
		int req_pos = 0;
		bool has_header = false;
		bool has_content = false;
		int content_length = 0;
		Vector<CharString> res_queue;
		int res_sent = 0;

		Error handle_data();
		Error send_data();
	};

	enum LSPErrorCode {
		RequestCancelled = -32800,
		ContentModified = -32801,
	};

	static GDScriptLanguageProtocol *singleton;

	HashMap<int, Ref<LSPeer>> clients;
	SceneCache scene_cache;
	Ref<TCPServer> server;
	int latest_client_id = 0;
	int next_client_id = 0;

	int next_server_id = 0;

	Ref<GDScriptTextDocument> text_document;
	Ref<GDScriptWorkspace> workspace;

	Error on_client_connected();
	void on_client_disconnected(const int &p_client_id);

	String process_message(const String &p_text);
	String format_output(const String &p_text);

	bool _initialized = false;

protected:
	static void _bind_methods();

	Dictionary initialize(const Dictionary &p_params);
	void initialized(const Variant &p_params);

public:
	_FORCE_INLINE_ static GDScriptLanguageProtocol *get_singleton() { return singleton; }
	_FORCE_INLINE_ Ref<GDScriptWorkspace> get_workspace() { return workspace; }
	_FORCE_INLINE_ Ref<GDScriptTextDocument> get_text_document() { return text_document; }
	_FORCE_INLINE_ SceneCache *get_scene_cache() { return &scene_cache; }

	_FORCE_INLINE_ bool is_initialized() const { return _initialized; }

	void poll(int p_limit_usec);
	Error start(int p_port, const IPAddress &p_bind_ip);
	void stop();

	void notify_client(const String &p_method, const Variant &p_params = Variant(), int p_client_id = -1);
	void request_client(const String &p_method, const Variant &p_params = Variant(), int p_client_id = -1);

	bool is_smart_resolve_enabled() const;
	bool is_goto_native_symbols_enabled() const;

	GDScriptLanguageProtocol();
};
