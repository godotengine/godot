/**************************************************************************/
/*  script_debugger_remote.h                                              */
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

#ifndef SCRIPT_DEBUGGER_REMOTE_H
#define SCRIPT_DEBUGGER_REMOTE_H

#include "core/list.h"
#include "core/os/os.h"
#include "core/script_language.h"
#include "debug_session_v2.h"
#include "debug_transport_v2.h"
#include "debug_variable_store_v2.h"

class SceneTree;

class ScriptDebuggerRemote : public ScriptDebugger {
	struct Message {
		String message;
		Array data;
	};

	struct ProfileInfoSort {
		bool operator()(ScriptLanguage::ProfilingInfo *A, ScriptLanguage::ProfilingInfo *B) const {
			return A->total_time < B->total_time;
		}
	};

	Vector<ScriptLanguage::ProfilingInfo> profile_info;
	Vector<ScriptLanguage::ProfilingInfo *> profile_info_ptrs;
	Vector<MultiplayerAPI::ProfilingInfo> network_profile_info;

	Map<StringName, int> profiler_function_signature_map;
	float frame_time, process_time, physics_time, physics_frame_time;

	bool profiling;
	bool profiling_network;
	int max_frame_functions;
	bool skip_profile_frame;
	bool reload_all_scripts;

	DebugTransportV2 transport;
	DebugSessionV2 session;
	DebugVariableStoreV2 variable_store;
	Ref<PacketPeerStream> packet_peer_stream;

	uint64_t last_perf_time;
	uint64_t last_net_prof_time;
	uint64_t last_net_bandwidth_time;
	Object *performance;
	bool requested_quit;
	Mutex mutex;

	struct OutputError {
		int hr;
		int min;
		int sec;
		int msec;
		String source_file;
		String source_func;
		int source_line;
		String error;
		String error_descr;
		bool warning;
		Array callstack;
	};

	struct OutputString {
		String message;
		int type;
	};

	List<OutputString> output_strings;
	List<Message> messages;
	int max_messages_per_frame;
	int n_messages_dropped;
	List<OutputError> errors;
	int max_errors_per_second;
	int max_warnings_per_second;
	int n_errors_dropped;
	int n_warnings_dropped;
	uint64_t last_transport_dropped;

	int max_cps;
	int char_count;
	int err_count;
	int warn_count;
	uint64_t last_msec;
	uint64_t msec_count;

	OS::ProcessID allow_focus_steal_pid;

	bool locking; //hack to avoid a deadloop
	static void _print_handler(void *p_this, const String &p_string, bool p_error);

	PrintHandlerList phl;

	void _get_output();
	void _poll_events();
	DebugSessionV2::Action _handle_control_message(const DebugProtocolV2::Envelope &p_message);
	void _send_control_ack(const DebugSessionV2::Decision &p_decision);
	bool _poll_data_request(ScriptLanguage *p_script);
	void _send_data_error(const DebugProtocolV2::Envelope &p_request, const String &p_code, const String &p_message);
	uint32_t poll_every;
	uint64_t next_reconnect_msec;
	uint64_t scene_revision;
	uint64_t pending_scene_request_id;
	uint64_t pending_scene_stop_id;

	SceneTree *scene_tree;

	bool _parse_live_edit(const Array &p_command);

	void _set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value);

	bool _send_object_id(ObjectID p_id, uint64_t p_request_id, uint64_t p_stop_id);
	void _send_video_memory(uint64_t p_request_id = 0, uint64_t p_stop_id = 0);

	Ref<MultiplayerAPI> multiplayer;

	ErrorHandlerList eh;
	static void _err_handler(void *, const char *, const char *, int p_line, const char *, const char *, ErrorHandlerType p_type);

	void _send_profiling_data(bool p_for_frame);
	void _send_network_profiling_data();
	void _send_network_bandwidth_usage();

	struct FrameData {
		StringName name;
		Array data;
	};

	Vector<FrameData> profile_frame_data;

	void _put_variable(const String &p_name, const Variant &p_variable);

	void _save_node(ObjectID id, const String &p_path);

	bool skip_breakpoints;

public:
	enum MessageType {
		MESSAGE_TYPE_LOG,
		MESSAGE_TYPE_ERROR,
	};

	struct ResourceUsage {
		String path;
		String format;
		String type;
		RID id;
		int vram;
		bool operator<(const ResourceUsage &p_img) const { return vram == p_img.vram ? id < p_img.id : vram > p_img.vram; }
	};

	typedef void (*ResourceUsageFunc)(List<ResourceUsage> *);

	static ResourceUsageFunc resource_usage_func;

	Error connect_to_host(const String &p_host, uint16_t p_port);
	virtual void debug(ScriptLanguage *p_script, bool p_can_continue = true, bool p_is_error_breakpoint = false);
	virtual void idle_poll();
	virtual void line_poll();

	virtual bool is_remote() const { return true; }
	virtual void request_quit();

	virtual void send_message(const String &p_message, const Array &p_args);
	virtual void send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, ErrorHandlerType p_type, const Vector<ScriptLanguage::StackInfo> &p_stack_info);

	virtual void set_multiplayer(Ref<MultiplayerAPI> p_multiplayer);

	virtual bool is_profiling() const;
	virtual void add_profiling_frame_data(const StringName &p_name, const Array &p_data);

	virtual void profiling_start();
	virtual void profiling_end();
	virtual void profiling_set_frame_times(float p_frame_time, float p_process_time, float p_physics_time, float p_physics_frame_time);

	virtual void set_skip_breakpoints(bool p_skip_breakpoints);

	void set_scene_tree(SceneTree *p_scene_tree) { scene_tree = p_scene_tree; };
	void set_allow_focus_steal_pid(OS::ProcessID p_pid);

	ScriptDebuggerRemote();
	~ScriptDebuggerRemote();
};

#endif // SCRIPT_DEBUGGER_REMOTE_H
