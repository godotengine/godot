/*************************************************************************/
/*  script_debugger_remote.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SCRIPT_DEBUGGER_REMOTE_H
#define SCRIPT_DEBUGGER_REMOTE_H


#include "script_language.h"
#include "io/stream_peer_tcp.h"
#include "io/packet_peer.h"
#include "list.h"

class ScriptDebuggerRemote : public ScriptDebugger {

	struct Message {

		String message;
		Array data;
	};



	Ref<StreamPeerTCP> tcp_client;
	Ref<PacketPeerStream> packet_peer_stream;

	uint64_t last_perf_time;
	Object *performance;
	bool requested_quit;
	Mutex *mutex;

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

	List<String> output_strings;
	List<Message> messages;
	List<OutputError> errors;

	int max_cps;
	int char_count;
	uint64_t last_msec;
	uint64_t msec_count;

	bool locking; //hack to avoid a deadloop
	static void _print_handler(void *p_this,const String& p_string);

	PrintHandlerList phl;

	void _get_output();
	void _poll_events();
	uint32_t poll_every;


	bool _parse_live_edit(const Array &p_command);

	RequestSceneTreeMessageFunc request_scene_tree;
	void *request_scene_tree_ud;

	void _send_video_memory();
	LiveEditFuncs *live_edit_funcs;

	ErrorHandlerList eh;
	static void _err_handler(void*,const char*,const char*,int p_line,const char *, const char *,ErrorHandlerType p_type);


public:

	struct ResourceUsage {

		String path;
		String format;
		String type;
		RID id;
		int vram;
		bool operator<(const ResourceUsage& p_img) const { return vram==p_img.vram ? id<p_img.id : vram > p_img.vram; }
	};

	typedef void (*ResourceUsageFunc)(List<ResourceUsage>*);

	static ResourceUsageFunc resource_usage_func;

	Error connect_to_host(const String& p_host,uint16_t p_port);
	virtual void debug(ScriptLanguage *p_script,bool p_can_continue=true);
	virtual void idle_poll();
	virtual void line_poll();

	virtual bool is_remote() const { return true; }
	virtual void request_quit();

	virtual void send_message(const String& p_message, const Array& p_args);

	virtual void set_request_scene_tree_message_func(RequestSceneTreeMessageFunc p_func, void *p_udata);
	virtual void set_live_edit_funcs(LiveEditFuncs *p_funcs);

	ScriptDebuggerRemote();
	~ScriptDebuggerRemote();
};

#endif // SCRIPT_DEBUGGER_REMOTE_H
