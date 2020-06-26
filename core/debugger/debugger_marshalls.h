/*************************************************************************/
/*  debugger_marshalls.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DEBUGGER_MARSHARLLS_H
#define DEBUGGER_MARSHARLLS_H

#include "core/script_language.h"
#include "servers/rendering_server.h"

struct DebuggerMarshalls {
	// Memory usage
	struct ResourceInfo {
		String path;
		String format;
		String type;
		RID id;
		int vram = 0;
		bool operator<(const ResourceInfo &p_img) const { return vram == p_img.vram ? id < p_img.id : vram > p_img.vram; }
	};

	struct ResourceUsage {
		List<ResourceInfo> infos;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

	// Network profiler
	struct MultiplayerNodeInfo {
		ObjectID node;
		String node_path;
		int incoming_rpc = 0;
		int incoming_rset = 0;
		int outgoing_rpc = 0;
		int outgoing_rset = 0;
	};

	struct NetworkProfilerFrame {
		Vector<MultiplayerNodeInfo> infos;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

	// Script Profiler
	class ScriptFunctionSignature {
	public:
		StringName name;
		int id = -1;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

	struct ScriptFunctionInfo {
		StringName name;
		int sig_id = -1;
		int call_count = 0;
		float self_time = 0;
		float total_time = 0;
	};

	// Servers profiler
	struct ServerFunctionInfo {
		StringName name;
		float time = 0;
	};

	struct ServerInfo {
		StringName name;
		List<ServerFunctionInfo> functions;
	};

	struct ServersProfilerFrame {
		int frame_number = 0;
		float frame_time = 0;
		float idle_time = 0;
		float physics_time = 0;
		float physics_frame_time = 0;
		float script_time = 0;
		List<ServerInfo> servers;
		Vector<ScriptFunctionInfo> script_functions;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

	struct ScriptStackVariable {
		String name;
		Variant value;
		int type = -1;

		Array serialize(int max_size = 1 << 20); // 1 MiB default.
		bool deserialize(const Array &p_arr);
	};

	struct ScriptStackDump {
		List<ScriptLanguage::StackInfo> frames;
		ScriptStackDump() {}

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

	struct OutputError {
		int hr = -1;
		int min = -1;
		int sec = -1;
		int msec = -1;
		String source_file;
		String source_func;
		int source_line = -1;
		String error;
		String error_descr;
		bool warning = false;
		Vector<ScriptLanguage::StackInfo> callstack;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

	// Visual Profiler
	struct VisualProfilerFrame {
		uint64_t frame_number;
		Vector<RS::FrameProfileArea> areas;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};
};

#endif // DEBUGGER_MARSHARLLS_H
