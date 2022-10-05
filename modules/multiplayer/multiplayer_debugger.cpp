/*************************************************************************/
/*  multiplayer_debugger.cpp                                             */
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

#include "multiplayer_debugger.h"

#include "core/debugger/engine_debugger.h"
#include "scene/main/node.h"

List<Ref<EngineProfiler>> multiplayer_profilers;

void MultiplayerDebugger::initialize() {
	Ref<BandwidthProfiler> bandwidth;
	bandwidth.instantiate();
	bandwidth->bind("multiplayer");
	multiplayer_profilers.push_back(bandwidth);

	Ref<RPCProfiler> rpc_profiler;
	rpc_profiler.instantiate();
	rpc_profiler->bind("rpc");
	multiplayer_profilers.push_back(rpc_profiler);
}

void MultiplayerDebugger::deinitialize() {
	multiplayer_profilers.clear();
}

// BandwidthProfiler

int MultiplayerDebugger::BandwidthProfiler::bandwidth_usage(const Vector<BandwidthFrame> &p_buffer, int p_pointer) {
	ERR_FAIL_COND_V(p_buffer.size() == 0, 0);
	int total_bandwidth = 0;

	uint64_t timestamp = OS::get_singleton()->get_ticks_msec();
	uint64_t final_timestamp = timestamp - 1000;

	int i = (p_pointer + p_buffer.size() - 1) % p_buffer.size();

	while (i != p_pointer && p_buffer[i].packet_size > 0) {
		if (p_buffer[i].timestamp < final_timestamp) {
			return total_bandwidth;
		}
		total_bandwidth += p_buffer[i].packet_size;
		i = (i + p_buffer.size() - 1) % p_buffer.size();
	}

	ERR_FAIL_COND_V_MSG(i == p_pointer, total_bandwidth, "Reached the end of the bandwidth profiler buffer, values might be inaccurate.");
	return total_bandwidth;
}

void MultiplayerDebugger::BandwidthProfiler::toggle(bool p_enable, const Array &p_opts) {
	if (!p_enable) {
		bandwidth_in.clear();
		bandwidth_out.clear();
	} else {
		bandwidth_in_ptr = 0;
		bandwidth_in.resize(16384); // ~128kB
		for (int i = 0; i < bandwidth_in.size(); ++i) {
			bandwidth_in.write[i].packet_size = -1;
		}
		bandwidth_out_ptr = 0;
		bandwidth_out.resize(16384); // ~128kB
		for (int i = 0; i < bandwidth_out.size(); ++i) {
			bandwidth_out.write[i].packet_size = -1;
		}
	}
}

void MultiplayerDebugger::BandwidthProfiler::add(const Array &p_data) {
	ERR_FAIL_COND(p_data.size() < 3);
	const String inout = p_data[0];
	int time = p_data[1];
	int size = p_data[2];
	if (inout == "in") {
		bandwidth_in.write[bandwidth_in_ptr].timestamp = time;
		bandwidth_in.write[bandwidth_in_ptr].packet_size = size;
		bandwidth_in_ptr = (bandwidth_in_ptr + 1) % bandwidth_in.size();
	} else if (inout == "out") {
		bandwidth_out.write[bandwidth_out_ptr].timestamp = time;
		bandwidth_out.write[bandwidth_out_ptr].packet_size = size;
		bandwidth_out_ptr = (bandwidth_out_ptr + 1) % bandwidth_out.size();
	}
}

void MultiplayerDebugger::BandwidthProfiler::tick(double p_frame_time, double p_process_time, double p_physics_time, double p_physics_frame_time) {
	uint64_t pt = OS::get_singleton()->get_ticks_msec();
	if (pt - last_bandwidth_time > 200) {
		last_bandwidth_time = pt;
		int incoming_bandwidth = bandwidth_usage(bandwidth_in, bandwidth_in_ptr);
		int outgoing_bandwidth = bandwidth_usage(bandwidth_out, bandwidth_out_ptr);

		Array arr;
		arr.push_back(incoming_bandwidth);
		arr.push_back(outgoing_bandwidth);
		EngineDebugger::get_singleton()->send_message("multiplayer:bandwidth", arr);
	}
}

// RPCProfiler

Array MultiplayerDebugger::RPCFrame::serialize() {
	Array arr;
	arr.push_back(infos.size() * 4);
	for (int i = 0; i < infos.size(); ++i) {
		arr.push_back(uint64_t(infos[i].node));
		arr.push_back(infos[i].node_path);
		arr.push_back(infos[i].incoming_rpc);
		arr.push_back(infos[i].outgoing_rpc);
	}
	return arr;
}

bool MultiplayerDebugger::RPCFrame::deserialize(const Array &p_arr) {
	ERR_FAIL_COND_V(p_arr.size() < 1, false);
	uint32_t size = p_arr[0];
	ERR_FAIL_COND_V(size % 4, false);
	ERR_FAIL_COND_V((uint32_t)p_arr.size() != size + 1, false);
	infos.resize(size / 4);
	int idx = 1;
	for (uint32_t i = 0; i < size / 4; ++i) {
		infos.write[i].node = uint64_t(p_arr[idx]);
		infos.write[i].node_path = p_arr[idx + 1];
		infos.write[i].incoming_rpc = p_arr[idx + 2];
		infos.write[i].outgoing_rpc = p_arr[idx + 3];
	}
	return true;
}

void MultiplayerDebugger::RPCProfiler::init_node(const ObjectID p_node) {
	if (rpc_node_data.has(p_node)) {
		return;
	}
	rpc_node_data.insert(p_node, RPCNodeInfo());
	rpc_node_data[p_node].node = p_node;
	rpc_node_data[p_node].node_path = Object::cast_to<Node>(ObjectDB::get_instance(p_node))->get_path();
	rpc_node_data[p_node].incoming_rpc = 0;
	rpc_node_data[p_node].outgoing_rpc = 0;
}

void MultiplayerDebugger::RPCProfiler::toggle(bool p_enable, const Array &p_opts) {
	rpc_node_data.clear();
}

void MultiplayerDebugger::RPCProfiler::add(const Array &p_data) {
	ERR_FAIL_COND(p_data.size() < 2);
	const ObjectID id = p_data[0];
	const String what = p_data[1];
	init_node(id);
	RPCNodeInfo &info = rpc_node_data[id];
	if (what == "rpc_in") {
		info.incoming_rpc++;
	} else if (what == "rpc_out") {
		info.outgoing_rpc++;
	}
}

void MultiplayerDebugger::RPCProfiler::tick(double p_frame_time, double p_process_time, double p_physics_time, double p_physics_frame_time) {
	uint64_t pt = OS::get_singleton()->get_ticks_msec();
	if (pt - last_profile_time > 100) {
		last_profile_time = pt;
		RPCFrame frame;
		for (const KeyValue<ObjectID, RPCNodeInfo> &E : rpc_node_data) {
			frame.infos.push_back(E.value);
		}
		rpc_node_data.clear();
		EngineDebugger::get_singleton()->send_message("multiplayer:rpc", frame.serialize());
	}
}
