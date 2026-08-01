/**************************************************************************/
/*  test_multiplayer_debugger.h                                          */
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

#include "../multiplayer_debugger.h"
#include "../multiplayer_synchronizer.h"

#include "core/debugger/engine_debugger.h"
#include "core/os/os.h"
#include "core/templates/list.h"
#include "tests/test_macros.h"

namespace TestMultiplayerDebugger {

class TestReplicationProfilerCapture : public EngineDebugger {
public:
	List<Array> messages;

	TestReplicationProfilerCapture() { singleton = this; }
	~TestReplicationProfilerCapture() { singleton = nullptr; }

	void send_message(const String &p_message, const Array &p_data) override {
		if (p_message == "multiplayer:syncs") {
			messages.push_back(p_data);
		}
	}
	void send_error(const String &p_func, const String &p_file, int p_line, const String &p_err, const String &p_descr, bool p_editor_notify, ErrorHandlerType p_type) override {}
	void debug(bool p_can_continue = true, bool p_is_error_breakpoint = false) override {}
};

// ReplicationProfiler::tick() only flushes a frame every 100msec of engine time, and its timer is shared static state across test cases,
// so poll for a bit instead of assuming a single iteration() is enough.
void wait_for_replication_profiler_message(TestReplicationProfilerCapture &p_debugger) {
	for (int i = 0; i < 21 && p_debugger.messages.is_empty(); i++) {
		EngineDebugger::get_singleton()->iteration(0, 0, 0, 0.0);
		OS::get_singleton()->delay_usec(5000);
	}
}

TEST_CASE("[Multiplayer][MultiplayerDebugger] ReplicationProfiler counts delta syncs alongside full syncs") {
	REQUIRE(EngineDebugger::has_profiler("multiplayer:replication"));

	TestReplicationProfilerCapture debugger;
	MultiplayerSynchronizer *sync = memnew(MultiplayerSynchronizer);
	const ObjectID id = sync->get_instance_id();

	EngineDebugger::get_singleton()->profiler_enable("multiplayer:replication", true);

	// "sync_in"/"sync_out" come from full (REPLICATION_MODE_ALWAYS) syncs,
	// "delta_in"/"delta_out" come from delta (REPLICATION_MODE_ON_CHANGE)
	Array sync_in_data = { String("sync_in"), id, 10 };
	EngineDebugger::profiler_add_frame_data("multiplayer:replication", sync_in_data);
	Array sync_out_data = { String("sync_out"), id, 5 };
	EngineDebugger::profiler_add_frame_data("multiplayer:replication", sync_out_data);
	Array delta_in_data = { String("delta_in"), id, 42 };
	EngineDebugger::profiler_add_frame_data("multiplayer:replication", delta_in_data);
	Array delta_out_data = { String("delta_out"), id, 24 };
	EngineDebugger::profiler_add_frame_data("multiplayer:replication", delta_out_data);

	wait_for_replication_profiler_message(debugger);
	REQUIRE_FALSE(debugger.messages.is_empty());

	MultiplayerDebugger::ReplicationFrame frame;
	REQUIRE(frame.deserialize(debugger.messages.back()->get()));
	REQUIRE(frame.infos.has(id));

	const MultiplayerDebugger::SyncInfo &info = frame.infos[id];
	CHECK_EQ(info.incoming_syncs, 2); // sync_in + delta_in
	CHECK_EQ(info.incoming_size, 52);
	CHECK_EQ(info.outgoing_syncs, 2); // sync_out + delta_out
	CHECK_EQ(info.outgoing_size, 29);

	EngineDebugger::get_singleton()->profiler_enable("multiplayer:replication", false);
	memdelete(sync);
}

} // namespace TestMultiplayerDebugger
