/**************************************************************************/
/*  saveload_debugger.h                                                   */
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

#ifndef SAVELOAD_DEBUGGER_H
#define SAVELOAD_DEBUGGER_H

#include "core/debugger/engine_profiler.h"

#include "core/os/os.h"

class SaveloadSynchronizer;

class SaveloadDebugger {
public:
	struct SyncInfo {
		ObjectID synchronizer;
		ObjectID config;
		ObjectID root_node;
		int incoming_syncs = 0;
		int incoming_size = 0;
		int outgoing_syncs = 0;
		int outgoing_size = 0;

		void write_to_array(Array &r_arr) const;
		bool read_from_array(const Array &p_arr, int p_offset);

		SyncInfo() {}
		SyncInfo(SaveloadSynchronizer *p_sync);
	};

	struct SaveloadFrame {
		HashMap<ObjectID, SyncInfo> infos;

		Array serialize();
		bool deserialize(const Array &p_arr);
	};

private:
	class BandwidthProfiler : public EngineProfiler {
	protected:
		struct BandwidthFrame {
			uint32_t timestamp;
			int packet_size;
		};

		int bandwidth_in_ptr = 0;
		Vector<BandwidthFrame> bandwidth_in;
		int bandwidth_out_ptr = 0;
		Vector<BandwidthFrame> bandwidth_out;
		uint64_t last_bandwidth_time = 0;

		int bandwidth_usage(const Vector<BandwidthFrame> &p_buffer, int p_pointer);

	public:
		void toggle(bool p_enable, const Array &p_opts);
		void add(const Array &p_data);
		void tick(double p_frame_time, double p_process_time, double p_physics_time, double p_physics_frame_time);
	};

	class SaveloadProfiler : public EngineProfiler {
	private:
		HashMap<ObjectID, SyncInfo> sync_data;
		uint64_t last_profile_time = 0;

	public:
		void toggle(bool p_enable, const Array &p_opts);
		void add(const Array &p_data);
		void tick(double p_frame_time, double p_process_time, double p_physics_time, double p_physics_frame_time);
	};

	static Error _capture(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured);

public:
	static void initialize();
	static void deinitialize();
};

#endif // SAVELOAD_DEBUGGER_H
