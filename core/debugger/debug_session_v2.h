/**************************************************************************/
/*  debug_session_v2.h                                                    */
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

#include "core/templates/rb_map.h"
#include "debug_protocol_v2.h"

class DebugSessionV2 {
public:
	enum Action {
		ACTION_NONE,
		ACTION_PAUSE,
		ACTION_CONTINUE,
		ACTION_STEP,
		ACTION_NEXT,
		ACTION_STEP_OUT,
	};

	struct Decision {
		Action action = ACTION_NONE;
		bool accepted = false;
		bool duplicate = false;
		uint64_t request_id = 0;
		uint64_t stop_id = 0;
		String reason;
	};

private:
	DebugProtocolV2::State state = DebugProtocolV2::STATE_DISCONNECTED;
	uint32_t generation = 0;
	uint64_t current_stop_id = 0;
	uint64_t highest_request_id = 0;
	uint64_t last_heartbeat_msec = 0;
	bool can_continue = true;
	RBMap<uint64_t, Decision> request_cache;

	Decision _remember(const Decision &p_decision);
	Decision _reject(const DebugProtocolV2::Envelope &p_message, const String &p_reason);

public:
	void connected(uint32_t p_generation, uint64_t p_now_msec);
	uint64_t enter_paused(bool p_can_continue, uint64_t p_now_msec);
	void mark_running(uint64_t p_now_msec);
	void control_lost();

	Decision handle_control(const DebugProtocolV2::Envelope &p_message, uint64_t p_now_msec);
	bool heartbeat_expired(uint64_t p_now_msec, uint32_t p_timeout_msec) const;

	DebugProtocolV2::State get_state() const { return state; }
	uint32_t get_generation() const { return generation; }
	uint64_t get_current_stop_id() const { return current_stop_id; }
	uint64_t get_last_heartbeat_msec() const { return last_heartbeat_msec; }
	bool is_paused() const { return state == DebugProtocolV2::STATE_PAUSED || state == DebugProtocolV2::STATE_RESUME_REQUESTED; }
};
