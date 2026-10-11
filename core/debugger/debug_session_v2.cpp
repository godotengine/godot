/**************************************************************************/
/*  debug_session_v2.cpp                                                  */
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
/* "Software"), to deal in the Software without restriction, including  */
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

#include "debug_session_v2.h"

DebugSessionV2::Decision DebugSessionV2::_remember(const Decision &p_decision) {
	if (p_decision.request_id != 0) {
		request_cache[p_decision.request_id] = p_decision;
		highest_request_id = MAX(highest_request_id, p_decision.request_id);
		while (request_cache.size() > 64) {
			request_cache.erase(request_cache.front()->key());
		}
	}
	return p_decision;
}

DebugSessionV2::Decision DebugSessionV2::_reject(const DebugProtocolV2::Envelope &p_message, const String &p_reason) {
	Decision decision;
	decision.request_id = p_message.request_id;
	decision.stop_id = p_message.stop_id;
	decision.reason = p_reason;
	return _remember(decision);
}

void DebugSessionV2::connected(uint32_t p_generation, uint64_t p_now_msec) {
	state = DebugProtocolV2::STATE_RUNNING;
	generation = p_generation;
	current_stop_id = 0;
	highest_request_id = 0;
	last_heartbeat_msec = p_now_msec;
	can_continue = true;
	request_cache.clear();
}

uint64_t DebugSessionV2::enter_paused(bool p_can_continue, uint64_t p_now_msec) {
	state = DebugProtocolV2::STATE_PAUSED;
	current_stop_id++;
	if (current_stop_id == 0) {
		current_stop_id = 1;
	}
	can_continue = p_can_continue;
	last_heartbeat_msec = p_now_msec;
	return current_stop_id;
}

void DebugSessionV2::mark_running(uint64_t p_now_msec) {
	state = DebugProtocolV2::STATE_RUNNING;
	last_heartbeat_msec = p_now_msec;
}

void DebugSessionV2::control_lost() {
	state = DebugProtocolV2::STATE_DISCONNECTED;
	can_continue = true;
}

DebugSessionV2::Decision DebugSessionV2::handle_control(const DebugProtocolV2::Envelope &p_message, uint64_t p_now_msec) {
	if (p_message.type == "heartbeat") {
		last_heartbeat_msec = p_now_msec;
		Decision decision;
		decision.accepted = true;
		decision.request_id = p_message.request_id;
		decision.stop_id = current_stop_id;
		decision.reason = "heartbeat";
		return decision;
	}

	if (p_message.request_id == 0) {
		return _reject(p_message, "missing_request_id");
	}
	if (request_cache.has(p_message.request_id)) {
		Decision duplicate = request_cache[p_message.request_id];
		duplicate.action = ACTION_NONE;
		duplicate.duplicate = true;
		return duplicate;
	}
	if (p_message.request_id < highest_request_id) {
		return _reject(p_message, "out_of_order_request_id");
	}

	Decision decision;
	decision.request_id = p_message.request_id;
	decision.stop_id = p_message.stop_id;

	if (p_message.type == "pause") {
		if (state == DebugProtocolV2::STATE_RUNNING) {
			state = DebugProtocolV2::STATE_PAUSE_REQUESTED;
			decision.action = ACTION_PAUSE;
			decision.accepted = true;
			decision.reason = "accepted";
		} else if (state == DebugProtocolV2::STATE_PAUSE_REQUESTED || state == DebugProtocolV2::STATE_PAUSED) {
			decision.accepted = true;
			decision.stop_id = current_stop_id;
			decision.reason = "already_paused";
		} else {
			decision.reason = "invalid_state";
		}
		return _remember(decision);
	}

	if (p_message.type == "continue" || p_message.type == "step" || p_message.type == "next" || p_message.type == "step_out") {
		if (state != DebugProtocolV2::STATE_PAUSED) {
			decision.reason = "not_paused";
			return _remember(decision);
		}
		if (p_message.stop_id != current_stop_id) {
			decision.reason = "stale_stop_id";
			return _remember(decision);
		}
		if (!can_continue) {
			decision.reason = "cannot_continue";
			return _remember(decision);
		}

		state = DebugProtocolV2::STATE_RESUME_REQUESTED;
		decision.accepted = true;
		decision.reason = "accepted";
		if (p_message.type == "continue") {
			decision.action = ACTION_CONTINUE;
		} else if (p_message.type == "step") {
			decision.action = ACTION_STEP;
		} else if (p_message.type == "step_out") {
			decision.action = ACTION_STEP_OUT;
		} else {
			decision.action = ACTION_NEXT;
		}
		return _remember(decision);
	}

	return _reject(p_message, "unknown_control_message");
}

bool DebugSessionV2::heartbeat_expired(uint64_t p_now_msec, uint32_t p_timeout_msec) const {
	return is_paused() && p_now_msec > last_heartbeat_msec && p_now_msec - last_heartbeat_msec > p_timeout_msec;
}
