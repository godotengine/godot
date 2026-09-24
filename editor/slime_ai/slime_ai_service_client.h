/**************************************************************************/
/*  slime_ai_service_client.h                                           */
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

#include "editor/slime_ai/slime_ai_protocol.h"
#include "core/io/file_access.h"
#include "core/os/os.h"

namespace SlimeAI {

class ServiceClient {
	Ref<FileAccess> stdio;
	Ref<FileAccess> stderr_pipe;
	ProcessID pid = 0;
	FrameDecoder decoder;
	String pending_id;
	String pending_method;
	String active_run_request_id;
	String active_run_id;
	String state = "disconnected";
	String last_error;
	uint64_t request_number = 0;

public:
	bool start(const String &p_script_path);
	String request(const String &p_method, const Dictionary &p_params);
	void poll(Vector<Dictionary> &r_responses);
	void stop();
	String get_state() const { return state; }
	String get_last_error() const { return last_error; }
	String get_pending_method() const { return pending_method; }
	String get_active_run_id() const { return active_run_id; }
	bool is_ready() const { return state == "ready"; }
	~ServiceClient();
};

} // namespace SlimeAI
