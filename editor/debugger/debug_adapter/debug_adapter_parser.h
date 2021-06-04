/*************************************************************************/
/*  debug_adapter_parser.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DEBUG_ADAPTER_PARSER_H
#define DEBUG_ADAPTER_PARSER_H

#include "debug_adapter_protocol.h"
#include "debug_adapter_types.h"

struct DAPeer;
class DebugAdapterProtocol;

class DebugAdapterParser : public Object {
	GDCLASS(DebugAdapterParser, Object);

private:
	int messageId;

	friend DebugAdapterProtocol;

protected:
	static void _bind_methods();

	Dictionary prepare_base_message(Ref<DAPeer> p_peer) const;
	Dictionary prepare_base_event(Ref<DAPeer> p_peer) const;
	Dictionary prepare_success_response(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary prepare_error_response(Ref<DAPeer> p_peer, const Dictionary &p_params, DAP::ErrorType err_type);

public:
	// Requests
	Dictionary req_initialize(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary req_launch(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary req_terminate(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary req_pause(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary req_continue(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary req_threads(Ref<DAPeer> p_peer, const Dictionary &p_params) const;
	Dictionary req_stackTrace(Ref<DAPeer> p_peer, const Dictionary &p_params) const;

	// Events
	Dictionary ev_initialized(Ref<DAPeer> p_peer) const;
	Dictionary ev_process(Ref<DAPeer> p_peer, const String &p_command) const;
	Dictionary ev_terminated(Ref<DAPeer> p_peer) const;
	Dictionary ev_exited(Ref<DAPeer> p_peer, const int &p_exitcode) const;
	Dictionary ev_stopped(Ref<DAPeer> p_peer, DAP::StopReason p_reason) const;
	Dictionary ev_continued(Ref<DAPeer> p_peer) const;

	DebugAdapterParser();

	void reset_ids();
	int generate_message_id();
};

#endif
