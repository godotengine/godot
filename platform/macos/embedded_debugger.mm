/**************************************************************************/
/*  embedded_debugger.mm                                                  */
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

#include "embedded_debugger.h"

#include "display_server_embedded.h"

#include "core/debugger/engine_debugger.h"
#include "core/input/input_event_codec.h"
#include "core/os/main_loop.h"

#ifdef DEBUG_ENABLED
HashMap<String, EmbeddedDebugger::ParseMessageFunc> EmbeddedDebugger::parse_message_handlers;
#endif

EmbeddedDebugger::EmbeddedDebugger(DisplayServerEmbedded *p_ds) {
	singleton = this;

#ifdef DEBUG_ENABLED
	ds = p_ds;
	if (parse_message_handlers.is_empty()) {
		_init_parse_message_handlers();
	}
	EngineDebugger::register_message_capture("embed", EngineDebugger::Capture(this, EmbeddedDebugger::parse_message));
#endif
}

EmbeddedDebugger::~EmbeddedDebugger() {
	singleton = nullptr;
}

void EmbeddedDebugger::initialize(DisplayServerEmbedded *p_ds) {
	if (EngineDebugger::is_active()) {
		memnew(EmbeddedDebugger(p_ds));
	}
}

void EmbeddedDebugger::deinitialize() {
	if (singleton) {
		memdelete(singleton);
	}
}

#ifdef DEBUG_ENABLED
void EmbeddedDebugger::_init_parse_message_handlers() {
	parse_message_handlers["window_size"] = &EmbeddedDebugger::_msg_window_size;
	parse_message_handlers["mouse_set_mode"] = &EmbeddedDebugger::_msg_mouse_set_mode;
	parse_message_handlers["event"] = &EmbeddedDebugger::_msg_event;
	parse_message_handlers["win_event"] = &EmbeddedDebugger::_msg_win_event;
	parse_message_handlers["notification"] = &EmbeddedDebugger::_msg_notification;
	parse_message_handlers["ime_update"] = &EmbeddedDebugger::_msg_ime_update;
	parse_message_handlers["ds_state"] = &EmbeddedDebugger::_msg_ds_state;
}

Error EmbeddedDebugger::_msg_window_size(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 1, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'window_size' message.");
	Size2i size = p_args[0];
	ds->_window_set_size(size);
	return OK;
}

Error EmbeddedDebugger::_msg_mouse_set_mode(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 1, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'mouse_set_mode' message.");
	DisplayServer::MouseMode mode = p_args[0];
	ds->mouse_set_mode(mode);
	return OK;
}

Error EmbeddedDebugger::_msg_event(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 1, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'event' message.");
	Input *input = Input::get_singleton();
	if (!input) {
		// Ignore if we've received an event before the process has initialized.
		return OK;
	}

	PackedByteArray data = p_args[0];
	Ref<InputEvent> event;
	decode_input_event(data, event);

	{
		Ref<InputEventMouse> e = event;
		if (e.is_valid()) {
			input->set_mouse_position(e->get_position());
		}
	}

	{
		Ref<InputEventMagnifyGesture> e = event;
		if (e.is_valid()) {
			input->set_mouse_position(e->get_position());
		}
	}

	{
		Ref<InputEventPanGesture> e = event;
		if (e.is_valid()) {
			input->set_mouse_position(e->get_position());
		}
	}

	if (event.is_valid()) {
		input->parse_input_event(event);
	}

	return OK;
}

Error EmbeddedDebugger::_msg_win_event(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 1, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'win_event' message.");
	DisplayServer::WindowEvent win_event = p_args[0];
	ds->send_window_event(win_event, DisplayServer::MAIN_WINDOW_ID);
	if (win_event == DisplayServer::WindowEvent::WINDOW_EVENT_MOUSE_EXIT) {
		Input::get_singleton()->release_pressed_events();
	}
	return OK;
}

Error EmbeddedDebugger::_msg_ime_update(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 2, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'ime_update' message.");
	String ime_text = p_args[0];
	Vector2i ime_selection = p_args[1];
	ds->update_im_text(ime_selection, ime_text);
	return OK;
}

Error EmbeddedDebugger::_msg_notification(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 1, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'notification' message.");
	int notification = p_args[0];
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(notification);
	}
	return OK;
}

Error EmbeddedDebugger::_msg_ds_state(const Array &p_args) {
	ERR_FAIL_COND_V_MSG(p_args.size() != 1, ERR_INVALID_PARAMETER, "Invalid number of arguments for 'ds_state' message.");
	PackedByteArray data = p_args[0];
	DisplayServerEmbeddedState state;
	state.deserialize(data);
	ds->set_state(state);
	return OK;
}

Error EmbeddedDebugger::parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured) {
	EmbeddedDebugger *self = static_cast<EmbeddedDebugger *>(p_user);
	r_captured = true;

	ParseMessageFunc *fn_ptr = parse_message_handlers.getptr(p_msg);
	if (fn_ptr) {
		return (self->**fn_ptr)(p_args);
	} else {
		// Any other messages with this prefix should be ignored.
		WARN_PRINT("Unknown message: " + p_msg);
		return ERR_SKIP;
	}
}
#endif
