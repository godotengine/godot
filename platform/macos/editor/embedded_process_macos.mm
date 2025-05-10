/**************************************************************************/
/*  embedded_process_macos.mm                                             */
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

#include "embedded_process_macos.h"

#include "platform/macos/display_server_macos.h"

#include "core/input/input_event_codec.h"
#include "editor/debugger/script_editor_debugger.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "scene/gui/control.h"
#include "scene/main/window.h"

void EmbeddedProcessMacOS::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED:
		case NOTIFICATION_VISIBILITY_CHANGED: {
			update_embedded_process();
		} break;
	}
}

void EmbeddedProcessMacOS::update_embedded_process() const {
	layer_host->set_rect(get_adjusted_embedded_window_rect(get_rect()));
	if (is_embedding_completed()) {
		ds->embed_process_update(window->get_window_id(), this);
		Rect2i rect = get_screen_embedded_window_rect();
		script_debugger->send_message("embed:window_size", { rect.size });
	}
}

void EmbeddedProcessMacOS::set_context_id(uint32_t p_context_id) {
	if (!window) {
		return;
	}

	context_id = p_context_id;

	_try_embed_process();
}

void EmbeddedProcessMacOS::set_script_debugger(ScriptEditorDebugger *p_debugger) {
	script_debugger = p_debugger;
	layer_host->set_script_debugger(script_debugger);
	_try_embed_process();
}

void EmbeddedProcessMacOS::embed_process(OS::ProcessID p_pid) {
	if (!window) {
		return;
	}

	if (current_process_id != 0) {
		// Stop embedding the last process.
		OS::get_singleton()->kill(current_process_id);
	}

	reset();

	current_process_id = p_pid;
	embedding_state = EmbeddingState::IN_PROGRESS;
	// Attempt to embed the process, but if it has just started and the window is not ready yet,
	// we will retry in this case.
	_try_embed_process();
}

void EmbeddedProcessMacOS::_joy_connection_changed(int p_index, bool p_connected) const {
	if (!script_debugger) {
		return;
	}

	if (p_connected) {
		String name = Input::get_singleton()->get_joy_name(p_index);
		script_debugger->send_message("embed:joy_add", { p_index, name });
	} else {
		script_debugger->send_message("embed:joy_del", { p_index });
	}
}

void EmbeddedProcessMacOS::reset() {
	if (!ds) {
		ds = static_cast<DisplayServerMacOS *>(DisplayServer::get_singleton());
	}
	if (current_process_id != 0 && is_embedding_completed()) {
		ds->remove_embedded_process(current_process_id);
	}
	current_process_id = 0;
	embedding_state = EmbeddingState::IDLE;
	context_id = 0;
	script_debugger = nullptr;
	queue_redraw();
}

void EmbeddedProcessMacOS::request_close() {
	if (current_process_id != 0 && is_embedding_completed()) {
		ds->request_close_embedded_process(current_process_id);
	}
}

void EmbeddedProcessMacOS::_try_embed_process() {
	if (current_process_id == 0 || script_debugger == nullptr || context_id == 0) {
		return;
	}

	Error err = ds->embed_process_update(window->get_window_id(), this);
	if (err == OK) {
		Rect2i rect = get_screen_embedded_window_rect();
		script_debugger->send_message("embed:window_size", { rect.size });
		embedding_state = EmbeddingState::COMPLETED;
		queue_redraw();
		emit_signal(SNAME("embedding_completed"));

		// Replicate some of the DisplayServer state.
		{
			Dictionary state;
			state["screen_get_max_scale"] = ds->screen_get_max_scale();
			// script_debugger->send_message("embed:ds_state", { state });
		}

		// Send initial joystick state.
		{
			Input *input = Input::get_singleton();
			TypedArray<int> joy_pads = input->get_connected_joypads();
			for (const Variant &idx : joy_pads) {
				String name = input->get_joy_name(idx);
				script_debugger->send_message("embed:joy_add", { idx, name });
			}
		}

		layer_host->grab_focus();
	} else {
		// Another unknown error.
		reset();
		emit_signal(SNAME("embedding_failed"));
	}
}

Rect2i EmbeddedProcessMacOS::get_adjusted_embedded_window_rect(const Rect2i &p_rect) const {
	Rect2i control_rect = Rect2i(p_rect.position + margin_top_left, (p_rect.size - get_margins_size()).maxi(1));
	if (window_size != Size2i()) {
		Rect2i desired_rect;
		if (!keep_aspect && control_rect.size.x >= window_size.x && control_rect.size.y >= window_size.y) {
			// Fixed at the desired size.
			desired_rect.size = window_size;
		} else {
			float ratio = MIN((float)control_rect.size.x / window_size.x, (float)control_rect.size.y / window_size.y);
			desired_rect.size = Size2i(window_size.x * ratio, window_size.y * ratio).maxi(1);
		}
		desired_rect.position = Size2i(control_rect.position.x + ((control_rect.size.x - desired_rect.size.x) / 2), control_rect.position.y + ((control_rect.size.y - desired_rect.size.y) / 2));
		return desired_rect;
	} else {
		// Stretch, use all the control area.
		return control_rect;
	}
}

void EmbeddedProcessMacOS::mouse_set_mode(DisplayServer::MouseMode p_mode) {
	mouse_mode = p_mode;
	// If the mouse is anything other than visible, we must ensure the Game view is active and the layer focused.
	if (mouse_mode != DisplayServer::MOUSE_MODE_VISIBLE) {
		EditorNode::get_singleton()->get_editor_main_screen()->select(EditorMainScreen::EDITOR_GAME);
		layer_host->grab_focus();
	}
	DisplayServer::get_singleton()->mouse_set_mode(p_mode);
}

EmbeddedProcessMacOS::EmbeddedProcessMacOS() :
		EmbeddedProcessBase() {
	layer_host = memnew(LayerHost(this));
	add_child(layer_host);
	layer_host->set_focus_mode(FOCUS_ALL);
	layer_host->set_anchors_and_offsets_preset(PRESET_FULL_RECT);
	layer_host->set_custom_minimum_size(Size2(100, 100));

	Input *input = Input::get_singleton();
	input->connect(SNAME("joy_connection_changed"), callable_mp(this, &EmbeddedProcessMacOS::_joy_connection_changed));

	// This shortcut allows a user to forcibly release a captured mouse from within the editor, regardless of whether
	// the embedded process has implemented support to release the cursor.
	ED_SHORTCUT("game_view/release_mouse", TTRC("Release Mouse"), KeyModifierMask::ALT | Key::ESCAPE);
}

void LayerHost::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_FOCUS_ENTER: {
			if (script_debugger) {
				script_debugger->send_message("embed:win_event", { DisplayServer::WINDOW_EVENT_MOUSE_ENTER });
			}
			// Temporarily release mouse capture, so we can interact with the editor.
			DisplayServer *ds = DisplayServer::get_singleton();
			if (process->get_mouse_mode() != ds->mouse_get_mode()) {
				// Restore embedded process mouse mode.
				ds->mouse_set_mode(process->get_mouse_mode());
			}
		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			if (script_debugger) {
				script_debugger->send_message("embed:win_event", { DisplayServer::WINDOW_EVENT_MOUSE_EXIT });
			}
			// Temporarily set mouse state back to visible, so the user can interact with the editor.
			DisplayServer *ds = DisplayServer::get_singleton();
			if (ds->mouse_get_mode() != DisplayServer::MOUSE_MODE_VISIBLE) {
				ds->mouse_set_mode(DisplayServer::MOUSE_MODE_VISIBLE);
			}
		} break;
		case MainLoop::NOTIFICATION_OS_IME_UPDATE: {
			if (script_debugger && has_focus()) {
				const String ime_text = DisplayServer::get_singleton()->ime_get_text();
				const Vector2i ime_selection = DisplayServer::get_singleton()->ime_get_selection();
				script_debugger->send_message("embed:ime_update", { ime_text, ime_selection });
			}
		} break;
	}
}

void LayerHost::gui_input(const Ref<InputEvent> &p_event) {
	if (!process->is_embedding_completed()) {
		return;
	}

	if (p_event->is_pressed()) {
		if (ED_IS_SHORTCUT("game_view/release_mouse", p_event)) {
			DisplayServer *ds = DisplayServer::get_singleton();
			if (ds->mouse_get_mode() != DisplayServer::MOUSE_MODE_VISIBLE) {
				ds->mouse_set_mode(DisplayServer::MOUSE_MODE_VISIBLE);
				script_debugger->send_message("embed:mouse_set_mode", { DisplayServer::MOUSE_MODE_VISIBLE });
			}
			accept_event();
			return;
		}
	}

	PackedByteArray data;
	if (encode_input_event(p_event, data)) {
		script_debugger->send_message("embed:event", { data });
		accept_event();
	}
}

LayerHost::LayerHost(EmbeddedProcessMacOS *p_process) :
		process(p_process) {}
