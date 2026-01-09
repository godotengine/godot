/**************************************************************************/
/*  embedded_process_macos.h                                              */
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

#include "editor/run/embedded_process.h"

class DisplayServerMacOS;
class EmbeddedProcessMacOS;

class LayerHost final : public Control {
	GDCLASS(LayerHost, Control);

	ScriptEditorDebugger *script_debugger = nullptr;
	EmbeddedProcessMacOS *process = nullptr;
	bool window_focused = true;

	struct CustomCursor {
		Ref<Image> image;
		Vector2 hotspot;
		CustomCursor() {}
		CustomCursor(const Ref<Image> &p_image, const Vector2 &p_hotspot) {
			image = p_image;
			hotspot = p_hotspot;
		}
	};
	HashMap<DisplayServer::CursorShape, CustomCursor> custom_cursors;

	virtual void gui_input(const Ref<InputEvent> &p_event) override;

protected:
	void _notification(int p_what);

public:
	void cursor_set_custom_image(const Ref<Image> &p_image, DisplayServer::CursorShape p_shape, const Vector2 &p_hotspot);
	void set_script_debugger(ScriptEditorDebugger *p_debugger) {
		script_debugger = p_debugger;
	}

	LayerHost(EmbeddedProcessMacOS *p_process);
};

class EmbeddedProcessMacOS final : public EmbeddedProcessBase {
	GDCLASS(EmbeddedProcessMacOS, EmbeddedProcessBase);

	enum class EmbeddingState {
		IDLE,
		IN_PROGRESS,
		COMPLETED,
		FAILED,
	};

	DisplayServerMacOS *ds = nullptr;
	EmbeddingState embedding_state = EmbeddingState::IDLE;
	uint32_t context_id = 0;
	ScriptEditorDebugger *script_debugger = nullptr;
	LayerHost *layer_host = nullptr;
	OS::ProcessID current_process_id = 0;

	// Embedded process state.

	// The last mouse mode sent by the embedded process.
	DisplayServer::MouseMode mouse_mode = DisplayServer::MOUSE_MODE_VISIBLE;

	// Helper functions.

	void _try_embed_process();
	void update_embedded_process();

protected:
	void _notification(int p_what);

public:
	// MARK: - Message Handlers

	void set_context_id(uint32_t p_context_id);
	void mouse_set_mode(DisplayServer::MouseMode p_mode);

	uint32_t get_context_id() const { return context_id; }
	void set_script_debugger(ScriptEditorDebugger *p_debugger) override;

	bool is_embedding_in_progress() const override {
		return embedding_state == EmbeddingState::IN_PROGRESS;
	}

	_FORCE_INLINE_ bool is_embedding_completed() const override {
		return embedding_state == EmbeddingState::COMPLETED;
	}

	bool is_process_focused() const override { return layer_host->has_focus(); }
	void embed_process(OS::ProcessID p_pid) override;
	int get_embedded_pid() const override { return current_process_id; }
	void reset() override;
	void request_close() override;
	void queue_update_embedded_process() override { update_embedded_process(); }

	Rect2i get_adjusted_embedded_window_rect(const Rect2i &p_rect) const override;

	_FORCE_INLINE_ LayerHost *get_layer_host() const { return layer_host; }

	void display_state_changed();

	// MARK: - Embedded process state
	_FORCE_INLINE_ DisplayServer::MouseMode get_mouse_mode() const { return mouse_mode; }

	EmbeddedProcessMacOS();
	~EmbeddedProcessMacOS() override;
};
