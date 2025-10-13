/**************************************************************************/
/*  embedded_process.h                                                    */
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

#include "scene/gui/control.h"

class ScriptEditorDebugger;
class Timer;

class EmbeddedProcessBase : public Control {
	GDCLASS(EmbeddedProcessBase, Control);

	void _draw();

protected:
	Ref<StyleBox> focus_style_box;
	Size2i window_size;
	bool keep_aspect = false;
	Point2i margin_top_left;
	Point2i margin_bottom_right;
	Window *window = nullptr;

	bool transp_enabled = false;
	Color clear_color;
	Ref<Texture2D> checkerboard;

	void _project_settings_changed();

	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual void set_script_debugger(ScriptEditorDebugger *p_debugger) {}

	virtual bool is_embedding_completed() const = 0;
	virtual bool is_embedding_in_progress() const = 0;
	virtual bool is_process_focused() const = 0;
	virtual void embed_process(OS::ProcessID p_pid) = 0;
	virtual int get_embedded_pid() const = 0;
	virtual void reset() = 0;
	virtual void request_close() = 0;
	virtual void queue_update_embedded_process() = 0;

	void set_window_size(const Size2i &p_window_size);
	void set_keep_aspect(bool p_keep_aspect);
	virtual Rect2i get_adjusted_embedded_window_rect(const Rect2i &p_rect) const = 0;
	Rect2i get_screen_embedded_window_rect() const;
	int get_margin_size(Side p_side) const;
	Size2 get_margins_size() const;

	EmbeddedProcessBase();
	virtual ~EmbeddedProcessBase();
};

class EmbeddedProcess : public EmbeddedProcessBase {
	GDCLASS(EmbeddedProcess, EmbeddedProcessBase);

	bool application_has_focus = true;
	uint64_t last_application_focus_time = 0;
	OS::ProcessID focused_process_id = 0;
	OS::ProcessID current_process_id = 0;
	bool embedding_grab_focus = false;
	bool embedding_completed = false;
	uint64_t start_embedding_time = 0;
	bool updated_embedded_process_queued = false;
	bool last_updated_embedded_process_focused = false;

	Timer *timer_embedding = nullptr;
	Timer *timer_update_embedded_process = nullptr;

	const int embedding_timeout = 45000;

	Rect2i last_global_rect;

	void _try_embed_process();
	void _update_embedded_process();
	void _timer_embedding_timeout();
	void _timer_update_embedded_process_timeout();
	void _check_mouse_over();
	void _check_focused_process_id();
	bool _is_embedded_process_updatable();
	Window *_get_current_modal_window();

protected:
	void _notification(int p_what);

public:
	bool is_embedding_in_progress() const override;
	bool is_embedding_completed() const override;
	bool is_process_focused() const override;
	void embed_process(OS::ProcessID p_pid) override;
	int get_embedded_pid() const override;
	void reset() override;
	void request_close() override;
	void queue_update_embedded_process() override;

	Rect2i get_adjusted_embedded_window_rect(const Rect2i &p_rect) const override;

	EmbeddedProcess();
	~EmbeddedProcess() override;
};
