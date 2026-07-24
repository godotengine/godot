/**************************************************************************/
/*  audio_stream_player_2d_editor_plugin.h                                */
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

#include "editor/plugins/editor_plugin.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "scene/2d/audio_stream_player_2d.h"

class AudioStreamPlayer2DEditor : public Control {
	GDCLASS(AudioStreamPlayer2DEditor, Control);

	CanvasItemEditor *canvas_item_editor = nullptr;
	AudioStreamPlayer2D *node = nullptr;

	bool pressed = false;
	float original_max_distance = 0.0;
	float grab_threshold = 8.0;

protected:
	void _node_removed(Node *p_node);
	void _notification(int p_what);

public:
	bool forward_canvas_gui_input(const Ref<InputEvent> &p_event);
	void forward_canvas_draw_over_viewport(Control *p_overlay);
	void edit(AudioStreamPlayer2D *p_node);

	AudioStreamPlayer2DEditor();
};

class AudioStreamPlayer2DEditorPlugin : public EditorPlugin {
	GDCLASS(AudioStreamPlayer2DEditorPlugin, EditorPlugin);

	AudioStreamPlayer2DEditor *audio_editor = nullptr;

public:
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) override { return audio_editor->forward_canvas_gui_input(p_event); }
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) override { audio_editor->forward_canvas_draw_over_viewport(p_overlay); }

	virtual String get_plugin_name() const override { return "AudioStreamPlayer2D"; }
	virtual bool handles(Object *p_object) const override;
	virtual void edit(Object *p_object) override;
	virtual void make_visible(bool p_visible) override;

	AudioStreamPlayer2DEditorPlugin();
};
