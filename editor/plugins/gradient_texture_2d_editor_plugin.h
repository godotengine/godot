/*************************************************************************/
/*  gradient_texture_2d_editor_plugin.h                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GRADIENT_TEXTURE_2D_EDITOR
#define GRADIENT_TEXTURE_2D_EDITOR

#include "editor/editor_plugin.h"
#include "editor/editor_spin_slider.h"

class GradientTexture2DEditorRect : public Control {
	GDCLASS(GradientTexture2DEditorRect, Control);

	enum Handle {
		HANDLE_NONE,
		HANDLE_FILL_FROM,
		HANDLE_FILL_TO
	};

	Ref<GradientTexture2D> texture;
	UndoRedo *undo_redo = nullptr;
	bool snap_enabled = false;
	float snap_size = 0;

	TextureRect *checkerboard = nullptr;

	Handle handle = HANDLE_NONE;
	Size2 handle_size;
	Point2 offset;
	Size2 size;

	Point2 _get_handle_position(const Handle p_handle);
	void _update_fill_position();
	void _gui_input(const Ref<InputEvent> &p_event);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_texture(Ref<GradientTexture2D> &p_texture);
	void set_snap_enabled(bool p_snap_enabled);
	void set_snap_size(float p_snap_size);

	GradientTexture2DEditorRect();
};

class GradientTexture2DEditor : public VBoxContainer {
	GDCLASS(GradientTexture2DEditor, VBoxContainer);

	Ref<GradientTexture2D> texture;
	UndoRedo *undo_redo = nullptr;

	Button *reverse_button = nullptr;
	Button *snap_button = nullptr;
	EditorSpinSlider *snap_size_edit = nullptr;
	GradientTexture2DEditorRect *texture_editor_rect = nullptr;

	void _reverse_button_pressed();
	void _set_snap_enabled(bool p_enabled);
	void _set_snap_size(float p_snap_size);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_texture(Ref<GradientTexture2D> &p_texture);

	GradientTexture2DEditor();
};

class EditorInspectorPluginGradientTexture2D : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginGradientTexture2D, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object);
	virtual void parse_begin(Object *p_object);
};

class GradientTexture2DEditorPlugin : public EditorPlugin {
	GDCLASS(GradientTexture2DEditorPlugin, EditorPlugin);

public:
	GradientTexture2DEditorPlugin(EditorNode *p_node);
};

#endif
