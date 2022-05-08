/*************************************************************************/
/*  sprite_frames_editor_plugin.h                                        */
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

#ifndef SPRITE_FRAMES_EDITOR_PLUGIN_H
#define SPRITE_FRAMES_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "scene/2d/animated_sprite_2d.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

class EditorFileDialog;

class SpriteFramesEditor : public HSplitContainer {
	GDCLASS(SpriteFramesEditor, HSplitContainer);

	enum {
		PARAM_USE_CURRENT, // Used in callbacks to indicate `dominant_param` should be not updated.
		PARAM_FRAME_COUNT, // Keep "Horizontal" & "Vertial" values.
		PARAM_SIZE, // Keep "Size" values.
	};
	int dominant_param = PARAM_FRAME_COUNT;

	Button *load = nullptr;
	Button *load_sheet = nullptr;
	Button *_delete = nullptr;
	Button *copy = nullptr;
	Button *paste = nullptr;
	Button *empty = nullptr;
	Button *empty2 = nullptr;
	Button *move_up = nullptr;
	Button *move_down = nullptr;
	Button *zoom_out = nullptr;
	Button *zoom_reset = nullptr;
	Button *zoom_in = nullptr;
	ItemList *tree = nullptr;
	bool loading_scene;
	int sel;

	Button *new_anim = nullptr;
	Button *remove_anim = nullptr;

	Tree *animations = nullptr;
	SpinBox *anim_speed = nullptr;
	CheckButton *anim_loop = nullptr;

	EditorFileDialog *file = nullptr;

	AcceptDialog *dialog = nullptr;

	SpriteFrames *frames = nullptr;

	StringName edited_anim;

	ConfirmationDialog *delete_dialog = nullptr;

	ConfirmationDialog *split_sheet_dialog = nullptr;
	ScrollContainer *split_sheet_scroll = nullptr;
	TextureRect *split_sheet_preview = nullptr;
	SpinBox *split_sheet_h = nullptr;
	SpinBox *split_sheet_v = nullptr;
	SpinBox *split_sheet_size_x = nullptr;
	SpinBox *split_sheet_size_y = nullptr;
	SpinBox *split_sheet_sep_x = nullptr;
	SpinBox *split_sheet_sep_y = nullptr;
	SpinBox *split_sheet_offset_x = nullptr;
	SpinBox *split_sheet_offset_y = nullptr;
	Button *split_sheet_zoom_out = nullptr;
	Button *split_sheet_zoom_reset = nullptr;
	Button *split_sheet_zoom_in = nullptr;
	EditorFileDialog *file_split_sheet = nullptr;
	Set<int> frames_selected;
	Set<int> frames_toggled_by_mouse_hover;
	int last_frame_selected = 0;

	float scale_ratio;
	int thumbnail_default_size;
	float thumbnail_zoom;
	float max_thumbnail_zoom;
	float min_thumbnail_zoom;
	float sheet_zoom;
	float max_sheet_zoom;
	float min_sheet_zoom;

	Size2i _get_frame_count() const;
	Size2i _get_frame_size() const;
	Size2i _get_offset() const;
	Size2i _get_separation() const;

	void _load_pressed();
	void _file_load_request(const Vector<String> &p_path, int p_at_pos = -1);
	void _copy_pressed();
	void _paste_pressed();
	void _empty_pressed();
	void _empty2_pressed();
	void _delete_pressed();
	void _up_pressed();
	void _down_pressed();
	void _update_library(bool p_skip_selector = false);

	void _animation_select();
	void _animation_name_edited();
	void _animation_add();
	void _animation_remove();
	void _animation_remove_confirmed();
	void _animation_loop_changed();
	void _animation_fps_changed(double p_value);

	void _tree_input(const Ref<InputEvent> &p_event);
	void _zoom_in();
	void _zoom_out();
	void _zoom_reset();

	bool updating;
	bool updating_split_settings = false; // Skip SpinBox/Range callback when setting value by code.

	UndoRedo *undo_redo = nullptr;

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _open_sprite_sheet();
	void _prepare_sprite_sheet(const String &p_file);
	int _sheet_preview_position_to_frame_index(const Vector2 &p_position);
	void _sheet_preview_draw();
	void _sheet_spin_changed(double p_value, int p_dominant_param);
	void _sheet_preview_input(const Ref<InputEvent> &p_event);
	void _sheet_scroll_input(const Ref<InputEvent> &p_event);
	void _sheet_add_frames();
	void _sheet_zoom_on_position(float p_zoom, const Vector2 &p_position);
	void _sheet_zoom_in();
	void _sheet_zoom_out();
	void _sheet_zoom_reset();
	void _sheet_select_clear_all_frames();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }

	void edit(SpriteFrames *p_frames);
	SpriteFramesEditor();
};

class SpriteFramesEditorPlugin : public EditorPlugin {
	GDCLASS(SpriteFramesEditorPlugin, EditorPlugin);

	SpriteFramesEditor *frames_editor = nullptr;
	Button *button = nullptr;

public:
	virtual String get_name() const override { return "SpriteFrames"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	SpriteFramesEditorPlugin();
	~SpriteFramesEditorPlugin();
};

#endif // SPRITE_FRAMES_EDITOR_PLUGIN_H
