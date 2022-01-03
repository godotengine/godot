/*************************************************************************/
/*  animation_bezier_editor.h                                            */
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

#ifndef ANIMATION_BEZIER_EDITOR_H
#define ANIMATION_BEZIER_EDITOR_H

#include "animation_track_editor.h"

class AnimationBezierTrackEdit : public Control {
	GDCLASS(AnimationBezierTrackEdit, Control);

	enum {
		MENU_KEY_INSERT,
		MENU_KEY_DUPLICATE,
		MENU_KEY_DELETE,
		MENU_KEY_SET_HANDLE_FREE,
		MENU_KEY_SET_HANDLE_BALANCED,
	};

	VBoxContainer *right_column;
	Button *close_button;

	AnimationTimelineEdit *timeline = nullptr;
	UndoRedo *undo_redo = nullptr;
	Node *root = nullptr;
	Control *play_position; //separate control used to draw so updates for only position changed are much faster
	float play_position_pos = 0;

	Ref<Animation> animation;
	int track;

	Vector<Rect2> view_rects;

	Ref<Texture2D> bezier_icon;
	Ref<Texture2D> bezier_handle_icon;
	Ref<Texture2D> selected_icon;

	Map<int, Rect2> subtracks;

	float v_scroll = 0;
	float v_zoom = 1;

	PopupMenu *menu = nullptr;

	void _zoom_changed();

	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _menu_selected(int p_index);

	void _play_position_draw();

	Vector2 insert_at_pos;

	bool moving_selection_attempt = false;
	int select_single_attempt = -1;
	bool moving_selection = false;
	int moving_selection_from_key;

	Vector2 moving_selection_offset;

	bool box_selecting_attempt = false;
	bool box_selecting = false;
	bool box_selecting_add = false;
	Vector2 box_selection_from;
	Vector2 box_selection_to;

	int moving_handle = 0; //0 no move -1 or +1 out
	int moving_handle_key = 0;
	Vector2 moving_handle_left;
	Vector2 moving_handle_right;
	int moving_handle_mode; // value from Animation::HandleMode

	void _clear_selection();
	void _clear_selection_for_anim(const Ref<Animation> &p_anim);
	void _select_at_anim(const Ref<Animation> &p_anim, int p_track, float p_pos);
	void _change_selected_keys_handle_mode(Animation::HandleMode p_mode);

	Vector2 menu_insert_key;

	struct AnimMoveRestore {
		int track = 0;
		float time = 0;
		Variant key;
		float transition = 0;
	};

	AnimationTrackEditor *editor;

	struct EditPoint {
		Rect2 point_rect;
		Rect2 in_rect;
		Rect2 out_rect;
	};

	Vector<EditPoint> edit_points;

	Set<int> selection;

	bool panning_timeline = false;
	float panning_timeline_from;
	float panning_timeline_at;

	void _draw_line_clipped(const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, int p_clip_left, int p_clip_right);
	void _draw_track(int p_track, const Color &p_color);

	float _bezier_h_to_pixel(float p_h);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	virtual String get_tooltip(const Point2 &p_pos) const override;

	Ref<Animation> get_animation() const;

	void set_animation_and_track(const Ref<Animation> &p_animation, int p_track);
	virtual Size2 get_minimum_size() const override;

	void set_undo_redo(UndoRedo *p_undo_redo);
	void set_timeline(AnimationTimelineEdit *p_timeline);
	void set_editor(AnimationTrackEditor *p_editor);
	void set_root(Node *p_root);

	void set_play_position(float p_pos);
	void update_play_position();

	void duplicate_selection();
	void delete_selection();

	AnimationBezierTrackEdit();
};

#endif // ANIMATION_BEZIER_EDITOR_H
