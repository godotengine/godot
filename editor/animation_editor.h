/*************************************************************************/
/*  animation_editor.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef ANIMATION_EDITOR_H
#define ANIMATION_EDITOR_H

#include "scene/gui/control.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"

#include "editor_data.h"
#include "property_editor.h"
#include "scene/animation/animation_cache.h"
#include "scene/resources/animation.h"
#include "scene_tree_editor.h"

class AnimationKeyEdit;
class AnimationCurveEdit;

class AnimationKeyEditor : public VBoxContainer {

	GDCLASS(AnimationKeyEditor, VBoxContainer);

	/*
	enum {

		MENU_NEW_ANIMATION,
		MENU_OPEN_ANIMATION,
		MENU_EDIT_ANIMATION,
		MENU_CLOSE_ANIMATION,
		MENU_KEYING_ACTIVE,
		MENU_SET_ROOT_NODE,
		MENU_SYNC_TO_PLAYER,
		MENU_ANIM_BASE=100,
	};

*/

	enum {

		ADD_TRACK_MENU_ADD_VALUE_TRACK,
		ADD_TRACK_MENU_ADD_TRANSFORM_TRACK,
		ADD_TRACK_MENU_ADD_CALL_TRACK,
		TRACK_MENU_SCALE,
		TRACK_MENU_SCALE_PIVOT,
		TRACK_MENU_MOVE_UP,
		TRACK_MENU_MOVE_DOWN,
		TRACK_MENU_REMOVE,
		TRACK_MENU_DUPLICATE,
		TRACK_MENU_DUPLICATE_TRANSPOSE,
		TRACK_MENU_SET_ALL_TRANS_LINEAR,
		TRACK_MENU_SET_ALL_TRANS_CONSTANT,
		TRACK_MENU_SET_ALL_TRANS_OUT,
		TRACK_MENU_SET_ALL_TRANS_IN,
		TRACK_MENU_SET_ALL_TRANS_INOUT,
		TRACK_MENU_SET_ALL_TRANS_OUTIN,
		TRACK_MENU_NEXT_STEP,
		TRACK_MENU_PREV_STEP,
		TRACK_MENU_OPTIMIZE,
		TRACK_MENU_CLEAN_UP,
		TRACK_MENU_CLEAN_UP_CONFIRM,
		CURVE_SET_LINEAR,
		CURVE_SET_IN,
		CURVE_SET_OUT,
		CURVE_SET_INOUT,
		CURVE_SET_OUTIN,
		CURVE_SET_CONSTANT
	};

	enum {
		RIGHT_MENU_DUPLICATE,
		RIGHT_MENU_DUPLICATE_TRANSPOSE,
		RIGHT_MENU_REMOVE
	};

	struct MouseOver {

		enum Over {
			OVER_NONE,
			OVER_NAME,
			OVER_KEY,
			OVER_VALUE,
			OVER_INTERP,
			OVER_WRAP,
			OVER_UP,
			OVER_DOWN,
			OVER_REMOVE,
			OVER_ADD_KEY,
		};

		Over over;
		int track;
		int over_key;

	} mouse_over;

	struct SelectedKey {

		int track;
		int key;
		bool operator<(const SelectedKey &p_key) const { return track == p_key.track ? key < p_key.key : track < p_key.track; };
	};

	struct KeyInfo {

		float pos;
	};

	Map<SelectedKey, KeyInfo> selection;

	struct ClickOver {

		enum Click {

			CLICK_NONE,
			CLICK_RESIZE_NAMES,
			CLICK_DRAG_TIMELINE,
			CLICK_MOVE_KEYS,
			CLICK_SELECT_KEYS

		};

		SelectedKey selk;
		bool shift;
		Click click;
		Point2 at;
		Point2 to;
	} click;

	float timeline_pos;

	float name_column_ratio;

	int track_name_editing;
	int interp_editing;
	int cont_editing;
	int wrap_editing;
	int selected_track;
	int track_ofs[5];

	int last_menu_track_opt;
	LineEdit *track_name;
	PopupMenu *track_menu;
	PopupMenu *type_menu;

	Control *ec;
	TextureRect *zoomicon;
	HSlider *zoom;
	//MenuButton *menu;
	SpinBox *length;
	Button *loop;
	bool keying;
	ToolButton *edit_button;
	ToolButton *move_up_button;
	ToolButton *move_down_button;
	ToolButton *remove_button;

	ToolButton *curve_linear;
	ToolButton *curve_in;
	ToolButton *curve_out;
	ToolButton *curve_inout;
	ToolButton *curve_outin;
	ToolButton *curve_constant;

	ConfirmationDialog *optimize_dialog;
	SpinBox *optimize_linear_error;
	SpinBox *optimize_angular_error;
	SpinBox *optimize_max_angle;

	ConfirmationDialog *cleanup_dialog;
	CheckButton *cleanup_keys;
	CheckButton *cleanup_tracks;
	CheckButton *cleanup_all;

	SpinBox *step;

	MenuButton *menu_add_track;
	MenuButton *menu_track;

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;

	Control *track_editor;
	Control *track_pos;
	TabContainer *key_editor_tab;

	ConfirmationDialog *scale_dialog;
	SpinBox *scale;

	PropertyEditor *key_editor;

	SceneTreeDialog *call_select;

	Ref<Animation> animation;
	void _update_paths();

	int last_idx;

	Node *root;

	UndoRedo *undo_redo;
	EditorHistory *history;
	ConfirmationDialog *insert_confirm;

	AnimationKeyEdit *key_edit;
	AnimationCurveEdit *curve_edit;

	bool inserting;

	bool updating;
	bool te_drawing;

	void _animation_len_changed(float p_len);
	void _animation_loop_changed();
	void _step_changed(float p_len);

	struct InsertData {

		Animation::TrackType type;
		NodePath path;
		int track_idx;
		Variant value;
		String query;
		bool advance;
	}; /* insert_data;*/

	bool insert_query;
	List<InsertData> insert_data;
	uint64_t insert_frame;

	int cvi_track;
	float cvi_pos;

	int right_data_size_cache;

	EditorSelection *editor_selection;

	Label *select_anim_warning;

	float _get_zoom_scale() const;

	void _track_editor_draw();
	void _track_editor_gui_input(const Ref<InputEvent> &p_input);
	void _track_position_draw();

	void _track_name_changed(const String &p_name);
	void _track_menu_selected(int p_idx);
	void _confirm_insert_list();
	int _confirm_insert(InsertData p_id, int p_last_track = -1);
	void _query_insert(const InsertData &p_id);
	void _update_menu();
	bool insert_queue;
	void _insert_delay();
	void _scale();

	void _clear_selection();

	//void _browse_path();

	StringName alc;

	void _animation_changed();
	void _animation_optimize();
	void _cleanup_animation(Ref<Animation> p_animation);

	void _scroll_changed(double);

	void _menu_add_track(int p_type);
	void _menu_track(int p_type);

	void _clear_selection_for_anim(const Ref<Animation> &p_anim);
	void _select_at_anim(const Ref<Animation> &p_anim, int p_track, float p_pos);
	void _curve_transition_changed(float p_what);

	PropertyInfo _find_hint_for_track(int p_idx, NodePath &r_base_path);

	void _create_value_item(int p_type);
	void _pane_drag(const Point2 &p_delta);
	bool _edit_if_single_selection();

	void _toggle_edit_curves();
	void _animation_len_update();

	void _add_call_track(const NodePath &p_base);

	void _anim_duplicate_keys(bool transpose = false);
	void _anim_delete_keys();

	void _root_removed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_animation(const Ref<Animation> &p_anim);
	Ref<Animation> get_current_animation() const;
	void set_root(Node *p_root);
	Node *get_root() const;
	void update_keying();
	bool has_keying() const;

	void cleanup();

	void set_anim_pos(float p_pos);
	void insert_node_value_key(Node *p_node, const String &p_property, const Variant &p_value, bool p_only_if_exists = false);
	void insert_value_key(const String &p_property, const Variant &p_value, bool p_advance);
	void insert_transform_key(Spatial *p_node, const String &p_sub, const Transform &p_xform);

	void show_select_node_warning(bool p_show) { select_anim_warning->set_visible(p_show); }
	AnimationKeyEditor();
	~AnimationKeyEditor();
};

#endif // ANIMATION_EDITOR_H
