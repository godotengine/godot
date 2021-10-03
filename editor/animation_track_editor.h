/*************************************************************************/
/*  animation_track_editor.h                                             */
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

#ifndef ANIMATION_TRACK_EDITOR_H
#define ANIMATION_TRACK_EDITOR_H

#include "editor/editor_data.h"
#include "editor/editor_spin_slider.h"
#include "editor/property_editor.h"
#include "editor/property_selector.h"
#include "scene/animation/animation_cache.h"
#include "scene/gui/control.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"
#include "scene/resources/animation.h"
#include "scene_tree_editor.h"

class AnimationTrackEdit;

class AnimationTimelineEdit : public Range {
	GDCLASS(AnimationTimelineEdit, Range);

	Ref<Animation> animation;
	AnimationTrackEdit *track_edit;
	int name_limit;
	Range *zoom;
	Range *h_scroll;
	float play_position_pos;

	HBoxContainer *len_hb;
	EditorSpinSlider *length;
	ToolButton *loop;
	TextureRect *time_icon;

	MenuButton *add_track;
	Control *play_position; //separate control used to draw so updates for only position changed are much faster
	HScrollBar *hscroll;

	void _zoom_changed(double);
	void _anim_length_changed(double p_new_len);
	void _anim_loop_pressed();

	void _play_position_draw();
	UndoRedo *undo_redo;
	Rect2 hsize_rect;

	bool editing;
	bool use_fps;

	bool panning_timeline;
	float panning_timeline_from;
	float panning_timeline_at;
	bool dragging_timeline;
	bool dragging_hsize;
	float dragging_hsize_from;
	float dragging_hsize_at;

	void _gui_input(const Ref<InputEvent> &p_event);
	void _track_added(int p_track);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	int get_name_limit() const;
	int get_buttons_width() const;

	float get_zoom_scale() const;

	virtual Size2 get_minimum_size() const;
	void set_animation(const Ref<Animation> &p_animation);
	void set_track_edit(AnimationTrackEdit *p_track_edit);
	void set_zoom(Range *p_zoom);
	Range *get_zoom() const { return zoom; }
	void set_undo_redo(UndoRedo *p_undo_redo);

	void set_play_position(float p_pos);
	float get_play_position() const;
	void update_play_position();

	void update_values();

	void set_use_fps(bool p_use_fps);
	bool is_using_fps() const;

	void set_hscroll(HScrollBar *p_hscroll);

	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const;

	AnimationTimelineEdit();
};

class AnimationTrackEditor;

class AnimationTrackEdit : public Control {
	GDCLASS(AnimationTrackEdit, Control);

	enum {
		MENU_CALL_MODE_CONTINUOUS,
		MENU_CALL_MODE_DISCRETE,
		MENU_CALL_MODE_TRIGGER,
		MENU_CALL_MODE_CAPTURE,
		MENU_INTERPOLATION_NEAREST,
		MENU_INTERPOLATION_LINEAR,
		MENU_INTERPOLATION_CUBIC,
		MENU_LOOP_WRAP,
		MENU_LOOP_CLAMP,
		MENU_KEY_INSERT,
		MENU_KEY_DUPLICATE,
		MENU_KEY_ADD_RESET,
		MENU_KEY_DELETE
	};
	AnimationTimelineEdit *timeline;
	UndoRedo *undo_redo;
	LineEdit *path;
	Node *root;
	Control *play_position; //separate control used to draw so updates for only position changed are much faster
	float play_position_pos;
	NodePath node_path;

	Ref<Animation> animation;
	int track;

	Rect2 check_rect;
	Rect2 path_rect;

	Rect2 update_mode_rect;
	Rect2 interp_mode_rect;
	Rect2 loop_mode_rect;
	Rect2 remove_rect;
	Rect2 bezier_edit_rect;

	Ref<Texture> type_icon;
	Ref<Texture> selected_icon;

	PopupMenu *menu;

	bool clicking_on_name;

	void _zoom_changed();

	Ref<Texture> icon_cache;
	String path_cache;

	void _menu_selected(int p_index);

	void _path_entered(const String &p_text);
	void _play_position_draw();
	bool _is_value_key_valid(const Variant &p_key_value, Variant::Type &r_valid_type) const;

	Ref<Texture> _get_key_type_icon() const;

	mutable int dropping_at;
	float insert_at_pos;
	bool moving_selection_attempt;
	int select_single_attempt;
	bool moving_selection;
	float moving_selection_from_ofs;

	bool in_group;
	AnimationTrackEditor *editor;

protected:
	static void _bind_methods();
	void _notification(int p_what);

	virtual void _gui_input(const Ref<InputEvent> &p_event);

public:
	virtual Variant get_drag_data(const Point2 &p_point);
	virtual bool can_drop_data(const Point2 &p_point, const Variant &p_data) const;
	virtual void drop_data(const Point2 &p_point, const Variant &p_data);

	virtual String get_tooltip(const Point2 &p_pos) const;

	virtual int get_key_height() const;
	virtual Rect2 get_key_rect(int p_index, float p_pixels_sec);
	virtual bool is_key_selectable_by_distance() const;
	virtual void draw_key_link(int p_index, float p_pixels_sec, int p_x, int p_next_x, int p_clip_left, int p_clip_right);
	virtual void draw_key(int p_index, float p_pixels_sec, int p_x, bool p_selected, int p_clip_left, int p_clip_right);
	virtual void draw_bg(int p_clip_left, int p_clip_right);
	virtual void draw_fg(int p_clip_left, int p_clip_right);

	//helper
	void draw_texture_clipped(const Ref<Texture> &p_texture, const Vector2 &p_pos);
	void draw_texture_region_clipped(const Ref<Texture> &p_texture, const Rect2 &p_rect, const Rect2 &p_region);
	void draw_rect_clipped(const Rect2 &p_rect, const Color &p_color, bool p_filled = true);

	int get_track() const;
	Ref<Animation> get_animation() const;
	AnimationTimelineEdit *get_timeline() const { return timeline; }
	AnimationTrackEditor *get_editor() const { return editor; }
	UndoRedo *get_undo_redo() const { return undo_redo; }
	NodePath get_path() const;
	void set_animation_and_track(const Ref<Animation> &p_animation, int p_track);
	virtual Size2 get_minimum_size() const;

	void set_undo_redo(UndoRedo *p_undo_redo);
	void set_timeline(AnimationTimelineEdit *p_timeline);
	void set_editor(AnimationTrackEditor *p_editor);
	void set_root(Node *p_root);

	void set_play_position(float p_pos);
	void update_play_position();
	void cancel_drop();

	void set_in_group(bool p_enable);
	void append_to_selection(const Rect2 &p_box, bool p_deselection);

	AnimationTrackEdit();
};

class AnimationTrackEditPlugin : public Reference {
	GDCLASS(AnimationTrackEditPlugin, Reference);

public:
	virtual AnimationTrackEdit *create_value_track_edit(Object *p_object, Variant::Type p_type, const String &p_property, PropertyHint p_hint, const String &p_hint_string, int p_usage);
	virtual AnimationTrackEdit *create_audio_track_edit();
	virtual AnimationTrackEdit *create_animation_track_edit(Object *p_object);
};

class AnimationTrackKeyEdit;
class AnimationMultiTrackKeyEdit;
class AnimationBezierTrackEdit;

class AnimationTrackEditGroup : public Control {
	GDCLASS(AnimationTrackEditGroup, Control);
	Ref<Texture> icon;
	String node_name;
	NodePath node;
	Node *root;
	AnimationTimelineEdit *timeline;

	void _zoom_changed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_type_and_name(const Ref<Texture> &p_type, const String &p_name, const NodePath &p_node);
	virtual Size2 get_minimum_size() const;
	void set_timeline(AnimationTimelineEdit *p_timeline);
	void set_root(Node *p_root);

	AnimationTrackEditGroup();
};

class AnimationTrackEditor : public VBoxContainer {
	GDCLASS(AnimationTrackEditor, VBoxContainer);

	Ref<Animation> animation;
	Node *root;

	MenuButton *edit;

	PanelContainer *main_panel;
	HScrollBar *hscroll;
	ScrollContainer *scroll;
	VBoxContainer *track_vbox;
	AnimationBezierTrackEdit *bezier_edit;

	Label *info_message;

	AnimationTimelineEdit *timeline;
	HSlider *zoom;
	EditorSpinSlider *step;
	TextureRect *zoom_icon;
	ToolButton *snap;
	OptionButton *snap_mode;

	Button *imported_anim_warning;
	void _show_imported_anim_warning() const;

	void _snap_mode_changed(int p_mode);
	Vector<AnimationTrackEdit *> track_edits;
	Vector<AnimationTrackEditGroup *> groups;

	bool animation_changing_awaiting_update;
	void _animation_update();
	int _get_track_selected();
	void _animation_changed();
	void _update_tracks();

	void _name_limit_changed();
	void _timeline_changed(float p_new_pos, bool p_drag);
	void _track_remove_request(int p_track);
	void _track_grab_focus(int p_track);

	UndoRedo *undo_redo;

	void _update_scroll(double);
	void _update_step(double p_new_step);
	void _update_length(double p_new_len);
	void _dropped_track(int p_from_track, int p_to_track);

	void _add_track(int p_type);
	void _new_track_node_selected(NodePath p_path);
	void _new_track_property_selected(String p_name);

	void _update_step_spinbox();

	PropertySelector *prop_selector;
	PropertySelector *method_selector;
	SceneTreeDialog *pick_track;
	int adding_track_type;
	NodePath adding_track_path;

	bool keying;

	struct InsertData {
		Animation::TrackType type;
		NodePath path;
		int track_idx;
		Variant value;
		String query;
		bool advance;
	}; /* insert_data;*/

	Label *insert_confirm_text;
	CheckBox *insert_confirm_bezier;
	CheckBox *insert_confirm_reset;
	ConfirmationDialog *insert_confirm;
	bool insert_queue;
	bool inserting;
	bool insert_query;
	List<InsertData> insert_data;
	uint64_t insert_frame;

	void _query_insert(const InsertData &p_id);
	Ref<Animation> _create_and_get_reset_animation();
	void _confirm_insert_list();
	struct TrackIndices {
		int normal;
		int reset;

		TrackIndices(const Animation *p_anim = nullptr, const Animation *p_reset_anim = nullptr) {
			normal = p_anim ? p_anim->get_track_count() : 0;
			reset = p_reset_anim ? p_reset_anim->get_track_count() : 0;
		}
	};
	TrackIndices _confirm_insert(InsertData p_id, TrackIndices p_next_tracks, bool p_create_reset, Ref<Animation> p_reset_anim, bool p_create_beziers);
	void _insert_delay(bool p_create_reset, bool p_create_beziers);

	void _root_removed();

	PropertyInfo _find_hint_for_track(int p_idx, NodePath &r_base_path, Variant *r_current_val = nullptr);

	void _timeline_value_changed(double);

	float insert_key_from_track_call_ofs;
	int insert_key_from_track_call_track;
	void _insert_key_from_track(float p_ofs, int p_track);
	void _add_method_key(const String &p_method);

	void _clear_selection(bool p_update = false);
	void _clear_selection_for_anim(const Ref<Animation> &p_anim);
	void _select_at_anim(const Ref<Animation> &p_anim, int p_track, float p_pos);

	//selection

	struct SelectedKey {
		int track;
		int key;
		bool operator<(const SelectedKey &p_key) const { return track == p_key.track ? key < p_key.key : track < p_key.track; };
	};

	struct KeyInfo {
		float pos;
	};

	Map<SelectedKey, KeyInfo> selection;

	void _key_selected(int p_key, bool p_single, int p_track);
	void _key_deselected(int p_key, int p_track);

	bool moving_selection;
	float moving_selection_offset;
	void _move_selection_begin();
	void _move_selection(float p_offset);
	void _move_selection_commit();
	void _move_selection_cancel();

	AnimationTrackKeyEdit *key_edit;
	AnimationMultiTrackKeyEdit *multi_key_edit;
	void _update_key_edit();

	void _clear_key_edit();

	Control *box_selection;
	void _box_selection_draw();
	bool box_selecting;
	Vector2 box_selecting_from;
	Rect2 box_select_rect;
	void _scroll_input(const Ref<InputEvent> &p_event);

	Vector<Ref<AnimationTrackEditPlugin>> track_edit_plugins;

	void _cancel_bezier_edit();
	void _bezier_edit(int p_for_track);

	////////////// edit menu stuff

	ConfirmationDialog *optimize_dialog;
	SpinBox *optimize_linear_error;
	SpinBox *optimize_angular_error;
	SpinBox *optimize_max_angle;

	ConfirmationDialog *cleanup_dialog;
	CheckBox *cleanup_keys;
	CheckBox *cleanup_tracks;
	CheckBox *cleanup_all;

	ConfirmationDialog *scale_dialog;
	SpinBox *scale;

	void _select_all_tracks_for_copy();

	void _edit_menu_about_to_show();
	void _edit_menu_pressed(int p_option);
	int last_menu_track_opt;

	void _cleanup_animation(Ref<Animation> p_animation);

	void _anim_duplicate_keys(bool transpose);

	void _view_group_toggle();
	ToolButton *view_group;
	ToolButton *selected_filter;

	void _selection_changed();

	ConfirmationDialog *track_copy_dialog;
	Tree *track_copy_select;

	struct TrackClipboard {
		NodePath full_path;
		NodePath base_path;
		Animation::TrackType track_type;
		Animation::InterpolationType interp_type;
		Animation::UpdateMode update_mode;
		bool loop_wrap;
		bool enabled;

		struct Key {
			float time;
			float transition;
			Variant value;
		};
		Vector<Key> keys;
	};

	Vector<TrackClipboard> track_clipboard;

	void _insert_animation_key(NodePath p_path, const Variant &p_value);

	void _pick_track_filter_text_changed(const String &p_text);
	void _pick_track_select_recursive(TreeItem *p_item, const String &p_filter, Vector<Node *> &p_select_candidates);
	void _pick_track_filter_input(const Ref<InputEvent> &p_ie);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	enum {
		EDIT_COPY_TRACKS,
		EDIT_COPY_TRACKS_CONFIRM,
		EDIT_PASTE_TRACKS,
		EDIT_SCALE_SELECTION,
		EDIT_SCALE_FROM_CURSOR,
		EDIT_SCALE_CONFIRM,
		EDIT_DUPLICATE_SELECTION,
		EDIT_DUPLICATE_TRANSPOSED,
		EDIT_ADD_RESET_KEY,
		EDIT_DELETE_SELECTION,
		EDIT_GOTO_NEXT_STEP,
		EDIT_GOTO_PREV_STEP,
		EDIT_APPLY_RESET,
		EDIT_OPTIMIZE_ANIMATION,
		EDIT_OPTIMIZE_ANIMATION_CONFIRM,
		EDIT_CLEAN_UP_ANIMATION,
		EDIT_CLEAN_UP_ANIMATION_CONFIRM
	};

	void add_track_edit_plugin(const Ref<AnimationTrackEditPlugin> &p_plugin);
	void remove_track_edit_plugin(const Ref<AnimationTrackEditPlugin> &p_plugin);

	void set_animation(const Ref<Animation> &p_anim);
	Ref<Animation> get_current_animation() const;
	void set_root(Node *p_root);
	Node *get_root() const;
	void update_keying();
	bool has_keying() const;

	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	void cleanup();

	void set_anim_pos(float p_pos);
	void insert_node_value_key(Node *p_node, const String &p_property, const Variant &p_value, bool p_only_if_exists = false);
	void insert_value_key(const String &p_property, const Variant &p_value, bool p_advance);
	void insert_transform_key(Spatial *p_node, const String &p_sub, const Transform &p_xform);

	void show_select_node_warning(bool p_show);

	bool is_key_selected(int p_track, int p_key) const;
	bool is_selection_active() const;
	bool is_moving_selection() const;
	bool is_snap_enabled() const;
	float get_moving_selection_offset() const;
	float snap_time(float p_value, bool p_relative = false);
	bool is_grouping_tracks();

	/** If `p_from_mouse_event` is `true`, handle Shift key presses for precise snapping. */
	void goto_prev_step(bool p_from_mouse_event);

	/** If `p_from_mouse_event` is `true`, handle Shift key presses for precise snapping. */
	void goto_next_step(bool p_from_mouse_event);

	MenuButton *get_edit_menu();
	AnimationTrackEditor();
	~AnimationTrackEditor();
};

#endif // ANIMATION_TRACK_EDITOR_H
