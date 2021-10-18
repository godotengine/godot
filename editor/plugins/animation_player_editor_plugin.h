/*************************************************************************/
/*  animation_player_editor_plugin.h                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ANIMATION_PLAYER_EDITOR_PLUGIN_H
#define ANIMATION_PLAYER_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_button.h"

class AnimationTrackEditor;
class AnimationPlayerEditorPlugin;

class AnimationPlayerEditor : public VBoxContainer {
	GDCLASS(AnimationPlayerEditor, VBoxContainer);

	EditorNode *editor;
	AnimationPlayerEditorPlugin *plugin;
	AnimationPlayer *player;

	enum {
		TOOL_NEW_ANIM,
		TOOL_LOAD_ANIM,
		TOOL_SAVE_ANIM,
		TOOL_SAVE_AS_ANIM,
		TOOL_DUPLICATE_ANIM,
		TOOL_RENAME_ANIM,
		TOOL_EDIT_TRANSITIONS,
		TOOL_REMOVE_ANIM,
		TOOL_COPY_ANIM,
		TOOL_PASTE_ANIM,
		TOOL_EDIT_RESOURCE
	};

	enum {
		ONION_SKINNING_ENABLE,
		ONION_SKINNING_PAST,
		ONION_SKINNING_FUTURE,
		ONION_SKINNING_1_STEP,
		ONION_SKINNING_2_STEPS,
		ONION_SKINNING_3_STEPS,
		ONION_SKINNING_LAST_STEPS_OPTION = ONION_SKINNING_3_STEPS,
		ONION_SKINNING_DIFFERENCES_ONLY,
		ONION_SKINNING_FORCE_WHITE_MODULATE,
		ONION_SKINNING_INCLUDE_GIZMOS,
	};

	enum {
		ANIM_OPEN,
		ANIM_SAVE,
		ANIM_SAVE_AS
	};

	enum {
		RESOURCE_LOAD,
		RESOURCE_SAVE
	};

	OptionButton *animation;
	Button *stop;
	Button *play;
	Button *play_from;
	Button *play_bw;
	Button *play_bw_from;
	Button *autoplay;

	MenuButton *tool_anim;
	ToolButton *onion_toggle;
	MenuButton *onion_skinning;
	ToolButton *pin;
	SpinBox *frame;
	LineEdit *scale;
	LineEdit *name;
	Label *name_title;
	UndoRedo *undo_redo;
	Ref<Texture> autoplay_icon;
	Ref<Texture> reset_icon;
	Ref<Texture> autoplay_reset_icon;
	bool last_active;
	float timeline_position;

	EditorFileDialog *file;
	AcceptDialog *accept;
	ConfirmationDialog *delete_dialog;
	int current_option;

	struct BlendEditor {
		AcceptDialog *dialog;
		Tree *tree;
		OptionButton *next;

	} blend_editor;

	ConfirmationDialog *name_dialog;
	ConfirmationDialog *error_dialog;
	bool renaming;

	bool updating;
	bool updating_blends;

	AnimationTrackEditor *track_editor;

	// Onion skinning.
	struct {
		// Settings.
		bool enabled;
		bool past;
		bool future;
		int steps;
		bool differences_only;
		bool force_white_modulate;
		bool include_gizmos;

		int get_needed_capture_count() const {
			// 'Differences only' needs a capture of the present.
			return (past && future ? 2 * steps : steps) + (differences_only ? 1 : 0);
		}

		// Rendering.
		int64_t last_frame;
		int can_overlay;
		Size2 capture_size;
		Vector<RID> captures;
		Vector<bool> captures_valid;
		struct {
			RID canvas;
			RID canvas_item;
			Ref<ShaderMaterial> material;
			Ref<Shader> shader;
		} capture;
	} onion;

	void _select_anim_by_name(const String &p_anim);
	double _get_editor_step() const;
	void _play_pressed();
	void _play_from_pressed();
	void _play_bw_pressed();
	void _play_bw_from_pressed();
	void _autoplay_pressed();
	void _stop_pressed();
	void _animation_selected(int p_which);
	void _animation_new();
	void _animation_rename();
	void _animation_name_edited();
	void _animation_load();

	void _animation_save_in_path(const Ref<Resource> &p_resource, const String &p_path);
	void _animation_save(const Ref<Resource> &p_resource);
	void _animation_save_as(const Ref<Resource> &p_resource);

	void _animation_remove();
	void _animation_remove_confirmed();
	void _animation_blend();
	void _animation_edit();
	void _animation_duplicate();
	void _animation_resource_edit();
	void _scale_changed(const String &p_scale);
	void _dialog_action(String p_file);
	void _seek_value_changed(float p_value, bool p_set = false);
	void _blend_editor_next_changed(const int p_idx);

	void _list_changed();
	void _update_animation();
	void _update_player();
	void _blend_edited();

	void _animation_player_changed(Object *p_pl);

	void _animation_key_editor_seek(float p_pos, bool p_drag);
	void _animation_key_editor_anim_len_changed(float p_len);

	void _unhandled_key_input(const Ref<InputEvent> &p_ev);
	void _animation_tool_menu(int p_option);
	void _onion_skinning_menu(int p_option);

	void _editor_visibility_changed();
	bool _are_onion_layers_valid();
	void _allocate_onion_layers();
	void _free_onion_layers();
	void _prepare_onion_layers_1();
	void _prepare_onion_layers_2();
	void _start_onion_skinning();
	void _stop_onion_skinning();

	void _pin_pressed();

	~AnimationPlayerEditor();

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

public:
	AnimationPlayer *get_player() const;
	static AnimationPlayerEditor *singleton;

	bool is_pinned() const { return pin->is_pressed(); }
	void unpin() { pin->set_pressed(false); }
	AnimationTrackEditor *get_track_editor() { return track_editor; }
	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	void ensure_visibility();

	void set_undo_redo(UndoRedo *p_undo_redo) { undo_redo = p_undo_redo; }
	void edit(AnimationPlayer *p_player);
	void forward_force_draw_over_viewport(Control *p_overlay);

	AnimationPlayerEditor(EditorNode *p_editor, AnimationPlayerEditorPlugin *p_plugin);
};

class AnimationPlayerEditorPlugin : public EditorPlugin {
	GDCLASS(AnimationPlayerEditorPlugin, EditorPlugin);

	AnimationPlayerEditor *anim_editor;
	EditorNode *editor;

protected:
	void _notification(int p_what);

public:
	virtual Dictionary get_state() const { return anim_editor->get_state(); }
	virtual void set_state(const Dictionary &p_state) { anim_editor->set_state(p_state); }

	virtual String get_name() const { return "Anim"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	virtual void forward_canvas_force_draw_over_viewport(Control *p_overlay) { anim_editor->forward_force_draw_over_viewport(p_overlay); }
	virtual void forward_spatial_force_draw_over_viewport(Control *p_overlay) { anim_editor->forward_force_draw_over_viewport(p_overlay); }

	AnimationPlayerEditorPlugin(EditorNode *p_node);
	~AnimationPlayerEditorPlugin();
};

#endif // ANIMATION_PLAYER_EDITOR_PLUGIN_H
