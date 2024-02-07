/**************************************************************************/
/*  merge_group_editor_plugin.h                                           */
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

#ifndef MERGE_GROUP_EDITOR_PLUGIN_H
#define MERGE_GROUP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/3d/merge_group.h"
#include "scene/resources/material.h"

class MergeGroupEditorPlugin;

class MergeGroupEditorBakeDialog : public ConfirmationDialog {
	GDCLASS(MergeGroupEditorBakeDialog, ConfirmationDialog);

	MergeGroupEditorPlugin *_owner_plugin = nullptr;
	CheckBox *_single_scene = nullptr;
	SpinBox *_subscene_polycount_threshold = nullptr;

	CheckBox *_shadow_proxy = nullptr;
	CheckBox *_convert_csgs = nullptr;
	CheckBox *_convert_gridmaps = nullptr;
	CheckBox *_combine_surfaces = nullptr;
	CheckBox *_clean_meshes = nullptr;
	SpinBox *_group_size = nullptr;
	SpinBox *_splits_horz = nullptr;
	SpinBox *_splits_vert = nullptr;
	SpinBox *_min_split_poly_count = nullptr;

	void _bake_confirm();
	void _add_bake_checkbox(Node *p_parent, CheckBox **pp_checkbox, const String &p_text, const String &p_tooltip, bool p_default = false);
	void _add_bake_spinbox(VBoxContainer *p_parent, SpinBox **pp_spinbox, const String &p_text, const String &p_tooltip, int32_t p_min, int32_t p_max, int32_t p_step, int32_t p_default);

protected:
	static void _bind_methods();

public:
	void fill_merge_group_params(MergeGroup &r_merge_group);
	MergeGroupEditorBakeDialog(MergeGroupEditorPlugin *p_owner);
};

class MergeGroupEditorPlugin : public EditorPlugin {
	GDCLASS(MergeGroupEditorPlugin, EditorPlugin);

	MergeGroup *_merge_group;
	ToolButton *button_bake;
	EditorFileDialog *file_dialog;
	EditorNode *editor;
	MergeGroupEditorBakeDialog *bake_dialog;

	struct Params {
		bool single_scene = false;
		int subscene_polycount_threshold = 1024;
	} _params;

	static EditorProgress *tmp_progress;
	static EditorProgress *tmp_subprogress;

	static bool bake_func_step(float p_progress, const String &p_description, void *, bool p_force_refresh);
	static bool bake_func_substep(float p_progress, const String &p_description, void *, bool p_force_refresh);
	static void bake_func_end(uint32_t p_time_started);

	void _bake();
	void _bake_select_file(const String &p_file);
	void _duplicate_branch(Node *p_branch, Node *p_new_parent);
	bool _save_scene(Node *p_branch, String p_filename);
	void _remove_queue_deleted_nodes_recursive(Node *p_node);
	void _push_mesh_data_to_gpu_recursive(Node *p_node);
	Spatial *_convert_merge_group_to_spatial(MergeGroup *p_merge_group);
	bool _find_visual_instances_recursive(Node *p_node);

	void _save_mesh_subscenes_recursive(Node *p_root, Node *p_node, String p_base_filename, int &r_subscene_count);
	bool _save_subscene(Node *p_root, Node *p_branch, String p_base_filename, int &r_subscene_count);
	bool _replace_with_branch_scene(const String &p_file, Node *p_base);
	uint32_t _get_mesh_poly_count(const MeshInstance &p_mi) const;

protected:
	static void _bind_methods();

public:
	virtual String get_name() const { return "MergeGroup"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	void dialog_pressed_bake(bool p_single_scene, int p_subscene_polycount_threshold);

	MergeGroupEditorPlugin(EditorNode *p_node);
	~MergeGroupEditorPlugin();
};

#endif // MERGE_GROUP_EDITOR_PLUGIN_H
