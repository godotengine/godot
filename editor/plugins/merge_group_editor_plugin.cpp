/**************************************************************************/
/*  merge_group_editor_plugin.cpp                                         */
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

#include "merge_group_editor_plugin.h"

#include "core/io/resource_saver.h"
#include "editor/spatial_editor_gizmos.h"
#include "scene/3d/mesh_instance.h"
#include "scene/resources/merging_tool.h"
#include "scene/resources/packed_scene.h"

EditorProgress *MergeGroupEditorPlugin::tmp_progress = nullptr;
EditorProgress *MergeGroupEditorPlugin::tmp_subprogress = nullptr;

void MergeGroupEditorBakeDialog::_bake_confirm() {
	_owner_plugin->dialog_pressed_bake(_single_scene->is_pressed(), (int)_subscene_polycount_threshold->get_value());
}

void MergeGroupEditorBakeDialog::_add_bake_checkbox(Node *p_parent, CheckBox **pp_checkbox, const String &p_text, const String &p_tooltip, bool p_default) {
	*pp_checkbox = memnew(CheckBox);
	(*pp_checkbox)->set_text(TTR(p_text));
	(*pp_checkbox)->set_tooltip(TTR(p_tooltip));
	(*pp_checkbox)->set_pressed(p_default);
	p_parent->add_child(*pp_checkbox);
}

void MergeGroupEditorBakeDialog::_add_bake_spinbox(VBoxContainer *p_parent, SpinBox **pp_spinbox, const String &p_text, const String &p_tooltip, int32_t p_min, int32_t p_max, int32_t p_step, int32_t p_default) {
	*pp_spinbox = memnew(SpinBox);
	(*pp_spinbox)->set_min(p_min);
	(*pp_spinbox)->set_max(p_max);
	(*pp_spinbox)->set_step(p_step);
	(*pp_spinbox)->set_value(p_default);
	(*pp_spinbox)->set_tooltip(p_tooltip);
	p_parent->add_margin_child(TTR(p_text), *pp_spinbox);
}

void MergeGroupEditorBakeDialog::fill_merge_group_params(MergeGroup &r_merge_group) {
	r_merge_group.set_param_enabled(MergeGroup::PARAM_ENABLED_SHADOW_PROXY, _shadow_proxy->is_pressed());
	r_merge_group.set_param_enabled(MergeGroup::PARAM_ENABLED_CONVERT_CSGS, _convert_csgs->is_pressed());
	r_merge_group.set_param_enabled(MergeGroup::PARAM_ENABLED_CONVERT_GRIDMAPS, _convert_gridmaps->is_pressed());
	r_merge_group.set_param_enabled(MergeGroup::PARAM_ENABLED_COMBINE_SURFACES, _combine_surfaces->is_pressed());
	r_merge_group.set_param_enabled(MergeGroup::PARAM_ENABLED_CLEAN_MESHES, _clean_meshes->is_pressed());

	r_merge_group.set_param(MergeGroup::PARAM_GROUP_SIZE, _group_size->get_value());
	r_merge_group.set_param(MergeGroup::PARAM_SPLITS_HORIZONTAL, _splits_horz->get_value());
	r_merge_group.set_param(MergeGroup::PARAM_SPLITS_VERTICAL, _splits_vert->get_value());
	r_merge_group.set_param(MergeGroup::PARAM_MIN_SPLIT_POLY_COUNT, _min_split_poly_count->get_value());
}

MergeGroupEditorBakeDialog::MergeGroupEditorBakeDialog(MergeGroupEditorPlugin *p_owner) {
	_owner_plugin = p_owner;

	set_title("Bake MergeGroup");

	get_ok()->connect("pressed", this, "_bake_confirm");

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	_add_bake_checkbox(vbc, &_single_scene, "Single Scene", "Save meshes as a single scene or separate scenes.", true);

	_add_bake_spinbox(vbc, &_subscene_polycount_threshold, "Subscene Polycount Threshold:", "Threshold polycount to split scenes into separate subscenes.", 0, 1024 * 128, 64, 1024);

	_add_bake_checkbox(vbc, &_shadow_proxy, "Shadow Proxy", "Separate meshes for shadow rendering.", true);
	_add_bake_checkbox(vbc, &_convert_csgs, "Convert CSGs", "Convert CSGs to meshes so they can be merged.", true);
	_add_bake_checkbox(vbc, &_convert_gridmaps, "Convert GridMaps", "Convert GridMaps to meshes so they can be merged.");
	_add_bake_checkbox(vbc, &_combine_surfaces, "Combine Surfaces", "Combine merged surfaces to form an \"ubermesh\".", true);
	_add_bake_checkbox(vbc, &_clean_meshes, "Clean Meshes", "Clean geometry data, remove degenerate triangles.");

	_add_bake_spinbox(vbc, &_group_size, "Group Size", "When non-zero, only local groups of the corresponding number of meshes will be merged.", 0, 128, 1, 0);
	_add_bake_spinbox(vbc, &_splits_horz, "Splits Horizontal", "When above 1, the final meshes will be split into a grid on the horizontal axis.", 0, 16, 1, 1);
	_add_bake_spinbox(vbc, &_splits_vert, "Splits Vertical", "When above 1, the final meshes will be split into a grid on the vertical axis.", 0, 16, 1, 1);
	_add_bake_spinbox(vbc, &_min_split_poly_count, "Min Split Polycount", "When splitting by grid, only meshes above this minimum polycount will be split.", 0, 1024 * 128, 256, 1024);
}

void MergeGroupEditorBakeDialog::_bind_methods() {
	ClassDB::bind_method("_bake_confirm", &MergeGroupEditorBakeDialog::_bake_confirm);
}

//////////////////////////////////////////////////////////

bool MergeGroupEditorPlugin::bake_func_step(float p_progress, const String &p_description, void *, bool p_force_refresh) {
	if (!tmp_progress) {
		tmp_progress = memnew(EditorProgress("bake_merge_group", TTR("Bake MergeGroup"), 1000, true));
		ERR_FAIL_NULL_V(tmp_progress, false);
	}
	return tmp_progress->step(p_description, p_progress * 1000, p_force_refresh);
}

bool MergeGroupEditorPlugin::bake_func_substep(float p_progress, const String &p_description, void *, bool p_force_refresh) {
	if (!tmp_subprogress) {
		tmp_subprogress = memnew(EditorProgress("bake_merge_group_substep", "", 1000, true));
		ERR_FAIL_NULL_V(tmp_subprogress, false);
	}
	return tmp_subprogress->step(p_description, p_progress * 1000, p_force_refresh);
}

void MergeGroupEditorPlugin::bake_func_end(uint32_t p_time_started) {
	if (tmp_progress != nullptr) {
		memdelete(tmp_progress);
		tmp_progress = nullptr;
	}

	if (tmp_subprogress != nullptr) {
		memdelete(tmp_subprogress);
		tmp_subprogress = nullptr;
	}

	const int time_taken = (OS::get_singleton()->get_ticks_msec() - p_time_started) * 0.001;
	if (time_taken >= 1) {
		// Only print a message and request attention if baking took at least 1 second.
		print_line(vformat("Done baking MergeGroup in %02d:%02d:%02d.", time_taken / 3600, (time_taken % 3600) / 60, time_taken % 60));

		// Request attention in case the user was doing something else.
		OS::get_singleton()->request_attention();
	}
}

void MergeGroupEditorPlugin::dialog_pressed_bake(bool p_single_scene, int p_subscene_polycount_threshold) {
	if (!_merge_group) {
		return;
	}

	_params.single_scene = p_single_scene;
	_params.subscene_polycount_threshold = p_subscene_polycount_threshold;

	Node *root = _merge_group->get_tree()->get_edited_scene_root();

	if (root == _merge_group) {
		EditorNode::get_singleton()->show_warning(TTR("Cannot bake MergeGroup when it is scene root."));
		return;
	}

	String scene_path = _merge_group->get_filename();
	if (scene_path == String()) {
		scene_path = _merge_group->get_owner()->get_filename();
	}
	if (scene_path == String()) {
		EditorNode::get_singleton()->show_warning(TTR("Can't determine a save path for merge group.\nSave your scene and try again."));
		return;
	}
	scene_path = scene_path.get_basename() + ".tscn";

	file_dialog->set_current_path(scene_path);
	file_dialog->popup_centered_ratio();
}

bool MergeGroupEditorPlugin::_find_visual_instances_recursive(Node *p_node) {
	if (Object::cast_to<VisualInstance>(p_node)) {
		return true;
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		if (_find_visual_instances_recursive(p_node->get_child(n))) {
			return true;
		}
	}

	return false;
}

void MergeGroupEditorPlugin::_bake() {
	ERR_FAIL_NULL(_merge_group);

	// If the merge group does not contain any VisualInstance children, flag an error.
	if (!_find_visual_instances_recursive(_merge_group)) {
		EditorNode::get_singleton()->show_warning(TTR("MergeGroup does not contain any VisualInstances.\nCannot Bake."));
		return;
	}

	bake_dialog->show();
}

Spatial *MergeGroupEditorPlugin::_convert_merge_group_to_spatial(MergeGroup *p_merge_group) {
	ERR_FAIL_NULL_V(p_merge_group, nullptr);
	Node *parent = p_merge_group->get_parent();
	ERR_FAIL_NULL_V(parent, nullptr);

	Spatial *spatial = memnew(Spatial);
	parent->add_child(spatial);

	// They can't have the same name at the same time.
	String name = p_merge_group->get_name();
	p_merge_group->set_name(name + "_temp");
	spatial->set_name(name);

	// Identical transforms.
	spatial->set_transform(p_merge_group->get_transform());

	// Move the children.
	// GODOT is abysmally bad at moving children in order unfortunately.
	// So reverse order for now.
	for (int n = p_merge_group->get_child_count() - 1; n >= 0; n--) {
		Node *child = p_merge_group->get_child(n);
		p_merge_group->remove_child(child);
		spatial->add_child(child);
	}

	// Change owners.
	MergingTool::_invalidate_owner_recursive(spatial, nullptr, p_merge_group->get_owner());

	// Delete AND detach the merge group from the tree.
	p_merge_group->_delete_node(p_merge_group);

	return spatial;
}

void MergeGroupEditorPlugin::_bake_select_file(const String &p_file) {
	if (!_merge_group) {
		return;
	}

	// Special treatment for scene root.
	Node *root = _merge_group->get_tree()->get_edited_scene_root();

	// Cannot bake scene root.
	if (root == _merge_group) {
		EditorNode::get_singleton()->show_warning(TTR("Cannot bake scene root.\nPlease move to a branch before baking."));
		ERR_FAIL_COND(root == _merge_group);
	}

	Node *parent = _merge_group->get_parent();
	ERR_FAIL_NULL(parent);

	// Disallow saving to the same scene as the root scene
	// (this is usually user error), and prevents losing work.
	if (root->get_filename() == p_file) {
		EditorNode::get_singleton()->show_warning(TTR("Cannot save to the currently edited scene.\nPlease save to a different scene."));
		ERR_FAIL_COND(root->get_filename() == p_file);
	}

	// Ensure to reset this when exiting this routine!
	// Spatial gizmos, especially for meshes are very expensive
	// in terms of RAM and performance, and are totally
	// unnecessary for temporary objects
	SpatialEditor::_prevent_gizmo_generation = true;

#ifdef GODOT_MERGING_VERBOSE
	MergingTool::debug_branch(_merge_group, "START_SCENE");
#endif

	Spatial *hanger = memnew(Spatial);
	hanger->set_name("hanger");
	parent->add_child(hanger);
	hanger->set_owner(_merge_group->get_owner());

	uint32_t time_start = OS::get_singleton()->get_ticks_msec();
	bake_func_step(0.0, "Duplicating Branch", nullptr, true);

	_duplicate_branch(_merge_group, hanger);

	// Temporarily hide source branch, to speed things up in the editor.
	bool was_visible = _merge_group->is_visible_in_tree();
	_merge_group->hide();

	MergeGroup *merge_group_copy = Object::cast_to<MergeGroup>(hanger->get_child(0));

	// Set the parameters in the copy mergegroup to those set up in the bake dialog.
	bake_dialog->fill_merge_group_params(*merge_group_copy);

	if (merge_group_copy->merge_meshes_in_editor()) {
		if (!bake_func_step(1.0, "Saving Scene", nullptr, true)) {
			// Convert the merge node to a spatial..
			// Once baked we don't want baked scenes to be merged AGAIN
			// when incorporated into scenes.
			Spatial *final_branch = _convert_merge_group_to_spatial(merge_group_copy);

			// Only save if not cancelled by user.
			_save_scene(final_branch, p_file);
		}
	}

#ifdef GODOT_MERGING_VERBOSE
	MergingTool::debug_branch(hanger, "END_SCENE");
#endif

	// Finished.
	hanger->queue_delete();
	_merge_group->set_visible(was_visible);

	SpatialEditor::_prevent_gizmo_generation = false;

	bake_func_end(time_start);
}

void MergeGroupEditorPlugin::_remove_queue_deleted_nodes_recursive(Node *p_node) {
	if (p_node->is_queued_for_deletion()) {
		p_node->get_parent()->remove_child(p_node);
		return;
	}

	for (int n = p_node->get_child_count() - 1; n >= 0; n--) {
		_remove_queue_deleted_nodes_recursive(p_node->get_child(n));
	}
}

uint32_t MergeGroupEditorPlugin::_get_mesh_poly_count(const MeshInstance &p_mi) const {
	Ref<Mesh> rmesh = p_mi.get_mesh();
	if (rmesh.is_valid()) {
		return rmesh->get_triangle_count();
	}

	return 0;
}

bool MergeGroupEditorPlugin::_replace_with_branch_scene(const String &p_file, Node *p_base) {
	Node *old_owner = p_base->get_owner();

	Ref<PackedScene> sdata = ResourceLoader::load(p_file);
	if (!sdata.is_valid()) {
		ERR_PRINT("Error loading scene from \"" + p_file + "\".");
		return false;
	}

	Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
	if (!instanced_scene) {
		ERR_PRINT("Error instancing scene from \"" + p_file + "\".");
		return false;
	}

	Node *parent = p_base->get_parent();
	int pos = p_base->get_index();

	parent->remove_child(p_base);
	parent->add_child(instanced_scene);
	parent->move_child(instanced_scene, pos);

	List<Node *> owned;
	p_base->get_owned_by(p_base->get_owner(), &owned);
	Array owners;
	for (List<Node *>::Element *F = owned.front(); F; F = F->next()) {
		owners.push_back(F->get());
	}

	instanced_scene->set_owner(old_owner);

	p_base->queue_delete();

	return true;
}

bool MergeGroupEditorPlugin::_save_subscene(Node *p_root, Node *p_branch, String p_base_filename, int &r_subscene_count) {
	bake_func_substep(0.0, p_branch->get_name(), nullptr, false);

	Node *scene_root = p_root;

	Map<Node *, Node *> reown;
	reown[scene_root] = p_branch;

	Node *copy = p_branch->duplicate_and_reown(reown);

	bake_func_substep(0.2, p_branch->get_name(), nullptr, false);

	if (copy) {
		Ref<PackedScene> sdata = memnew(PackedScene);
		Error err = sdata->pack(copy);
		memdelete(copy);

		bake_func_substep(0.4, p_branch->get_name(), nullptr, false);

		if (err != OK) {
			WARN_PRINT("Couldn't save subscene \"" + p_branch->get_name() + "\" . Likely dependencies (instances) couldn't be satisfied. Saving as part of main scene instead.");
			return false;
		}

		String filename = p_base_filename + "_" + itos(r_subscene_count++) + ".scn";

#ifdef DEV_ENABLED
		print_verbose("Save subscene: " + filename);
#endif

		err = ResourceSaver::save(filename, sdata, ResourceSaver::FLAG_COMPRESS);

		bake_func_substep(0.6, p_branch->get_name(), nullptr, false);

		if (err != OK) {
			WARN_PRINT("Error saving subscene \"" + p_branch->get_name() + "\" , saving as part of main scene instead.");
			return false;
		}
		_replace_with_branch_scene(filename, p_branch);
	} else {
		WARN_PRINT("Error duplicating subscene \"" + p_branch->get_name() + "\" , saving as part of main scene instead.");
		return false;
	}

	return true;
}

void MergeGroupEditorPlugin::_save_mesh_subscenes_recursive(Node *p_root, Node *p_node, String p_base_filename, int &r_subscene_count) {
	if (p_node->is_queued_for_deletion()) {
		return;
	}
	// Is a subscene already?
	if (p_node->get_filename().length() && (p_node != p_root)) {
		return;
	}

	// Is it a mesh instance?
	MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

	// Don't save subscenes for trivially sized meshes.
	if (mi && (!_params.subscene_polycount_threshold || ((int)_get_mesh_poly_count(*mi) > _params.subscene_polycount_threshold))) {
		// Save as subscene.
		if (_save_subscene(p_root, p_node, p_base_filename, r_subscene_count)) {
			return;
		}
	}

	// Replaced subscenes will be added to the last child, so going in reverse order is necessary.
	for (int n = p_node->get_child_count() - 1; n >= 0; n--) {
		_save_mesh_subscenes_recursive(p_root, p_node->get_child(n), p_base_filename, r_subscene_count);
	}
}

void MergeGroupEditorPlugin::_push_mesh_data_to_gpu_recursive(Node *p_node) {
	// Is it a mesh instance?
	MeshInstance *mi = Object::cast_to<MeshInstance>(p_node);

	if (mi) {
		Ref<Mesh> rmesh = mi->get_mesh();
		if (rmesh.is_valid()) {
			rmesh->set_storage_mode(Mesh::STORAGE_MODE_GPU);
		}
	}

	for (int n = 0; n < p_node->get_child_count(); n++) {
		_push_mesh_data_to_gpu_recursive(p_node->get_child(n));
	}
}

bool MergeGroupEditorPlugin::_save_scene(Node *p_branch, String p_filename) {
	// For some reason the saving machinery doesn't deal well with nodes queued for deletion,
	// so we will remove them from the scene tree (as risk of leaks, but the queue delete machinery
	// should still work when detached).
	_remove_queue_deleted_nodes_recursive(p_branch);

	// All mesh data must be on the GPU for the Mesh saving routines to work.
	_push_mesh_data_to_gpu_recursive(p_branch);

	Node *scene_root = p_branch->get_tree()->get_edited_scene_root();

	Map<Node *, Node *> reown;
	reown[scene_root] = p_branch;

	Node *copy = p_branch->duplicate_and_reown(reown);

#ifdef GODOT_MERGING_VERBOSE
	MergingTool::debug_branch(copy, "SAVE SCENE:");
#endif

	bake_func_substep(0.0, p_filename, nullptr, false);

	if (copy) {
		// Save any large meshes as compressed resources.
		if (!_params.single_scene) {
			int subscene_count = 0;
			_save_mesh_subscenes_recursive(copy, copy, p_filename.get_basename(), subscene_count);
		}

		bake_func_substep(0.4, p_filename, nullptr, false);

		Ref<PackedScene> sdata = memnew(PackedScene);
		Error err = sdata->pack(copy);
		memdelete(copy);

		bake_func_substep(0.8, p_filename, nullptr, false);

		if (err != OK) {
			EditorNode::get_singleton()->show_warning(TTR("Couldn't save merged branch.\nLikely dependencies (instances) couldn't be satisfied."));
			return false;
		}

		err = ResourceSaver::save(p_filename, sdata, ResourceSaver::FLAG_COMPRESS);
		if (err != OK) {
			EditorNode::get_singleton()->show_warning(TTR("Error saving scene."));
			return false;
		}
	} else {
		EditorNode::get_singleton()->show_warning(TTR("Error duplicating scene to save it."));
		return false;
	}

	return true;
}

void MergeGroupEditorPlugin::_duplicate_branch(Node *p_branch, Node *p_new_parent) {
	Node *dup = p_branch->duplicate();

	ERR_FAIL_NULL(dup);

	p_new_parent->add_child(dup);

	Node *new_owner = p_new_parent->get_owner();
	dup->set_owner(new_owner);

	MergingTool::_invalidate_owner_recursive(dup, nullptr, new_owner);
}

void MergeGroupEditorPlugin::edit(Object *p_object) {
	MergeGroup *mg = Object::cast_to<MergeGroup>(p_object);
	if (!mg) {
		return;
	}

	_merge_group = mg;
}

bool MergeGroupEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("MergeGroup");
}

void MergeGroupEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button_bake->show();
	} else {
		button_bake->hide();
		bake_dialog->hide();
	}
}

void MergeGroupEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_bake", &MergeGroupEditorPlugin::_bake);
	ClassDB::bind_method("_bake_select_file", &MergeGroupEditorPlugin::_bake_select_file);
}

MergeGroupEditorPlugin::MergeGroupEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	button_bake = memnew(ToolButton);
	button_bake->set_icon(editor->get_gui_base()->get_icon("Bake", "EditorIcons"));
	button_bake->set_text(TTR("Bake"));
	button_bake->set_tooltip(TTR("Bake MergeGroup to Scene."));
	button_bake->hide();
	button_bake->connect("pressed", this, "_bake");

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	file_dialog->add_filter("*.tscn ; " + TTR("Scene"));
	file_dialog->add_filter("*.scn ; " + TTR("Binary Scene"));
	file_dialog->set_title(TTR("Save Merged Scene As..."));
	file_dialog->connect("file_selected", this, "_bake_select_file");
	button_bake->add_child(file_dialog);

	bake_dialog = memnew(MergeGroupEditorBakeDialog(this));
	bake_dialog->set_anchors_and_margins_preset(Control::PRESET_CENTER);
	bake_dialog->hide();
	button_bake->add_child(bake_dialog);

	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, button_bake);

	_merge_group = nullptr;

	MergeGroup::bake_step_function = bake_func_step;
	MergeGroup::bake_substep_function = bake_func_substep;
	MergeGroup::bake_end_function = bake_func_end;
}

MergeGroupEditorPlugin::~MergeGroupEditorPlugin() {
}
