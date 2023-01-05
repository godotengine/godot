/**************************************************************************/
/*  many_bone_ik_plugin.h                                                 */
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

#ifndef MANY_BONE_IK_PLUGIN_H
#define MANY_BONE_IK_PLUGIN_H

#include "modules/many_bone_ik/src/many_bone_ik_3d.h"

#include "editor/editor_inspector.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/plugins/skeleton_3d_editor_plugin.h"

class ManyBoneIK3DEditorPlugin;
class ManyBoneIK3DEditor;
class EditorInspectorPluginManyBoneIK : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginManyBoneIK, EditorInspectorPlugin);

	friend class ManyBoneIK3DEditorPlugin;
	ManyBoneIK3DEditor *skel_editor = nullptr;

public:
	EditorInspectorPluginManyBoneIK() {
	}
	virtual ~EditorInspectorPluginManyBoneIK() {
	}
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class ManyBoneIK3DEditor : public VBoxContainer {
	GDCLASS(ManyBoneIK3DEditor, VBoxContainer);

	Tree *joint_tree = nullptr;
	ManyBoneIK3D *ik = nullptr;
	BoneId selected_bone = -1;

	EditorInspectorSection *constraint_bone_section = nullptr;
	EditorPropertyFloat *bone_damp_float = nullptr;
	EditorPropertyNodePath *target_nodepath = nullptr;
	EditorPropertyFloat *twist_from_float = nullptr;
	EditorPropertyFloat *twist_range_float = nullptr;
	static const int32_t MAX_KUSUDAMA_CONES = 10;
	EditorPropertyFloat *cone_count_float = nullptr;
	EditorPropertyVector3 *center_vector3[MAX_KUSUDAMA_CONES] = {};
	EditorPropertyFloat *radius_float[MAX_KUSUDAMA_CONES] = {};
	EditorPropertyTransform3D *twist_constraint_transform = nullptr;
	EditorPropertyTransform3D *orientation_constraint_transform = nullptr;
	EditorPropertyTransform3D *bone_direction_transform = nullptr;
	EditorPropertyFloat *passthrough_float = nullptr;
	EditorPropertyFloat *weight_float = nullptr;
	EditorPropertyVector3 *direction_priorities_vector3 = nullptr;

protected:
	void _notification(int p_what);

public:
	ManyBoneIK3DEditor(EditorInspectorPluginManyBoneIK *e_plugin, ManyBoneIK3D *p_ik) {
		ik = p_ik;
		create_editors();
	}

	void _update_properties();
	void update_joint_tree();
	void create_editors();
	void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);
	void select_bone(int p_idx);
	void _joint_tree_selection_changed();
	TreeItem *_find(TreeItem *p_node, const NodePath &p_path);
};

class ManyBoneIK3DEditorPlugin : public EditorPlugin {
	GDCLASS(ManyBoneIK3DEditorPlugin, EditorPlugin);
	EditorInspectorPluginManyBoneIK *skeleton_plugin = nullptr;

public:
	virtual EditorPlugin::AfterGUIInput forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) override;
	bool has_main_screen() const override;
	virtual bool handles(Object *p_object) const override;
	virtual String get_name() const override;
	ManyBoneIK3DEditorPlugin() {
		skeleton_plugin = memnew(EditorInspectorPluginManyBoneIK);

		EditorInspector::add_inspector_plugin(skeleton_plugin);

		Ref<Skeleton3DGizmoPlugin> gizmo_plugin = Ref<Skeleton3DGizmoPlugin>(memnew(Skeleton3DGizmoPlugin));
		Node3DEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
	}
};

#endif // MANY_BONE_IK_PLUGIN_H
