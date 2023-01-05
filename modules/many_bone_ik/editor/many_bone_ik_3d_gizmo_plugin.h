/**************************************************************************/
/*  many_bone_ik_3d_gizmo_plugin.h                                        */
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

#ifndef MANY_BONE_IK_3D_GIZMO_PLUGIN_H
#define MANY_BONE_IK_3D_GIZMO_PLUGIN_H

#include "modules/many_bone_ik/editor/many_bone_ik_shader.h"
#include "modules/many_bone_ik/src/ik_bone_3d.h"
#include "modules/many_bone_ik/src/many_bone_ik_3d.h"

#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "editor/editor_inspector.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/editor_settings.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/plugins/skeleton_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/immediate_mesh.h"

class Joint;
class PhysicalBone3D;
class ManyBoneIKEditorPlugin;
class Button;

class ManyBoneIK3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(ManyBoneIK3DGizmoPlugin, EditorNode3DGizmoPlugin);
	Ref<Shader> kusudama_shader;

protected:
	static void _bind_methods();

public:
	const Color bone_color = EditorSettings::get_singleton()->get("editors/3d_gizmos/gizmo_colors/skeleton");
	const int32_t KUSUDAMA_MAX_CONES = 10;
	bool has_gizmo(Node3D *p_spatial) override;
	String get_gizmo_name() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;
	ManyBoneIK3DGizmoPlugin();
	int32_t get_priority() const override;
	void create_gizmo_mesh(BoneId current_bone_idx, Ref<IKBone3D> ik_bone, EditorNode3DGizmo *p_gizmo, Color current_bone_color, Skeleton3D *many_bone_ik_skeleton, ManyBoneIK3D *p_many_bone_ik);
};

class EditorPluginManyBoneIK : public EditorPlugin {
	GDCLASS(EditorPluginManyBoneIK, EditorPlugin);

public:
	EditorPluginManyBoneIK();
};

#endif // MANY_BONE_IK_3D_GIZMO_PLUGIN_H
