/**************************************************************************/
/*  vr_editor.cpp                                                         */
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

#include "vr_editor.h"

#include "editor/editor_settings.h"
#include "editor/filesystem_dock.h"
#include "editor/import_dock.h"
#include "editor/inspector_dock.h"
#include "editor/node_dock.h"
#include "editor/scene_tree_dock.h"
#include "servers/xr/xr_interface.h"
#include "servers/xr_server.h"

EditorNode *VREditor::init_editor(SceneTree *p_scene_tree) {
	Ref<XRInterface> xr_interface = XRServer::get_singleton()->find_interface("OpenXR");
	if (xr_interface.is_valid() && xr_interface->is_initialized()) {
		Viewport *vp = p_scene_tree->get_root();

		// Make sure V-Sync is OFF or our monitor frequency will limit our headset
		// TODO improve this to only override v-sync when the player is wearing the headset
		DisplayServer::get_singleton()->window_set_vsync_mode(DisplayServer::VSYNC_DISABLED);

		// Enable our viewport for VR use
		vp->set_vrs_mode(Viewport::VRS_XR);
		vp->set_use_xr(true);

		// Now add our VR editor
		VREditor *vr_pm = memnew(VREditor(vp));
		vp->add_child(vr_pm);

		return vr_pm->get_editor_node();
	} else {
		// Our XR interface was never setup properly, for now assume this means we're not running in XR mode
		// TODO improve this so if we were meant to be in XR mode but failed, especially if we're stand alone, we should hard exit.
		return nullptr;
	}
}

void VREditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("update_transform_gizmo_view"), &VREditor::update_transform_gizmo_view); // Used by call_deferred.
}

void VREditor::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
			_init_gizmo_instance();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_finish_gizmo_instances();
		} break;

		case NOTIFICATION_PROCESS: {
			// For now call this every frame, this needs to change but we currently don't have the hooks Node3DEditorViewport has
			update_transform_gizmo_view();
		} break;
	}
}

/* Camera/Avatar */

void VREditor::_update_layers() {
	uint32_t layers = ((1 << 20) - 1) | (1 << GIZMO_VR_LAYER) | (1 << Node3DEditorViewport::GIZMO_GRID_LAYER) | (1 << Node3DEditorViewport::MISC_TOOL_LAYER);

	// TODO make this optional
	// if (current) {
	//	layers |= (1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	//}
	layers |= (1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);

	avatar->set_camera_cull_layers(layers);
}

/* Gizmo handling */

void VREditor::_init_gizmo_instance() {
	uint32_t layer = 1 << GIZMO_VR_LAYER;

	for (int i = 0; i < 3; i++) {
		move_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(move_gizmo_instance[i], spatial_editor->get_move_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(move_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(move_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(move_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(move_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(move_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		move_plane_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(move_plane_gizmo_instance[i], spatial_editor->get_move_plane_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(move_plane_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(move_plane_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(move_plane_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(move_plane_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(move_plane_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		rotate_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(rotate_gizmo_instance[i], spatial_editor->get_rotate_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(rotate_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(rotate_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(rotate_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		scale_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(scale_gizmo_instance[i], spatial_editor->get_scale_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(scale_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(scale_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(scale_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(scale_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(scale_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(scale_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		scale_plane_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(scale_plane_gizmo_instance[i], spatial_editor->get_scale_plane_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(scale_plane_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(scale_plane_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(scale_plane_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(scale_plane_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(scale_plane_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		axis_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(axis_gizmo_instance[i], spatial_editor->get_axis_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(axis_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(axis_gizmo_instance[i], true);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(axis_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(axis_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(axis_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(axis_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	}

	// Rotation white outline
	rotate_gizmo_instance[3] = RS::get_singleton()->instance_create();
	RS::get_singleton()->instance_set_base(rotate_gizmo_instance[3], spatial_editor->get_rotate_gizmo(3)->get_rid());
	RS::get_singleton()->instance_set_scenario(rotate_gizmo_instance[3], get_tree()->get_root()->get_world_3d()->get_scenario());
	RS::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], false);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(rotate_gizmo_instance[3], RS::SHADOW_CASTING_SETTING_OFF);
	RS::get_singleton()->instance_set_layer_mask(rotate_gizmo_instance[3], layer);
	RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[3], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[3], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
}

void VREditor::_finish_gizmo_instances() {
	for (int i = 0; i < 3; i++) {
		RS::get_singleton()->free(move_gizmo_instance[i]);
		RS::get_singleton()->free(move_plane_gizmo_instance[i]);
		RS::get_singleton()->free(rotate_gizmo_instance[i]);
		RS::get_singleton()->free(scale_gizmo_instance[i]);
		RS::get_singleton()->free(scale_plane_gizmo_instance[i]);
		RS::get_singleton()->free(axis_gizmo_instance[i]);
	}
	// Rotation white outline
	RS::get_singleton()->free(rotate_gizmo_instance[3]);
}

void VREditor::update_transform_gizmo_view() {
	// While our Gizmo logic is mostly the same as in Node3DEditorViewport
	// we can't scale our gizmo the same way as on desktop as stereoscopic
	// rendering let the brain know exactly what is going on.

	if (!is_visible_in_tree()) {
		return;
	}

	Transform3D xform = spatial_editor->get_gizmo_transform();

	Transform3D camera_xform = avatar->get_camera_transform();

	if (xform.origin.is_equal_approx(camera_xform.origin)) {
		for (int i = 0; i < 3; i++) {
			RenderingServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(axis_gizmo_instance[i], false);
		}
		// Rotation white outline
		RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], false);
		return;
	}

	// if the determinant is zero, we should disable the gizmo from being rendered
	// this prevents supplying bad values to the renderer and then having to filter it out again
	if (xform.basis.determinant() == 0) {
		for (int i = 0; i < 3; i++) {
			RenderingServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], false);
		}
		// Rotation white outline
		RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], false);
		return;
	}

	for (int i = 0; i < 3; i++) {
		Transform3D axis_angle;
		if (xform.basis.get_column(i).normalized().dot(xform.basis.get_column((i + 1) % 3).normalized()) < 1.0) {
			axis_angle = axis_angle.looking_at(xform.basis.get_column(i).normalized(), xform.basis.get_column((i + 1) % 3).normalized());
		}
		axis_angle.origin = xform.origin;
		RenderingServer::get_singleton()->instance_set_transform(move_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_MOVE));
		RenderingServer::get_singleton()->instance_set_transform(move_plane_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_MOVE));
		RenderingServer::get_singleton()->instance_set_transform(rotate_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_ROTATE));
		RenderingServer::get_singleton()->instance_set_transform(scale_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(scale_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SCALE));
		RenderingServer::get_singleton()->instance_set_transform(scale_plane_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SCALE));
		RenderingServer::get_singleton()->instance_set_transform(axis_gizmo_instance[i], xform);
	}

	/* re-implement, this is used when editing
	bool show_axes = spatial_editor->is_gizmo_visible() && _edit.mode != TRANSFORM_NONE;
	RenderingServer *rs = RenderingServer::get_singleton();
	rs->instance_set_visible(axis_gizmo_instance[0], show_axes && (_edit.plane == TRANSFORM_X_AXIS || _edit.plane == TRANSFORM_XY || _edit.plane == TRANSFORM_XZ));
	rs->instance_set_visible(axis_gizmo_instance[1], show_axes && (_edit.plane == TRANSFORM_Y_AXIS || _edit.plane == TRANSFORM_XY || _edit.plane == TRANSFORM_YZ));
	rs->instance_set_visible(axis_gizmo_instance[2], show_axes && (_edit.plane == TRANSFORM_Z_AXIS || _edit.plane == TRANSFORM_XZ || _edit.plane == TRANSFORM_YZ));
	*/

	// Rotation white outline
	xform.orthonormalize();
	RenderingServer::get_singleton()->instance_set_transform(rotate_gizmo_instance[3], xform);
	RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_ROTATE));
}

/* Initialisation */

VREditor::VREditor(Viewport *p_xr_viewport) {
	xr_viewport = p_xr_viewport;

	// Make sure our editor settings are loaded...
	if (!EditorSettings::get_singleton()) {
		EditorSettings::create();
	}

	// Add our avatar
	avatar = memnew(VREditorAvatar);
	xr_viewport->add_child(avatar);

	// Create a window to add our normal editor in
	editor_window = memnew(VRWindow(Size2i(2000.0, 1000.0), 0.0015));
	editor_window->set_name("Editor");
	editor_window->set_curve_depth(0.2);
	editor_window->set_position(Vector3(0.0, 0.0, -1.0));
	avatar->get_hud_root()->add_child(editor_window);

	editor_node = memnew(EditorNode);
	editor_window->get_scene_root()->add_child(editor_node);

	// Now we're going to hijack our 3d editor node
	spatial_editor = Node3DEditor::get_singleton();
	spatial_editor->disable_3d_viewports();

	// Setup our editing view (maybe more this into a handler class?)
	_update_layers();

	// make sure process is called
	set_process(true);
}

VREditor::~VREditor() {
}
