/**************************************************************************/
/*  node_3d_editor_plugin.cpp                                             */
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

#include "node_3d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/math/geometry_3d.h"
#include "core/math/math_funcs.h"
#include "core/math/projection.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_spin_slider.h"
#include "editor/plugins/editor_plugin_list.h"
#include "editor/run/editor_run_bar.h"
#include "editor/scene/3d/gizmos/audio_listener_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/audio_stream_player_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/camera_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/chain_ik_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/cpu_particles_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/decal_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/fog_volume_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/geometry_instance_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/gpu_particles_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/gpu_particles_collision_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/label_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/light_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/lightmap_gi_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/lightmap_probe_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/marker_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/mesh_instance_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/occluder_instance_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/particles_3d_emission_shape_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/collision_object_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/collision_polygon_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/collision_shape_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/joint_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/physics_bone_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/ray_cast_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/shape_cast_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/soft_body_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/spring_arm_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/physics/vehicle_body_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/reflection_probe_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/spring_bone_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/sprite_base_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/two_bone_ik_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/visible_on_screen_notifier_3d_gizmo_plugin.h"
#include "editor/scene/3d/gizmos/voxel_gi_gizmo_plugin.h"
#include "editor/scene/3d/node_3d_editor_constants.h"
#include "editor/scene/3d/node_3d_editor_gizmos.h"
#include "editor/scene/3d/node_3d_editor_viewport.h"
#include "editor/settings/editor_settings.h"
#include "editor/translations/editor_translation_preview_menu.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "scene/3d/physics/physics_body_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/gui/button.h"
#include "scene/gui/center_container.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/3d/sky_material.h"
#include "scene/resources/sky.h"
#include "scene/resources/surface_tool.h"
#include "servers/physics_3d/physics_server_3d_types.h"
#include "servers/rendering/rendering_server.h"

using namespace Node3DEditorConstants;

///////////////////////////////////////////////////////////////////

Node3DEditor *Node3DEditor::singleton = nullptr;

void Node3DEditor::select_gizmo_highlight_axis(int p_axis) {
	for (int i = 0; i < 3; i++) {
		move_gizmo[i]->surface_set_material(0, i == p_axis ? gizmo_color_hl[i] : gizmo_color[i]);
		move_plane_gizmo[i]->surface_set_material(0, (i + 6) == p_axis ? plane_gizmo_color_hl[i] : plane_gizmo_color[i]);
		scale_gizmo[i]->surface_set_material(0, (i + 9) == p_axis ? gizmo_color_hl[i] : gizmo_color[i]);
		scale_plane_gizmo[i]->surface_set_material(0, (i + 12) == p_axis ? plane_gizmo_color_hl[i] : plane_gizmo_color[i]);
	}

	for (int i = 0; i < 4; i++) {
		bool highlight;
		if (i == 3) {
			highlight = (p_axis == GIZMO_HIGHLIGHT_AXIS_VIEW_ROTATION);
		} else {
			highlight = (i + 3) == p_axis;
		}
		rotate_gizmo[i]->surface_set_material(0, highlight ? rotate_gizmo_color_hl[i] : rotate_gizmo_color[i]);
	}

	bool highlight_trackball = (p_axis == GIZMO_HIGHLIGHT_AXIS_TRACKBALL);
	trackball_sphere_gizmo->surface_set_material(0, highlight_trackball ? trackball_sphere_material_hl : trackball_sphere_material);
}

void Node3DEditor::update_transform_gizmo() {
	int count = 0;
	bool local_gizmo_coords = are_local_coords_enabled();

	Vector3 gizmo_center;
	Basis gizmo_basis;

	Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;

	if (se && se->gizmo.is_valid()) {
		for (const KeyValue<int, Transform3D> &E : se->subgizmos) {
			Transform3D xf = se->sp->get_global_transform() * se->gizmo->get_subgizmo_transform(E.key);
			if (!xf.is_finite()) {
				continue;
			}
			gizmo_center += xf.origin;
			if ((unsigned int)count == se->subgizmos.size() - 1 && local_gizmo_coords) {
				gizmo_basis = xf.basis;
			}
			count++;
		}
	} else {
		const List<Node *> &selection = editor_selection->get_top_selected_node_list();
		for (Node *E : selection) {
			Node3D *sp = Object::cast_to<Node3D>(E);
			if (!sp) {
				continue;
			}

			if (sp->has_meta("_edit_lock_")) {
				continue;
			}

			Node3DEditorSelectedItem *sel_item = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
			if (!sel_item) {
				continue;
			}

			Transform3D xf = sel_item->sp->get_global_transform();
			if (!xf.is_finite()) {
				continue;
			}
			gizmo_center += xf.origin;
			if (count == selection.size() - 1 && local_gizmo_coords) {
				gizmo_basis = xf.basis;
			}
			count++;
		}
	}

	gizmo.visible = count > 0;
	gizmo.transform.origin = (count > 0) ? gizmo_center / count : Vector3();
	gizmo.transform.basis = gizmo_basis;

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->update_transform_gizmo_view();
	}
}

void _update_all_gizmos(Node *p_node) {
	for (int i = p_node->get_child_count() - 1; 0 <= i; --i) {
		Node3D *spatial_node = Object::cast_to<Node3D>(p_node->get_child(i));
		if (spatial_node) {
			spatial_node->update_gizmos();
		}

		_update_all_gizmos(p_node->get_child(i));
	}
}

void Node3DEditor::update_all_gizmos(Node *p_node) {
	if (!p_node && is_inside_tree()) {
		p_node = get_tree()->get_edited_scene_root();
	}

	if (!p_node) {
		// No edited scene, so nothing to update.
		return;
	}
	_update_all_gizmos(p_node);
}

Object *Node3DEditor::_get_editor_data(Object *p_what) {
	Node3D *sp = Object::cast_to<Node3D>(p_what);
	if (!sp) {
		return nullptr;
	}

	Node3DEditorSelectedItem *si = memnew(Node3DEditorSelectedItem);

	si->sp = sp;
	si->sbox_instance = RenderingServer::get_singleton()->instance_create2(
			selection_box->get_rid(),
			sp->get_world_3d()->get_scenario());
	si->sbox_instance_offset = RenderingServer::get_singleton()->instance_create2(
			selection_box->get_rid(),
			sp->get_world_3d()->get_scenario());
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance,
			RSE::SHADOW_CASTING_SETTING_OFF);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance_offset,
			RSE::SHADOW_CASTING_SETTING_OFF);
	// Use the Edit layer to hide the selection box when View Gizmos is disabled, since it is a bit distracting.
	// It's still possible to approximately guess what is selected by looking at the manipulation gizmo position.
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance_offset, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_offset, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_offset, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	si->sbox_instance_xray = RenderingServer::get_singleton()->instance_create2(
			selection_box_xray->get_rid(),
			sp->get_world_3d()->get_scenario());
	si->sbox_instance_xray_offset = RenderingServer::get_singleton()->instance_create2(
			selection_box_xray->get_rid(),
			sp->get_world_3d()->get_scenario());
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance_xray,
			RSE::SHADOW_CASTING_SETTING_OFF);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance_xray_offset,
			RSE::SHADOW_CASTING_SETTING_OFF);
	// Use the Edit layer to hide the selection box when View Gizmos is disabled, since it is a bit distracting.
	// It's still possible to approximately guess what is selected by looking at the manipulation gizmo position.
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance_xray, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance_xray_offset, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray_offset, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray_offset, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

	return si;
}

void Node3DEditor::_generate_selection_boxes() {
	// Use two AABBs to create the illusion of a slightly thicker line.
	AABB aabb(Vector3(), Vector3(1, 1, 1));

	// Create a x-ray (visible through solid surfaces) and standard version of the selection box.
	// Both will be drawn at the same position, but with different opacity.
	// This lets the user see where the selection is while still having a sense of depth.
	Ref<SurfaceTool> st = memnew(SurfaceTool);
	Ref<SurfaceTool> st_xray = memnew(SurfaceTool);
	Ref<SurfaceTool> active_st = memnew(SurfaceTool);
	Ref<SurfaceTool> active_st_xray = memnew(SurfaceTool);

	st->begin(Mesh::PRIMITIVE_LINES);
	st_xray->begin(Mesh::PRIMITIVE_LINES);
	active_st->begin(Mesh::PRIMITIVE_LINES);
	active_st_xray->begin(Mesh::PRIMITIVE_LINES);
	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);

		st->add_vertex(a);
		st->add_vertex(b);
		active_st->add_vertex(a);
		active_st->add_vertex(b);
		st_xray->add_vertex(a);
		st_xray->add_vertex(b);
		active_st_xray->add_vertex(a);
		active_st_xray->add_vertex(b);
	}

	const Color selection_box_color = EDITOR_GET("editors/3d/selection_box_color");
	const Color active_selection_box_color = EDITOR_GET("editors/3d/active_selection_box_color");

	selection_box_mat.instantiate();
	selection_box_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	selection_box_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	selection_box_mat->set_albedo(selection_box_color);
	selection_box_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st->set_material(selection_box_mat);
	selection_box = st->commit();

	selection_box_mat_xray.instantiate();
	selection_box_mat_xray->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	selection_box_mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	selection_box_mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	selection_box_mat_xray->set_albedo(selection_box_color * Color(1, 1, 1, 0.15));
	selection_box_mat_xray->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st_xray->set_material(selection_box_mat_xray);
	selection_box_xray = st_xray->commit();

	active_selection_box_mat.instantiate();
	active_selection_box_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	active_selection_box_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	active_selection_box_mat->set_albedo(active_selection_box_color);
	active_selection_box_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	active_st->set_material(active_selection_box_mat);
	active_selection_box = active_st->commit();

	active_selection_box_mat_xray.instantiate();
	active_selection_box_mat_xray->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	active_selection_box_mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	active_selection_box_mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	active_selection_box_mat_xray->set_albedo(active_selection_box_color * Color(1, 1, 1, 0.15));
	active_selection_box_mat_xray->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	active_st_xray->set_material(active_selection_box_mat_xray);
	active_selection_box_xray = active_st_xray->commit();
}

Dictionary Node3DEditor::get_state() const {
	Dictionary d;

	d["snap_enabled"] = snap_enabled;
	d["trackball_enabled"] = trackball_enabled;
	d["translate_snap"] = snap_translate_value;
	d["rotate_snap"] = snap_rotate_value;
	d["scale_snap"] = snap_scale_value;

	d["local_coords"] = tool_option_button[TOOL_OPT_LOCAL_COORDS]->is_pressed();
	d["preserve_children_transform"] = tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->is_pressed();

	int vc = 0;
	if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT))) {
		vc = 1;
	} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS))) {
		vc = 2;
	} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS))) {
		vc = 3;
	} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS))) {
		vc = 4;
	} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT))) {
		vc = 5;
	} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT))) {
		vc = 6;
	}

	d["viewport_mode"] = vc;
	d["viewport_splits"] = viewport_base->get_split_state();
	Array vpdata;
	for (int i = 0; i < 4; i++) {
		vpdata.push_back(viewports[i]->get_state());
	}

	d["viewports"] = vpdata;

	d["vertex_snap_origin_mode"] = vertex_snap_origin_mode;
	d["vertex_snap_use_collision"] = vertex_snap_use_collision;
	d["show_grid"] = view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_GRID));
	d["show_origin"] = view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN));
	d["fov"] = get_fov();
	d["znear"] = get_znear();
	d["zfar"] = get_zfar();

	Dictionary gizmos_status;
	for (int i = 0; i < gizmo_plugins_by_name.size(); i++) {
		if (!gizmo_plugins_by_name[i]->can_be_hidden()) {
			continue;
		}
		int state = gizmos_menu->get_item_state(gizmos_menu->get_item_index(i));
		String name = gizmo_plugins_by_name[i]->get_gizmo_name();
		gizmos_status[name] = state;
	}

	d["gizmos_status"] = gizmos_status;
	{
		Dictionary pd;

		pd["sun_rotation"] = sun_rotation;

		pd["environ_sky_color"] = environ_sky_color->get_pick_color();
		pd["environ_ground_color"] = environ_ground_color->get_pick_color();
		pd["environ_energy"] = environ_energy->get_value();
		pd["environ_glow_enabled"] = environ_glow_button->is_pressed();
		pd["environ_tonemap_enabled"] = environ_tonemap_button->is_pressed();
		pd["environ_ao_enabled"] = environ_ao_button->is_pressed();
		pd["environ_gi_enabled"] = environ_gi_button->is_pressed();
		pd["sun_shadow_max_distance"] = sun_shadow_max_distance->get_value();

		pd["sun_color"] = sun_color->get_pick_color();
		pd["sun_energy"] = sun_energy->get_value();

		pd["sun_enabled"] = sun_button->is_pressed();
		pd["environ_enabled"] = environ_button->is_pressed();

		d["preview_sun_env"] = pd;
	}

	return d;
}

void Node3DEditor::set_state(const Dictionary &p_state) {
	Dictionary d = p_state;

	if (d.has("snap_enabled")) {
		snap_enabled = d["snap_enabled"];
		tool_option_button[TOOL_OPT_USE_SNAP]->set_pressed(d["snap_enabled"]);
	}

	if (d.has("trackball_enabled")) {
		trackball_enabled = d["trackball_enabled"];
		tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_pressed(d["trackball_enabled"]);
	}

	if (d.has("translate_snap")) {
		snap_translate_value = d["translate_snap"];
	}

	if (d.has("rotate_snap")) {
		snap_rotate_value = d["rotate_snap"];
	}

	if (d.has("scale_snap")) {
		snap_scale_value = d["scale_snap"];
	}

	_snap_update();

	if (d.has("preserve_children_transform")) {
		tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_pressed(d["preserve_children_transform"]);
	}

	if (d.has("vertex_snap_origin_mode")) {
		vertex_snap_origin_mode = d["vertex_snap_origin_mode"];
		int idx_vertex = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_BASE_VERTEX);
		int idx_origin = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_BASE_ORIGIN);
		transform_menu->get_popup()->set_item_checked(idx_vertex, !vertex_snap_origin_mode);
		transform_menu->get_popup()->set_item_checked(idx_origin, vertex_snap_origin_mode);
	}

	if (d.has("vertex_snap_use_collision")) {
		vertex_snap_use_collision = d["vertex_snap_use_collision"];
		int idx_mesh = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_SOURCE_MESH);
		int idx_collision = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_SOURCE_COLLISION);
		transform_menu->get_popup()->set_item_checked(idx_mesh, !vertex_snap_use_collision);
		transform_menu->get_popup()->set_item_checked(idx_collision, vertex_snap_use_collision);
	}

	if (d.has("local_coords")) {
		tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_pressed(d["local_coords"]);
		update_transform_gizmo();
	}

	if (d.has("viewport_mode")) {
		int vc = d["viewport_mode"];

		if (vc == 1) {
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);
		} else if (vc == 2) {
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS);
		} else if (vc == 3) {
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS);
		} else if (vc == 4) {
			_menu_item_pressed(MENU_VIEW_USE_4_VIEWPORTS);
		} else if (vc == 5) {
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS_ALT);
		} else if (vc == 6) {
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS_ALT);
		}
	}

	if (d.has("zfar")) {
		settings_zfar->set_value(double(d["zfar"]));
	}
	if (d.has("znear")) {
		settings_znear->set_value(double(d["znear"]));
	}
	if (d.has("fov")) {
		settings_fov->set_value(double(d["fov"]));
	}
	if (d.has("viewport_splits")) {
		viewport_base->set_split_state(d["viewport_splits"]);
	}

	if (d.has("viewports")) {
		Array vp = d["viewports"];
		uint32_t vp_size = static_cast<uint32_t>(vp.size());
		if (vp_size > VIEWPORTS_COUNT) {
			WARN_PRINT("Ignoring superfluous viewport settings from spatial editor state.");
			vp_size = VIEWPORTS_COUNT;
		}

		for (uint32_t i = 0; i < vp_size; i++) {
			viewports[i]->set_state(vp[i]);
		}
	}

	if (d.has("show_grid")) {
		bool use = d["show_grid"];

		if (use != view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_GRID))) {
			_menu_item_pressed(MENU_VIEW_GRID);
		}
	}

	if (d.has("show_origin")) {
		bool use = d["show_origin"];

		if (use != view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN))) {
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN), use);
			RenderingServer::get_singleton()->instance_set_visible(origin_instance, use);
		}
	}

	if (d.has("gizmos_status")) {
		Dictionary gizmos_status = d["gizmos_status"];

		for (int j = 0; j < gizmo_plugins_by_name.size(); ++j) {
			if (!gizmo_plugins_by_name[j]->can_be_hidden()) {
				continue;
			}
			int state = EditorNode3DGizmoPlugin::VISIBLE;
			for (const KeyValue<Variant, Variant> &kv : gizmos_status) {
				if (gizmo_plugins_by_name.write[j]->get_gizmo_name() == String(kv.key)) {
					state = kv.value;
					break;
				}
			}

			gizmo_plugins_by_name.write[j]->set_state(state);
		}
		_update_gizmos_menu();
	}

	if (d.has("preview_sun_env")) {
		sun_environ_updating = true;
		Dictionary pd = d["preview_sun_env"];
		sun_rotation = pd["sun_rotation"];

		environ_sky_color->set_pick_color(pd["environ_sky_color"]);
		environ_ground_color->set_pick_color(pd["environ_ground_color"]);
		environ_energy->set_value_no_signal(pd["environ_energy"]);
		environ_glow_button->set_pressed_no_signal(pd["environ_glow_enabled"]);
		environ_tonemap_button->set_pressed_no_signal(pd["environ_tonemap_enabled"]);
		environ_ao_button->set_pressed_no_signal(pd["environ_ao_enabled"]);
		environ_gi_button->set_pressed_no_signal(pd["environ_gi_enabled"]);
		sun_shadow_max_distance->set_value_no_signal(pd["sun_shadow_max_distance"]);

		sun_color->set_pick_color(pd["sun_color"]);
		sun_energy->set_value_no_signal(pd["sun_energy"]);

		sun_button->set_pressed(pd["sun_enabled"]);
		environ_button->set_pressed(pd["environ_enabled"]);

		sun_environ_updating = false;

		_preview_settings_changed();
		_update_preview_environment();
	} else {
		_load_default_preview_settings();
		sun_button->set_pressed(true);
		environ_button->set_pressed(true);
		_preview_settings_changed();
		_update_preview_environment();
	}
}

void Node3DEditor::edit(Node3D *p_spatial) {
	if (p_spatial != selected) {
		if (selected) {
			Vector<Ref<Node3DGizmo>> gizmos = selected->get_gizmos();
			for (int i = 0; i < gizmos.size(); i++) {
				Ref<EditorNode3DGizmo> seg = gizmos[i];
				if (seg.is_null()) {
					continue;
				}
				seg->set_selected(false);
			}

			Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected);
			if (se) {
				se->gizmo.unref();
				se->subgizmos.clear();
			}

			selected->update_gizmos();
		}

		selected = p_spatial;
		current_hover_gizmo = Ref<EditorNode3DGizmo>();
		current_hover_gizmo_handle = -1;
		current_hover_gizmo_handle_secondary = false;

		if (selected) {
			Vector<Ref<Node3DGizmo>> gizmos = selected->get_gizmos();
			for (int i = 0; i < gizmos.size(); i++) {
				Ref<EditorNode3DGizmo> seg = gizmos[i];
				if (seg.is_null()) {
					continue;
				}
				seg->set_selected(true);
			}
			selected->update_gizmos();
		}
	}
}

void Node3DEditor::_snap_changed() {
	snap_translate_value = snap_translate->get_value();
	snap_rotate_value = snap_rotate->get_value();
	snap_scale_value = snap_scale->get_value();

	EditorSettings::get_singleton()->set_project_metadata("3d_editor", "snap_translate_value", snap_translate_value);
	EditorSettings::get_singleton()->set_project_metadata("3d_editor", "snap_rotate_value", snap_rotate_value);
	EditorSettings::get_singleton()->set_project_metadata("3d_editor", "snap_scale_value", snap_scale_value);
}

void Node3DEditor::_snap_update() {
	snap_translate->set_value(snap_translate_value);
	snap_rotate->set_value(snap_rotate_value);
	snap_scale->set_value(snap_scale_value);
}

void Node3DEditor::_update_vertex_snap_tooltips() {
	String snap_key = ED_GET_SHORTCUT("spatial_editor/vertex_snap")->get_as_text();
	PopupMenu *p = transform_menu->get_popup();
	p->set_item_tooltip(p->get_item_index(MENU_VERTEX_SNAP_BASE_VERTEX),
			vformat(TTR("Hold %s to highlight a vertex on the currently selected node,\nthen drag to move the node and snap it to vertices on neighboring nodes.\n\nFor nodes without a vertex-based representation,\nSnap Origin to Vertex is always used instead."), snap_key));
	p->set_item_tooltip(p->get_item_index(MENU_VERTEX_SNAP_BASE_ORIGIN),
			vformat(TTR("Hold %s to highlight another node's vertex,\nthen click to teleport the selected node to the highlighted vertex."), snap_key));
	p->set_item_tooltip(p->get_item_index(MENU_VERTEX_SNAP_SOURCE_MESH),
			TTR("Snap to vertices of visual meshes.\nHold Shift while vertex snapping to temporarily snap to collision shapes instead."));
	p->set_item_tooltip(p->get_item_index(MENU_VERTEX_SNAP_SOURCE_COLLISION),
			TTR("Snap to vertices of collision shapes.\nHold Shift while vertex snapping to temporarily snap to mesh vertices instead."));
}

void Node3DEditor::_xform_dialog_action() {
	Transform3D t;
	//translation
	Vector3 scale;
	Vector3 rotate;
	Vector3 translate;

	for (int i = 0; i < 3; i++) {
		translate[i] = xform_translate[i]->get_text().to_float();
		rotate[i] = Math::deg_to_rad(xform_rotate[i]->get_text().to_float());
		scale[i] = xform_scale[i]->get_text().to_float();
	}

	t.basis.scale(scale);
	t.basis.rotate(rotate);
	t.origin = translate;

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("XForm Dialog"));

	const List<Node *> &selection = editor_selection->get_top_selected_node_list();

	for (Node *E : selection) {
		Node3D *sp = Object::cast_to<Node3D>(E);
		if (!sp) {
			continue;
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
		if (!se) {
			continue;
		}

		bool post = xform_type->get_selected() > 0;

		Transform3D tr = sp->get_global_gizmo_transform();
		if (post) {
			tr = tr * t;
		} else {
			tr.basis = t.basis * tr.basis;
			tr.origin += t.origin;
		}

		Node3D *parent = sp->get_parent_node_3d();
		Transform3D local_tr = parent ? parent->get_global_transform().affine_inverse() * tr : tr;
		undo_redo->add_do_method(sp, "set_transform", local_tr);
		undo_redo->add_undo_method(sp, "set_transform", sp->get_transform());
	}
	undo_redo->commit_action();
}

void Node3DEditor::_menu_item_toggled(bool pressed, int p_option) {
	switch (p_option) {
		case MENU_TOOL_LOCAL_COORDS: {
			tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_pressed(pressed);
			update_transform_gizmo();
		} break;

		case MENU_TOOL_USE_SNAP: {
			tool_option_button[TOOL_OPT_USE_SNAP]->set_pressed(pressed);
			snap_enabled = pressed;
		} break;

		case MENU_TOOL_USE_TRACKBALL: {
			tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_pressed(pressed);
			trackball_enabled = pressed;
			for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
				viewports[i]->update_transform_gizmo_highlight();
			}
		} break;

		case MENU_TOOL_PRESERVE_CHILDREN_TRANSFORM: {
			tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_pressed(pressed);
			if (pressed) {
				EditorNode::get_editor_data().add_undo_redo_inspector_hook_callback(callable_mp(this, &Node3DEditor::_undo_redo_inspector_callback));
			} else {
				EditorNode::get_editor_data().remove_undo_redo_inspector_hook_callback(callable_mp(this, &Node3DEditor::_undo_redo_inspector_callback));
			}
		} break;
	}
}

void Node3DEditor::_undo_redo_inspector_callback(Object *p_undo_redo, Object *p_edited, const String &p_property, const Variant &p_new_value) {
	Node3D *node = Object::cast_to<Node3D>(p_edited);
	if (!node) {
		return;
	}

	static const char *transform_properties[] = { "position", "rotation", "scale", "quaternion", "basis", "transform", nullptr };
	bool is_transform_prop = false;
	for (int i = 0; transform_properties[i]; i++) {
		if (p_property == transform_properties[i]) {
			is_transform_prop = true;
			break;
		}
	}
	if (!is_transform_prop) {
		return;
	}

	EditorUndoRedoManager *undo_redo = Object::cast_to<EditorUndoRedoManager>(p_undo_redo);
	ERR_FAIL_NULL(undo_redo);

	int child_count = node->get_child_count();
	for (int i = 0; i < child_count; i++) {
		Node3D *child = Object::cast_to<Node3D>(node->get_child(i));
		if (child) {
			Transform3D child_global = child->get_global_transform();
			undo_redo->add_do_method(child, "set_global_transform", child_global);
			undo_redo->add_undo_method(child, "set_global_transform", child_global);
		}
	}
}

void Node3DEditor::_menu_gizmo_toggled(int p_option) {
	const int idx = gizmos_menu->get_item_index(p_option);
	gizmos_menu->toggle_item_multistate(idx);

	// Change icon
	const int state = gizmos_menu->get_item_state(idx);
	switch (state) {
		case EditorNode3DGizmoPlugin::VISIBLE:
			gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityVisible")));
			break;
		case EditorNode3DGizmoPlugin::ON_TOP:
			gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityXray")));
			break;
		case EditorNode3DGizmoPlugin::HIDDEN:
			gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityHidden")));
			break;
	}

	gizmo_plugins_by_name.write[p_option]->set_state(state);

	update_all_gizmos();
}

void Node3DEditor::_menu_item_pressed(int p_option) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	switch (p_option) {
		case MENU_TOOL_TRANSFORM:
		case MENU_TOOL_MOVE:
		case MENU_TOOL_ROTATE:
		case MENU_TOOL_SCALE:
		case MENU_TOOL_SELECT:
		case MENU_TOOL_LIST_SELECT: {
			for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
				if (viewports[i]->_edit.mode != Node3DEditorViewport::TRANSFORM_NONE) {
					viewports[i]->commit_transform();
				}
			}

			for (int i = 0; i < TOOL_MAX; i++) {
				tool_button[i]->set_pressed(i == p_option);
			}
			tool_mode = (ToolMode)p_option;
			update_transform_gizmo();

			for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
				viewports[i]->update_transform_gizmo_highlight();
			}
		} break;
		case MENU_TRANSFORM_CONFIGURE_SNAP: {
			snap_dialog->popup_centered(Size2(200, 180));
		} break;
		case MENU_VERTEX_SNAP_BASE_VERTEX: {
			vertex_snap_origin_mode = false;
			int idx_vertex = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_BASE_VERTEX);
			int idx_origin = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_BASE_ORIGIN);
			transform_menu->get_popup()->set_item_checked(idx_vertex, true);
			transform_menu->get_popup()->set_item_checked(idx_origin, false);
		} break;
		case MENU_VERTEX_SNAP_BASE_ORIGIN: {
			vertex_snap_origin_mode = true;
			int idx_vertex = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_BASE_VERTEX);
			int idx_origin = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_BASE_ORIGIN);
			transform_menu->get_popup()->set_item_checked(idx_vertex, false);
			transform_menu->get_popup()->set_item_checked(idx_origin, true);
		} break;
		case MENU_VERTEX_SNAP_SOURCE_MESH: {
			vertex_snap_use_collision = false;
			int idx_mesh = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_SOURCE_MESH);
			int idx_collision = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_SOURCE_COLLISION);
			transform_menu->get_popup()->set_item_checked(idx_mesh, true);
			transform_menu->get_popup()->set_item_checked(idx_collision, false);
		} break;
		case MENU_VERTEX_SNAP_SOURCE_COLLISION: {
			vertex_snap_use_collision = true;
			int idx_mesh = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_SOURCE_MESH);
			int idx_collision = transform_menu->get_popup()->get_item_index(MENU_VERTEX_SNAP_SOURCE_COLLISION);
			transform_menu->get_popup()->set_item_checked(idx_mesh, false);
			transform_menu->get_popup()->set_item_checked(idx_collision, true);
		} break;
		case MENU_TRANSFORM_DIALOG: {
			for (int i = 0; i < 3; i++) {
				xform_translate[i]->set_text("0");
				xform_rotate[i]->set_text("0");
				xform_scale[i]->set_text("1");
			}

			xform_dialog->popup_centered(Size2(320, 240) * EDSCALE);

		} break;
		case MENU_VIEW_USE_1_VIEWPORT: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_1_VIEWPORT);
			if (last_used_viewport > 0) {
				last_used_viewport = 0;
			}

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), true);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_2_VIEWPORTS: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_2_VIEWPORTS);
			if (last_used_viewport > 1) {
				last_used_viewport = 0;
			}

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), true);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_2_VIEWPORTS_ALT: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_2_VIEWPORTS_ALT);
			if (last_used_viewport > 1) {
				last_used_viewport = 0;
			}

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), true);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_3_VIEWPORTS: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_3_VIEWPORTS);
			if (last_used_viewport > 2) {
				last_used_viewport = 0;
			}

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), true);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_3_VIEWPORTS_ALT: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_3_VIEWPORTS_ALT);
			if (last_used_viewport > 2) {
				last_used_viewport = 0;
			}

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), true);

		} break;
		case MENU_VIEW_USE_4_VIEWPORTS: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_4_VIEWPORTS);

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), true);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_ORIGIN: {
			bool is_checked = view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(p_option));

			origin_enabled = !is_checked;
			RenderingServer::get_singleton()->instance_set_visible(origin_instance, origin_enabled);
			// Update the grid since its appearance depends on whether the origin is enabled
			_finish_grid();
			_init_grid();

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(p_option), origin_enabled);
		} break;
		case MENU_VIEW_GRID: {
			bool is_checked = view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(p_option));

			grid_enabled = !is_checked;

			for (int i = 0; i < 3; ++i) {
				if (grid_enable[i]) {
					grid_visible[i] = grid_enabled;
				}
			}
			_finish_grid();
			_init_grid();

			view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(p_option), grid_enabled);

		} break;
		case MENU_VIEW_CAMERA_SETTINGS: {
			settings_dialog->popup_centered(settings_vbc->get_combined_minimum_size() + Size2(50, 50));
		} break;
		case MENU_SNAP_TO_FLOOR: {
			snap_selected_nodes_to_floor();
		} break;
		case MENU_LOCK_SELECTED: {
			undo_redo->create_action(TTR("Lock Selected"));

			const List<Node *> &selection = editor_selection->get_top_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				undo_redo->add_do_method(spatial, "set_meta", "_edit_lock_", true);
				undo_redo->add_undo_method(spatial, "remove_meta", "_edit_lock_");
				undo_redo->add_do_method(this, "emit_signal", "item_lock_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_lock_status_changed");
			}

			undo_redo->add_do_method(this, "_refresh_menu_icons");
			undo_redo->add_undo_method(this, "_refresh_menu_icons");
			undo_redo->add_do_method(this, "update_transform_gizmo");
			undo_redo->add_undo_method(this, "update_transform_gizmo");
			undo_redo->commit_action();
		} break;
		case MENU_UNLOCK_SELECTED: {
			undo_redo->create_action(TTR("Unlock Selected"));

			const List<Node *> &selection = editor_selection->get_top_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				undo_redo->add_do_method(spatial, "remove_meta", "_edit_lock_");
				undo_redo->add_undo_method(spatial, "set_meta", "_edit_lock_", true);
				undo_redo->add_do_method(this, "emit_signal", "item_lock_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_lock_status_changed");
			}

			undo_redo->add_do_method(this, "_refresh_menu_icons");
			undo_redo->add_undo_method(this, "_refresh_menu_icons");
			undo_redo->add_do_method(this, "update_transform_gizmo");
			undo_redo->add_undo_method(this, "update_transform_gizmo");
			undo_redo->commit_action();
		} break;
		case MENU_GROUP_SELECTED: {
			undo_redo->create_action(TTR("Group Selected"));

			const List<Node *> &selection = editor_selection->get_top_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				undo_redo->add_do_method(spatial, "set_meta", "_edit_group_", true);
				undo_redo->add_undo_method(spatial, "remove_meta", "_edit_group_");
				undo_redo->add_do_method(this, "emit_signal", "item_group_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_group_status_changed");
			}

			undo_redo->add_do_method(this, "_refresh_menu_icons");
			undo_redo->add_undo_method(this, "_refresh_menu_icons");
			undo_redo->commit_action();
		} break;
		case MENU_UNGROUP_SELECTED: {
			undo_redo->create_action(TTR("Ungroup Selected"));
			const List<Node *> &selection = editor_selection->get_top_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				undo_redo->add_do_method(spatial, "remove_meta", "_edit_group_");
				undo_redo->add_undo_method(spatial, "set_meta", "_edit_group_", true);
				undo_redo->add_do_method(this, "emit_signal", "item_group_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_group_status_changed");
			}

			undo_redo->add_do_method(this, "_refresh_menu_icons");
			undo_redo->add_undo_method(this, "_refresh_menu_icons");
			undo_redo->commit_action();
		} break;
		case MENU_RULER: {
			for (int i = 0; i < TOOL_MAX; i++) {
				tool_button[i]->set_pressed(i == p_option);
			}
			tool_button[TOOL_RULER]->set_pressed(true);
			tool_mode = ToolMode::TOOL_RULER;
			update_transform_gizmo();
		} break;
	}
}

void Node3DEditor::_init_indicators() {
	{
		origin_enabled = true;
		grid_enabled = true;

		Ref<Shader> origin_shader = memnew(Shader);
		origin_shader->set_code(R"(
// 3D editor origin line shader.

shader_type spatial;
render_mode blend_mix, cull_disabled, unshaded, fog_disabled;

void vertex() {
	vec3 point_a = MODEL_MATRIX[3].xyz;
	// Encoded in scale.
	vec3 point_b = vec3(MODEL_MATRIX[0].x, MODEL_MATRIX[1].y, MODEL_MATRIX[2].z);

	// Points are already in world space, so no need for MODEL_MATRIX anymore.
	vec4 clip_a = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(point_a, 1.0));
	vec4 clip_b = PROJECTION_MATRIX * (VIEW_MATRIX * vec4(point_b, 1.0));

	vec2 screen_a = VIEWPORT_SIZE * (0.5 * clip_a.xy / clip_a.w + 0.5);
	vec2 screen_b = VIEWPORT_SIZE * (0.5 * clip_b.xy / clip_b.w + 0.5);

	vec2 x_basis = normalize(screen_b - screen_a);
	vec2 y_basis = vec2(-x_basis.y, x_basis.x);

	float width = 3.0;
	vec2 screen_point_a = screen_a + width * (VERTEX.x * x_basis + VERTEX.y * y_basis);
	vec2 screen_point_b = screen_b + width * (VERTEX.x * x_basis + VERTEX.y * y_basis);
	vec2 screen_point_final = mix(screen_point_a, screen_point_b, VERTEX.z);

	vec4 clip_final = mix(clip_a, clip_b, VERTEX.z);

	POSITION = vec4(clip_final.w * ((2.0 * screen_point_final) / VIEWPORT_SIZE - 1.0), clip_final.z, clip_final.w);
	UV = VERTEX.yz * clip_final.w;

	if (!OUTPUT_IS_SRGB) {
		COLOR.rgb = mix(pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), COLOR.rgb * (1.0 / 12.92), lessThan(COLOR.rgb, vec3(0.04045)));
	}
}

void fragment() {
	// Multiply by 0.5 since UV is actually UV is [-1, 1].
	float line_width = fwidth(UV.x * 0.5);
	float line_uv = abs(UV.x * 0.5);
	float line = smoothstep(line_width * 1.0, line_width * 0.25, line_uv);

	ALBEDO = COLOR.rgb;
	ALPHA *= COLOR.a * line;
}
)");

		origin_mat.instantiate();
		origin_mat->set_shader(origin_shader);

		Vector<Vector3> origin_points;
		origin_points.resize(6);

		origin_points.set(0, Vector3(0.0, -0.5, 0.0));
		origin_points.set(1, Vector3(0.0, -0.5, 1.0));
		origin_points.set(2, Vector3(0.0, 0.5, 1.0));

		origin_points.set(3, Vector3(0.0, -0.5, 0.0));
		origin_points.set(4, Vector3(0.0, 0.5, 1.0));
		origin_points.set(5, Vector3(0.0, 0.5, 0.0));

		Array d;
		d.resize(RSE::ARRAY_MAX);
		d[RSE::ARRAY_VERTEX] = origin_points;

		origin_mesh = RenderingServer::get_singleton()->mesh_create();

		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(origin_mesh, RSE::PRIMITIVE_TRIANGLES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(origin_mesh, 0, origin_mat->get_rid());

		origin_multimesh = RenderingServer::get_singleton()->multimesh_create();
		RenderingServer::get_singleton()->multimesh_set_mesh(origin_multimesh, origin_mesh);
		RenderingServer::get_singleton()->multimesh_allocate_data(origin_multimesh, 12, RSE::MultimeshTransformFormat::MULTIMESH_TRANSFORM_3D, true, false);
		RenderingServer::get_singleton()->multimesh_set_visible_instances(origin_multimesh, -1);

		LocalVector<float> distances;
		distances.resize(5);
		distances[0] = -1000000.0;
		distances[1] = -1000.0;
		distances[2] = 0.0;
		distances[3] = 1000.0;
		distances[4] = 1000000.0;

		for (int i = 0; i < 3; i++) {
			Color origin_color;
			switch (i) {
				case 0:
					origin_color = get_theme_color(SNAME("axis_x_color"), EditorStringName(Editor));
					break;
				case 1:
					origin_color = get_theme_color(SNAME("axis_y_color"), EditorStringName(Editor));
					break;
				case 2:
					origin_color = get_theme_color(SNAME("axis_z_color"), EditorStringName(Editor));
					break;
				default:
					origin_color = Color();
					break;
			}

			Vector3 axis;
			axis[i] = 1;

			for (int j = 0; j < 4; j++) {
				Transform3D t = Transform3D();
				if (distances[j] > 0.0) {
					t = t.scaled(axis * distances[j + 1]);
					t = t.translated(axis * distances[j]);
				} else {
					t = t.scaled(axis * distances[j]);
					t = t.translated(axis * distances[j + 1]);
				}
				RenderingServer::get_singleton()->multimesh_instance_set_transform(origin_multimesh, i * 4 + j, t);
				RenderingServer::get_singleton()->multimesh_instance_set_color(origin_multimesh, i * 4 + j, origin_color);
			}
		}

		origin_instance = RenderingServer::get_singleton()->instance_create2(origin_multimesh, get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_layer_mask(origin_instance, 1 << Node3DEditorViewport::GIZMO_GRID_LAYER);
		RS::get_singleton()->instance_geometry_set_flag(origin_instance, RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(origin_instance, RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		RenderingServer::get_singleton()->instance_geometry_set_cast_shadows_setting(origin_instance, RSE::SHADOW_CASTING_SETTING_OFF);

		Ref<Shader> grid_shader = memnew(Shader);
		grid_shader->set_code(R"(
// 3D editor grid shader.

shader_type spatial;

render_mode unshaded, fog_disabled;

uniform bool orthogonal;
uniform float grid_size;

void vertex() {
	// From FLAG_SRGB_VERTEX_COLOR.
	if (!OUTPUT_IS_SRGB) {
		COLOR.rgb = mix(pow((COLOR.rgb + vec3(0.055)) * (1.0 / (1.0 + 0.055)), vec3(2.4)), COLOR.rgb * (1.0 / 12.92), lessThan(COLOR.rgb, vec3(0.04045)));
	}
}

void fragment() {
	ALBEDO = COLOR.rgb;
	vec3 dir = orthogonal ? -vec3(0, 0, 1) : VIEW;
	float angle_fade = abs(dot(dir, NORMAL));
	angle_fade = smoothstep(0.05, 0.2, angle_fade);

	vec3 world_pos = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 world_normal = (INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz;
	vec3 camera_world_pos = INV_VIEW_MATRIX[3].xyz;
	vec3 camera_world_pos_on_plane = camera_world_pos * (1.0 - world_normal);
	float dist_fade = 1.0 - (distance(world_pos, camera_world_pos_on_plane) / grid_size);
	dist_fade = smoothstep(0.02, 0.3, dist_fade);

	ALPHA = COLOR.a * dist_fade * angle_fade;
}
)");

		for (int i = 0; i < 3; i++) {
			grid_mat[i].instantiate();
			grid_mat[i]->set_shader(grid_shader);
		}

		grid_enable[0] = EDITOR_GET("editors/3d/grid_xy_plane");
		grid_enable[1] = EDITOR_GET("editors/3d/grid_yz_plane");
		grid_enable[2] = EDITOR_GET("editors/3d/grid_xz_plane");
		grid_visible[0] = grid_enable[0];
		grid_visible[1] = grid_enable[1];
		grid_visible[2] = grid_enable[2];

		_init_grid();
	}

	{
		//move gizmo

		// Inverted zxy.
		Vector3 ivec = Vector3(0, 0, -1);
		Vector3 nivec = Vector3(-1, -1, 0);
		Vector3 ivec2 = Vector3(-1, 0, 0);
		Vector3 ivec3 = Vector3(0, -1, 0);

		for (int i = 0; i < 4; i++) {
			Color col;
			switch (i) {
				case 0:
					col = get_theme_color(SNAME("axis_x_color"), EditorStringName(Editor));
					break;
				case 1:
					col = get_theme_color(SNAME("axis_y_color"), EditorStringName(Editor));
					break;
				case 2:
					col = get_theme_color(SNAME("axis_z_color"), EditorStringName(Editor));
					break;
				case 3:
					col = get_theme_color(SNAME("axis_view_plane_color"), EditorStringName(Editor));
					break;
				default:
					col = Color();
					break;
			}

			col.a = col.a * (float)EDITOR_GET("editors/3d/manipulator_gizmo_opacity");

			if (i < 3) {
				move_gizmo[i].instantiate();
				move_plane_gizmo[i].instantiate();
				scale_gizmo[i].instantiate();
				scale_plane_gizmo[i].instantiate();
				axis_gizmo[i].instantiate();
			}

			rotate_gizmo[i].instantiate();

			const Color albedo = col.from_hsv(col.get_h(), col.get_s() * 0.25, 1.0, 1);

			Ref<StandardMaterial3D> mat;
			Ref<StandardMaterial3D> mat_hl;

			if (i < 3) {
				// Only create standard materials for X, Y, Z axes (move/scale gizmos).
				mat.instantiate();
				mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
				mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
				mat->set_on_top_of_alpha();
				mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
				mat->set_albedo(col);
				gizmo_color[i] = mat;

				mat_hl = mat->duplicate();
				mat_hl->set_albedo(albedo);
				gizmo_color_hl[i] = mat_hl;
			}

			// Only create translate gizmo for X, Y, Z axes (not view rotation).
			if (i < 3) {
				//translate
				{
					Ref<SurfaceTool> surftool;
					surftool.instantiate();
					surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

					// Arrow profile
					const int arrow_points = 5;
					Vector3 arrow[5] = {
						nivec * 0.0 + ivec * 0.0,
						nivec * 0.01 + ivec * 0.0,
						nivec * 0.01 + ivec * GIZMO_ARROW_OFFSET,
						nivec * 0.065 + ivec * GIZMO_ARROW_OFFSET,
						nivec * 0.0 + ivec * (GIZMO_ARROW_OFFSET + GIZMO_ARROW_SIZE),
					};

					int arrow_sides = 16;

					const real_t arrow_sides_step = Math::TAU / arrow_sides;
					for (int k = 0; k < arrow_sides; k++) {
						Basis ma(ivec, k * arrow_sides_step);
						Basis mb(ivec, (k + 1) * arrow_sides_step);

						for (int j = 0; j < arrow_points - 1; j++) {
							Vector3 points[4] = {
								ma.xform(arrow[j]),
								mb.xform(arrow[j]),
								mb.xform(arrow[j + 1]),
								ma.xform(arrow[j + 1]),
							};
							surftool->add_vertex(points[0]);
							surftool->add_vertex(points[1]);
							surftool->add_vertex(points[2]);

							surftool->add_vertex(points[0]);
							surftool->add_vertex(points[2]);
							surftool->add_vertex(points[3]);
						}
					}

					surftool->set_material(mat);
					surftool->commit(move_gizmo[i]);
				}

				// Plane Translation
				{
					Ref<SurfaceTool> surftool;
					surftool.instantiate();
					surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

					Vector3 vec = ivec2 - ivec3;
					Vector3 plane[4] = {
						vec * GIZMO_PLANE_DST,
						vec * GIZMO_PLANE_DST + ivec2 * GIZMO_PLANE_SIZE,
						vec * (GIZMO_PLANE_DST + GIZMO_PLANE_SIZE),
						vec * GIZMO_PLANE_DST - ivec3 * GIZMO_PLANE_SIZE
					};

					Basis ma(ivec, Math::PI / 2);

					Vector3 points[4] = {
						ma.xform(plane[0]),
						ma.xform(plane[1]),
						ma.xform(plane[2]),
						ma.xform(plane[3]),
					};
					surftool->add_vertex(points[0]);
					surftool->add_vertex(points[1]);
					surftool->add_vertex(points[2]);

					surftool->add_vertex(points[0]);
					surftool->add_vertex(points[2]);
					surftool->add_vertex(points[3]);

					Ref<StandardMaterial3D> plane_mat;
					plane_mat.instantiate();
					plane_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
					plane_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
					plane_mat->set_on_top_of_alpha();
					plane_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
					plane_mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
					plane_mat->set_albedo(col);
					plane_gizmo_color[i] = plane_mat; // Needed, so we can draw planes from both sides.
					surftool->set_material(plane_mat);
					surftool->commit(move_plane_gizmo[i]);

					Ref<StandardMaterial3D> plane_mat_hl = plane_mat->duplicate();
					plane_mat_hl->set_albedo(albedo);
					plane_gizmo_color_hl[i] = plane_mat_hl; // Needed, so we can draw planes from both sides.
				}
			}

			// Rotate - for all 4 indices (X, Y, Z, and view rotation).
			{
				Ref<SurfaceTool> surftool;
				surftool.instantiate();
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				int n = 128; // number of circle segments
				int m = 3; // number of thickness segments

				real_t step = Math::TAU / n;
				for (int j = 0; j < n; ++j) {
					Basis basis = Basis(ivec, j * step);

					Vector3 vertex = basis.xform(ivec2 * GIZMO_CIRCLE_SIZE);

					for (int k = 0; k < m; ++k) {
						Vector2 ofs = Vector2(Math::cos((Math::TAU * k) / m), Math::sin((Math::TAU * k) / m));
						Vector3 normal = ivec * ofs.x + ivec2 * ofs.y;

						surftool->set_normal(basis.xform(normal));
						surftool->add_vertex(vertex);
					}
				}

				for (int j = 0; j < n; ++j) {
					for (int k = 0; k < m; ++k) {
						int current_ring = j * m;
						int next_ring = ((j + 1) % n) * m;
						int current_segment = k;
						int next_segment = (k + 1) % m;

						surftool->add_index(current_ring + next_segment);
						surftool->add_index(current_ring + current_segment);
						surftool->add_index(next_ring + current_segment);

						surftool->add_index(next_ring + current_segment);
						surftool->add_index(next_ring + next_segment);
						surftool->add_index(current_ring + next_segment);
					}
				}

				Ref<Shader> rotate_shader = memnew(Shader);

				// Use special shader for view rotation (index 3) with camera-relative transformation.
				if (i == 3) {
					rotate_shader->set_code(R"(

shader_type spatial;

render_mode unshaded, depth_test_disabled, fog_disabled;

uniform vec4 albedo;

mat3 orthonormalize(mat3 m) {
	vec3 x = normalize(m[0]);
	vec3 y = normalize(m[1] - x * dot(x, m[1]));
	vec3 z = m[2] - x * dot(x, m[2]);
	z = normalize(z - y * (dot(y, m[2])));
	return mat3(x,y,z);
}

void vertex() {
	mat3 mv = orthonormalize(mat3(MODELVIEW_MATRIX));
	mv = inverse(mv);
	VERTEX += NORMAL * 0.008;
	vec3 camera_dir_local = mv * vec3(0.0, 0.0, 1.0);
	vec3 camera_up_local = mv * vec3(0.0, 1.0, 0.0);
	mat3 rotation_matrix = mat3(cross(camera_dir_local, camera_up_local), camera_up_local, camera_dir_local);
	VERTEX = rotation_matrix * VERTEX;
}

void fragment() {
	ALBEDO = albedo.rgb;
	ALPHA = albedo.a;
}
)");
				} else {
					// Standard shader for X, Y, Z rotation gizmos.
					rotate_shader->set_code(R"(
// 3D editor rotation manipulator gizmo shader.

shader_type spatial;

render_mode unshaded, depth_test_disabled, fog_disabled;

uniform vec4 albedo;

mat3 orthonormalize(mat3 m) {
	vec3 x = normalize(m[0]);
	vec3 y = normalize(m[1] - x * dot(x, m[1]));
	vec3 z = m[2] - x * dot(x, m[2]);
	z = normalize(z - y * (dot(y, m[2])));
	return mat3(x, y, z);
}

void vertex() {
	mat3 mv = orthonormalize(mat3(MODELVIEW_MATRIX));
	vec3 n = mv * VERTEX;
	float orientation = dot(vec3(0.0, 0.0, -1.0), n);
	if (orientation <= 0.005) {
		VERTEX += NORMAL * 0.02;
	}
}

void fragment() {
	ALBEDO = albedo.rgb;
	ALPHA = albedo.a;
}
)");
				}

				Ref<ShaderMaterial> rotate_mat;
				rotate_mat.instantiate();
				rotate_mat->set_render_priority(Material::RENDER_PRIORITY_MAX);
				rotate_mat->set_shader(rotate_shader);
				rotate_mat->set_shader_parameter("albedo", col);
				rotate_gizmo_color[i] = rotate_mat;

				Array arrays = surftool->commit_to_arrays();
				rotate_gizmo[i]->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
				rotate_gizmo[i]->surface_set_material(0, rotate_mat);

				Ref<ShaderMaterial> rotate_mat_hl = rotate_mat->duplicate();
				rotate_mat_hl->set_shader_parameter("albedo", albedo);
				rotate_gizmo_color_hl[i] = rotate_mat_hl;
			}

			// Only create scale gizmo for X, Y, Z axes (not view rotation).
			if (i < 3) {
				// Scale.
				{
					Ref<SurfaceTool> surftool;
					surftool.instantiate();
					surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

					// Cube arrow profile.
					const int arrow_points = 6;
					Vector3 arrow[6] = {
						nivec * 0.0 + ivec * 0.0,
						nivec * 0.01 + ivec * 0.0,
						nivec * 0.01 + ivec * 1.0 * GIZMO_SCALE_OFFSET,
						nivec * 0.07 + ivec * 1.0 * GIZMO_SCALE_OFFSET,
						nivec * 0.07 + ivec * 1.11 * GIZMO_SCALE_OFFSET,
						nivec * 0.0 + ivec * 1.11 * GIZMO_SCALE_OFFSET,
					};

					int arrow_sides = 4;

					const real_t arrow_sides_step = Math::TAU / arrow_sides;
					for (int k = 0; k < 4; k++) {
						Basis ma(ivec, k * arrow_sides_step);
						Basis mb(ivec, (k + 1) * arrow_sides_step);

						for (int j = 0; j < arrow_points - 1; j++) {
							Vector3 points[4] = {
								ma.xform(arrow[j]),
								mb.xform(arrow[j]),
								mb.xform(arrow[j + 1]),
								ma.xform(arrow[j + 1]),
							};
							surftool->add_vertex(points[0]);
							surftool->add_vertex(points[1]);
							surftool->add_vertex(points[2]);

							surftool->add_vertex(points[0]);
							surftool->add_vertex(points[2]);
							surftool->add_vertex(points[3]);
						}
					}

					surftool->set_material(mat);
					surftool->commit(scale_gizmo[i]);
				}

				// Plane Scale.
				{
					Ref<SurfaceTool> surftool;
					surftool.instantiate();
					surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

					Vector3 vec = ivec2 - ivec3;
					Vector3 plane[4] = {
						vec * GIZMO_PLANE_DST,
						vec * GIZMO_PLANE_DST + ivec2 * GIZMO_PLANE_SIZE,
						vec * (GIZMO_PLANE_DST + GIZMO_PLANE_SIZE),
						vec * GIZMO_PLANE_DST - ivec3 * GIZMO_PLANE_SIZE
					};

					Basis ma(ivec, Math::PI / 2);

					Vector3 points[4] = {
						ma.xform(plane[0]),
						ma.xform(plane[1]),
						ma.xform(plane[2]),
						ma.xform(plane[3]),
					};
					surftool->add_vertex(points[0]);
					surftool->add_vertex(points[1]);
					surftool->add_vertex(points[2]);

					surftool->add_vertex(points[0]);
					surftool->add_vertex(points[2]);
					surftool->add_vertex(points[3]);

					Ref<StandardMaterial3D> plane_mat;
					plane_mat.instantiate();
					plane_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
					plane_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
					plane_mat->set_on_top_of_alpha();
					plane_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
					plane_mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
					plane_mat->set_albedo(col);
					plane_gizmo_color[i] = plane_mat; // needed, so we can draw planes from both sides.
					surftool->set_material(plane_mat);
					surftool->commit(scale_plane_gizmo[i]);

					Ref<StandardMaterial3D> plane_mat_hl = plane_mat->duplicate();
					plane_mat_hl->set_albedo(col.from_hsv(col.get_h(), col.get_s() * 0.25, 1.0, 1));
					plane_gizmo_color_hl[i] = plane_mat_hl; // needed, so we can draw planes from both sides.
				}

				// Lines to visualize transforms locked to an axis/plane.
				{
					Ref<SurfaceTool> surftool;
					surftool.instantiate();
					surftool->begin(Mesh::PRIMITIVE_LINE_STRIP);

					Vector3 vec;
					vec[i] = 1;

					// Line extending through like infinity.
					surftool->add_vertex(vec * -1048576);
					surftool->add_vertex(Vector3());
					surftool->add_vertex(vec * 1048576);
					surftool->set_material(mat_hl);
					surftool->commit(axis_gizmo[i]);
				}
			}
		}
	}

	// Create trackball sphere
	{
		trackball_sphere_gizmo.instantiate();
		Ref<SurfaceTool> surftool;
		surftool.instantiate();
		surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

		const int sphere_rings = TRACKBALL_SPHERE_RINGS;
		const int sphere_sectors = TRACKBALL_SPHERE_SECTORS;
		const real_t sphere_radius = GIZMO_CIRCLE_SIZE;

		for (int r = 0; r <= sphere_rings; ++r) {
			for (int s = 0; s <= sphere_sectors; ++s) {
				real_t ring_angle = Math::PI * r / sphere_rings;
				real_t sector_angle = 2.0 * Math::PI * s / sphere_sectors;

				Vector3 vertex(
						sphere_radius * Math::sin(ring_angle) * Math::cos(sector_angle),
						sphere_radius * Math::cos(ring_angle),
						sphere_radius * Math::sin(ring_angle) * Math::sin(sector_angle));

				surftool->set_normal(vertex.normalized());
				surftool->add_vertex(vertex);
			}
		}

		for (int r = 0; r < sphere_rings; ++r) {
			for (int s = 0; s < sphere_sectors; ++s) {
				int current = r * (sphere_sectors + 1) + s;
				int next = current + sphere_sectors + 1;

				surftool->add_index(current);
				surftool->add_index(next);
				surftool->add_index(current + 1);

				surftool->add_index(current + 1);
				surftool->add_index(next);
				surftool->add_index(next + 1);
			}
		}

		trackball_sphere_material.instantiate();
		trackball_sphere_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		trackball_sphere_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
		trackball_sphere_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
		trackball_sphere_material->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
		trackball_sphere_material->set_albedo(Color(1.0, 1.0, 1.0, 0.0));
		trackball_sphere_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);

		trackball_sphere_material_hl = trackball_sphere_material->duplicate();
		trackball_sphere_material_hl->set_albedo(Color(1.0, 1.0, 1.0, TRACKBALL_HIGHLIGHT_ALPHA));

		surftool->set_material(trackball_sphere_material);
		surftool->commit(trackball_sphere_gizmo);
	}

	_generate_selection_boxes();
}

void Node3DEditor::_update_gizmos_menu() {
	gizmos_menu->clear();

	for (int i = 0; i < gizmo_plugins_by_name.size(); ++i) {
		if (!gizmo_plugins_by_name[i]->can_be_hidden()) {
			continue;
		}
		String plugin_name = gizmo_plugins_by_name[i]->get_gizmo_name();
		const int plugin_state = gizmo_plugins_by_name[i]->get_state();
		gizmos_menu->add_multistate_item(plugin_name, 3, plugin_state, i);
		const int idx = gizmos_menu->get_item_index(i);
		gizmos_menu->set_item_tooltip(
				idx,
				TTR("Click to toggle between visibility states.\n\nOpen eye: Gizmo is visible.\nClosed eye: Gizmo is hidden.\nHalf-open eye: Gizmo is also visible through opaque surfaces (\"x-ray\")."));
		switch (plugin_state) {
			case EditorNode3DGizmoPlugin::VISIBLE:
				gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityVisible")));
				break;
			case EditorNode3DGizmoPlugin::ON_TOP:
				gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityXray")));
				break;
			case EditorNode3DGizmoPlugin::HIDDEN:
				gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityHidden")));
				break;
		}
	}
}

void Node3DEditor::_update_gizmos_menu_theme() {
	for (int i = 0; i < gizmo_plugins_by_name.size(); ++i) {
		if (!gizmo_plugins_by_name[i]->can_be_hidden()) {
			continue;
		}
		const int plugin_state = gizmo_plugins_by_name[i]->get_state();
		const int idx = gizmos_menu->get_item_index(i);
		switch (plugin_state) {
			case EditorNode3DGizmoPlugin::VISIBLE:
				gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityVisible")));
				break;
			case EditorNode3DGizmoPlugin::ON_TOP:
				gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityXray")));
				break;
			case EditorNode3DGizmoPlugin::HIDDEN:
				gizmos_menu->set_item_icon(idx, get_editor_theme_icon(SNAME("GuiVisibilityHidden")));
				break;
		}
	}
}

void Node3DEditor::_init_grid() {
	if (!grid_enabled) {
		return;
	}
	Camera3D *camera = get_editor_viewport(0)->camera;
	Vector3 camera_position = camera->get_position();
	if (camera_position == Vector3()) {
		return; // Camera3D is invalid, don't draw the grid.
	}

	bool orthogonal = camera->get_projection() == Camera3D::PROJECTION_ORTHOGONAL;

	static LocalVector<Color> grid_colors[3];
	static LocalVector<Vector3> grid_points[3];
	static LocalVector<Vector3> grid_normals[3];

	for (uint32_t n = 0; n < 3; n++) {
		grid_colors[n].clear();
		grid_points[n].clear();
		grid_normals[n].clear();
	}

	Color primary_grid_color = EDITOR_GET("editors/3d/primary_grid_color");
	Color secondary_grid_color = EDITOR_GET("editors/3d/secondary_grid_color");
	int grid_size = EDITOR_GET("editors/3d/grid_size");
	int primary_grid_steps = EDITOR_GET("editors/3d/primary_grid_steps");

	// Which grid planes are enabled? Which should we generate?
	grid_enable[0] = grid_visible[0] = orthogonal || EDITOR_GET("editors/3d/grid_xy_plane");
	grid_enable[1] = grid_visible[1] = orthogonal || EDITOR_GET("editors/3d/grid_yz_plane");
	grid_enable[2] = grid_visible[2] = orthogonal || EDITOR_GET("editors/3d/grid_xz_plane");

	// Offsets division_level for bigger or smaller grids.
	// Default value is -0.2. -1.0 gives Blender-like behavior, 0.5 gives huge grids.
	real_t division_level_bias = EDITOR_GET("editors/3d/grid_division_level_bias");
	// Default largest grid size is 8^2 (default value is 2) when primary_grid_steps is 8 (64m apart, so primary grid lines are 512m apart).
	int division_level_max = EDITOR_GET("editors/3d/grid_division_level_max");
	// Default smallest grid size is 8^0 (default value is 0) when primary_grid_steps is 8.
	int division_level_min = EDITOR_GET("editors/3d/grid_division_level_min");
	ERR_FAIL_COND_MSG(division_level_max < division_level_min, "The 3D grid's maximum division level cannot be lower than its minimum division level.");

	if (primary_grid_steps != 10) { // Log10 of 10 is 1.
		// Change of base rule, divide by ln(10).
		real_t div = Math::log((real_t)primary_grid_steps) / (real_t)2.302585092994045901094;
		// Truncation (towards zero) is intentional.
		division_level_max = (int)(division_level_max / div);
		division_level_min = (int)(division_level_min / div);
	}

	for (int a = 0; a < 3; a++) {
		if (!grid_enable[a]) {
			continue; // If this grid plane is disabled, skip generation.
		}
		int b = (a + 1) % 3;
		int c = (a + 2) % 3;

		Vector3 normal;
		normal[c] = 1.0;

		real_t camera_distance = Math::abs(camera_position[c]);

		if (orthogonal) {
			camera_distance = camera->get_size() / 2.0;
			Vector3 camera_direction = -camera->get_global_transform().get_basis().get_column(2);
			Plane grid_plane = Plane(normal);
			Vector3 intersection;
			if (grid_plane.intersects_ray(camera_position, camera_direction, &intersection)) {
				camera_position = intersection;
			}
		}

		real_t division_level = Math::log(Math::abs(camera_distance)) / Math::log((double)primary_grid_steps) + division_level_bias;

		real_t clamped_division_level = CLAMP(division_level, division_level_min, division_level_max);
		real_t division_level_floored = Math::floor(clamped_division_level);
		real_t division_level_decimals = clamped_division_level - division_level_floored;

		real_t small_step_size = Math::pow(primary_grid_steps, division_level_floored);
		real_t large_step_size = small_step_size * primary_grid_steps;
		real_t center_a = large_step_size * (int)(camera_position[a] / large_step_size);
		real_t center_b = large_step_size * (int)(camera_position[b] / large_step_size);

		real_t bgn_a = center_a - grid_size * small_step_size;
		real_t end_a = center_a + grid_size * small_step_size;
		real_t bgn_b = center_b - grid_size * small_step_size;
		real_t end_b = center_b + grid_size * small_step_size;

		real_t fade_size = Math::pow(primary_grid_steps, division_level - 1.0);
		real_t min_fade_size = Math::pow(primary_grid_steps, float(division_level_min));
		real_t max_fade_size = Math::pow(primary_grid_steps, float(division_level_max));
		fade_size = CLAMP(fade_size, min_fade_size, max_fade_size);

		real_t grid_fade_size = (grid_size - primary_grid_steps) * fade_size;
		grid_mat[c]->set_shader_parameter("grid_size", grid_fade_size);
		grid_mat[c]->set_shader_parameter("orthogonal", orthogonal);

		LocalVector<Vector3> &ref_grid = grid_points[c];
		LocalVector<Vector3> &ref_grid_normals = grid_normals[c];
		LocalVector<Color> &ref_grid_colors = grid_colors[c];

		// Count our elements same as code below it.
		int expected_size = 0;
		for (int i = -grid_size; i <= grid_size; i++) {
			const real_t position_a = center_a + i * small_step_size;
			const real_t position_b = center_b + i * small_step_size;

			// Don't draw lines over the origin if it's enabled.
			if (!(origin_enabled && Math::is_zero_approx(position_a))) {
				expected_size += 2;
			}

			if (!(origin_enabled && Math::is_zero_approx(position_b))) {
				expected_size += 2;
			}
		}

		int idx = 0;
		ref_grid.resize(expected_size);
		ref_grid_normals.resize(expected_size);
		ref_grid_colors.resize(expected_size);

		// In each iteration of this loop, draw one line in each direction (so two lines per loop, in each if statement).
		for (int i = -grid_size; i <= grid_size; i++) {
			Color line_color;
			// Is this a primary line? Set the appropriate color.
			if (i % primary_grid_steps == 0) {
				line_color = primary_grid_color.lerp(secondary_grid_color, division_level_decimals);
			} else {
				line_color = secondary_grid_color;
				line_color.a = line_color.a * (1 - division_level_decimals);
			}

			real_t position_a = center_a + i * small_step_size;
			real_t position_b = center_b + i * small_step_size;

			// Don't draw lines over the origin if it's enabled.
			if (!(origin_enabled && Math::is_zero_approx(position_a))) {
				Vector3 line_bgn;
				Vector3 line_end;
				line_bgn[a] = position_a;
				line_end[a] = position_a;
				line_bgn[b] = bgn_b;
				line_end[b] = end_b;
				ref_grid[idx] = line_bgn;
				ref_grid[idx + 1] = line_end;
				ref_grid_colors[idx] = line_color;
				ref_grid_colors[idx + 1] = line_color;
				ref_grid_normals[idx] = normal;
				ref_grid_normals[idx + 1] = normal;
				idx += 2;
			}

			if (!(origin_enabled && Math::is_zero_approx(position_b))) {
				Vector3 line_bgn;
				Vector3 line_end;
				line_bgn[b] = position_b;
				line_end[b] = position_b;
				line_bgn[a] = bgn_a;
				line_end[a] = end_a;
				ref_grid[idx] = line_bgn;
				ref_grid[idx + 1] = line_end;
				ref_grid_colors[idx] = line_color;
				ref_grid_colors[idx + 1] = line_color;
				ref_grid_normals[idx] = normal;
				ref_grid_normals[idx + 1] = normal;
				idx += 2;
			}
		}

		// Create a mesh from the pushed vector points and colors.
		grid[c] = RenderingServer::get_singleton()->mesh_create();
		Array d;
		d.resize(RSE::ARRAY_MAX);
		d[RSE::ARRAY_VERTEX] = (Vector<Vector3>)grid_points[c];
		d[RSE::ARRAY_COLOR] = (Vector<Color>)grid_colors[c];
		d[RSE::ARRAY_NORMAL] = (Vector<Vector3>)grid_normals[c];
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(grid[c], RSE::PRIMITIVE_LINES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(grid[c], 0, grid_mat[c]->get_rid());
		grid_instance[c] = RenderingServer::get_singleton()->instance_create2(grid[c], get_tree()->get_root()->get_world_3d()->get_scenario());

		// Yes, the end of this line is supposed to be a.
		RenderingServer::get_singleton()->instance_set_visible(grid_instance[c], grid_visible[a]);
		RenderingServer::get_singleton()->instance_geometry_set_cast_shadows_setting(grid_instance[c], RSE::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(grid_instance[c], 1 << Node3DEditorViewport::GIZMO_GRID_LAYER);
		RS::get_singleton()->instance_geometry_set_flag(grid_instance[c], RSE::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(grid_instance[c], RSE::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	}
}

void Node3DEditor::_finish_indicators() {
	RenderingServer::get_singleton()->free_rid(origin_instance);
	RenderingServer::get_singleton()->free_rid(origin_multimesh);
	RenderingServer::get_singleton()->free_rid(origin_mesh);

	_finish_grid();
}

void Node3DEditor::_finish_grid() {
	for (int i = 0; i < 3; i++) {
		RenderingServer::get_singleton()->free_rid(grid_instance[i]);
		RenderingServer::get_singleton()->free_rid(grid[i]);
	}
}

void Node3DEditor::update_gizmo_opacity() {
	if (!origin_instance.is_valid()) {
		return;
	}

	const float opacity = EDITOR_GET("editors/3d/manipulator_gizmo_opacity");

	for (int i = 0; i < 3; i++) {
		Color col = gizmo_color[i]->get_albedo();
		col.a = opacity;
		gizmo_color[i]->set_albedo(col);

		col = gizmo_color_hl[i]->get_albedo();
		col.a = 1.0;
		gizmo_color_hl[i]->set_albedo(col);

		col = plane_gizmo_color[i]->get_albedo();
		col.a = opacity;
		plane_gizmo_color[i]->set_albedo(col);

		col = plane_gizmo_color_hl[i]->get_albedo();
		col.a = 1.0;
		plane_gizmo_color_hl[i]->set_albedo(col);
	}
}

void Node3DEditor::update_grid() {
	const Camera3D::ProjectionType current_projection = viewports[0]->camera->get_projection();

	if (current_projection != grid_camera_last_update_perspective) {
		grid_init_draw = false; // redraw
		grid_camera_last_update_perspective = current_projection;
	}

	// Gets a orthogonal or perspective position correctly (for the grid comparison)
	const Vector3 camera_position = get_editor_viewport(0)->camera->get_position();

	if (!grid_init_draw || grid_camera_last_update_position.distance_squared_to(camera_position) >= 100.0f) {
		_finish_grid();
		_init_grid();
		grid_init_draw = true;
		grid_camera_last_update_position = camera_position;
	}
}

void Node3DEditor::_selection_changed() {
	_refresh_menu_icons();

	const HashMap<ObjectID, Object *> &selection = editor_selection->get_selection();

	for (const KeyValue<ObjectID, Object *> &E : selection) {
		Node3D *sp = ObjectDB::get_instance<Node3D>(E.key);
		if (!sp) {
			continue;
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
		if (!se) {
			continue;
		}

		if (sp == editor_selection->get_top_selected_node_list().back()->get()) {
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance, active_selection_box->get_rid());
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance_xray, active_selection_box_xray->get_rid());
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance_offset, active_selection_box->get_rid());
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance_xray_offset, active_selection_box_xray->get_rid());
		} else {
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance, selection_box->get_rid());
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance_xray, selection_box_xray->get_rid());
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance_offset, selection_box->get_rid());
			RenderingServer::get_singleton()->instance_set_base(se->sbox_instance_xray_offset, selection_box_xray->get_rid());
		}
	}

	if (selected && editor_selection->get_top_selected_node_list().size() != 1) {
		Vector<Ref<Node3DGizmo>> gizmos = selected->get_gizmos();
		for (int i = 0; i < gizmos.size(); i++) {
			Ref<EditorNode3DGizmo> seg = gizmos[i];
			if (seg.is_null()) {
				continue;
			}
			seg->set_selected(false);
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected);
		if (se) {
			se->gizmo.unref();
			se->subgizmos.clear();
		}
		selected->update_gizmos();
		selected = nullptr;
	}

	// Ensure gizmo updates are performed when the selection changes
	// outside of the 3D view (see GH-106713).
	if (!is_visible()) {
		const List<Node *> &top_selected = editor_selection->get_top_selected_node_list();
		if (top_selected.size() == 1) {
			Node3D *new_selected = Object::cast_to<Node3D>(top_selected.back()->get());
			if (new_selected != selected) {
				gizmos_dirty = true;
			}
		}
	}

	update_transform_gizmo();
}

void Node3DEditor::refresh_dirty_gizmos() {
	if (!gizmos_dirty) {
		return;
	}

	const List<Node *> &top_selected = editor_selection->get_top_selected_node_list();
	if (top_selected.size() == 1) {
		Node3D *new_selected = Object::cast_to<Node3D>(top_selected.back()->get());
		if (new_selected != selected) {
			edit(new_selected);
		}
	}
	gizmos_dirty = false;
}

void Node3DEditor::_refresh_menu_icons() {
	bool all_locked = true;
	bool all_grouped = true;
	bool has_node3d_item = false;

	const List<Node *> &selection = editor_selection->get_top_selected_node_list();

	if (selection.is_empty()) {
		all_locked = false;
		all_grouped = false;
	} else {
		for (Node *E : selection) {
			Node3D *node = Object::cast_to<Node3D>(E);
			if (node) {
				if (all_locked && !node->has_meta("_edit_lock_")) {
					all_locked = false;
				}
				if (all_grouped && !node->has_meta("_edit_group_")) {
					all_grouped = false;
				}
				has_node3d_item = true;
			}
			if (!all_locked && !all_grouped) {
				break;
			}
		}
	}

	all_locked = all_locked && has_node3d_item;
	all_grouped = all_grouped && has_node3d_item;

	tool_button[TOOL_LOCK_SELECTED]->set_visible(!all_locked);
	tool_button[TOOL_LOCK_SELECTED]->set_disabled(!has_node3d_item);
	tool_button[TOOL_UNLOCK_SELECTED]->set_visible(all_locked);
	tool_button[TOOL_UNLOCK_SELECTED]->set_disabled(!has_node3d_item);

	tool_button[TOOL_GROUP_SELECTED]->set_visible(!all_grouped);
	tool_button[TOOL_GROUP_SELECTED]->set_disabled(!has_node3d_item);
	tool_button[TOOL_UNGROUP_SELECTED]->set_visible(all_grouped);
	tool_button[TOOL_UNGROUP_SELECTED]->set_disabled(!has_node3d_item);
}

template <typename T>
HashSet<T *> _get_child_nodes(Node *parent_node) {
	HashSet<T *> nodes = HashSet<T *>();
	T *node = Node::cast_to<T>(parent_node);
	if (node) {
		nodes.insert(node);
	}

	for (int i = 0; i < parent_node->get_child_count(); i++) {
		Node *child_node = parent_node->get_child(i);
		HashSet<T *> child_nodes = _get_child_nodes<T>(child_node);
		for (T *I : child_nodes) {
			nodes.insert(I);
		}
	}

	return nodes;
}

HashSet<RID> _get_physics_bodies_rid(Node *node) {
	HashSet<RID> rids = HashSet<RID>();
	PhysicsBody3D *pb = Node::cast_to<PhysicsBody3D>(node);
	if (pb) {
		rids.insert(pb->get_rid());
	}
	HashSet<PhysicsBody3D *> child_nodes = _get_child_nodes<PhysicsBody3D>(node);
	for (const PhysicsBody3D *I : child_nodes) {
		rids.insert(I->get_rid());
	}

	return rids;
}

void Node3DEditor::snap_selected_nodes_to_floor() {
	do_snap_selected_nodes_to_floor = true;
}

void Node3DEditor::_snap_selected_nodes_to_floor() {
	const List<Node *> &selection = editor_selection->get_top_selected_node_list();
	Dictionary snap_data;

	for (Node *E : selection) {
		Node3D *sp = Object::cast_to<Node3D>(E);
		if (sp) {
			Vector3 from;
			Vector3 position_offset;

			// Priorities for snapping to floor are CollisionShapes, VisualInstances and then origin
			HashSet<VisualInstance3D *> vi = _get_child_nodes<VisualInstance3D>(sp);
			HashSet<CollisionShape3D *> cs = _get_child_nodes<CollisionShape3D>(sp);
			bool found_valid_shape = false;

			if (cs.size()) {
				AABB aabb;
				HashSet<CollisionShape3D *>::Iterator I = cs.begin();
				if ((*I)->get_shape().is_valid()) {
					CollisionShape3D *collision_shape = *cs.begin();
					aabb = collision_shape->get_global_transform().xform(collision_shape->get_shape()->get_debug_mesh()->get_aabb());
					found_valid_shape = true;
				}

				for (++I; I; ++I) {
					CollisionShape3D *col_shape = *I;
					if (col_shape->get_shape().is_valid()) {
						aabb.merge_with(col_shape->get_global_transform().xform(col_shape->get_shape()->get_debug_mesh()->get_aabb()));
						found_valid_shape = true;
					}
				}
				if (found_valid_shape) {
					Vector3 size = aabb.size * Vector3(0.5, 0.0, 0.5);
					from = aabb.position + size;
					position_offset.y = from.y - sp->get_global_transform().origin.y;
				}
			}
			if (!found_valid_shape && vi.size()) {
				VisualInstance3D *begin = *vi.begin();
				AABB aabb = begin->get_global_transform().xform(begin->get_aabb());
				for (const VisualInstance3D *I : vi) {
					aabb.merge_with(I->get_global_transform().xform(I->get_aabb()));
				}
				Vector3 size = aabb.size * Vector3(0.5, 0.0, 0.5);
				from = aabb.position + size;
				position_offset.y = from.y - sp->get_global_transform().origin.y;
			} else if (!found_valid_shape) {
				from = sp->get_global_transform().origin;
			}

			// We add a bit of margin to the from position to avoid it from snapping
			// when the spatial is already on a floor and there's another floor under
			// it
			from = from + Vector3(0.0, 1, 0.0);

			Dictionary d;

			d["from"] = from;
			d["position_offset"] = position_offset;
			snap_data[sp] = d;
		}
	}

	PhysicsDirectSpaceState3D *ss = get_tree()->get_root()->get_world_3d()->get_direct_space_state();
	PS3DT::RayResult result;

	// The maximum height an object can travel to be snapped
	const float max_snap_height = 500.0;

	// Will be set to `true` if at least one node from the selection was successfully snapped
	bool snapped_to_floor = false;

	if (!snap_data.is_empty()) {
		// For snapping to be performed, there must be solid geometry under at least one of the selected nodes.
		// We need to check this before snapping to register the undo/redo action only if needed.
		for (const KeyValue<Variant, Variant> &kv : snap_data) {
			Node *node = Object::cast_to<Node>(kv.key);
			Node3D *sp = Object::cast_to<Node3D>(node);
			Dictionary d = kv.value;
			Vector3 from = d["from"];
			Vector3 to = from - Vector3(0.0, max_snap_height, 0.0);
			HashSet<RID> excluded = _get_physics_bodies_rid(sp);

			PS3DT::RayParameters ray_params;
			ray_params.from = from;
			ray_params.to = to;
			ray_params.exclude = excluded;

			if (ss->intersect_ray(ray_params, result)) {
				snapped_to_floor = true;
			}
		}

		if (snapped_to_floor) {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Snap Nodes to Floor"));

			// Perform snapping if at least one node can be snapped
			for (const KeyValue<Variant, Variant> &kv : snap_data) {
				Node *node = Object::cast_to<Node>(kv.key);
				Node3D *sp = Object::cast_to<Node3D>(node);
				Dictionary d = kv.value;
				Vector3 from = d["from"];
				Vector3 to = from - Vector3(0.0, max_snap_height, 0.0);
				HashSet<RID> excluded = _get_physics_bodies_rid(sp);

				PS3DT::RayParameters ray_params;
				ray_params.from = from;
				ray_params.to = to;
				ray_params.exclude = excluded;

				if (ss->intersect_ray(ray_params, result)) {
					Vector3 position_offset = d["position_offset"];
					Transform3D new_transform = sp->get_global_transform();

					new_transform.origin.y = result.position.y;
					new_transform.origin = new_transform.origin - position_offset;

					Node3D *parent = sp->get_parent_node_3d();
					Transform3D new_local_xform = parent ? parent->get_global_transform().affine_inverse() * new_transform : new_transform;
					undo_redo->add_do_method(sp, "set_transform", new_local_xform);
					undo_redo->add_undo_method(sp, "set_transform", sp->get_transform());
				}
			}

			undo_redo->commit_action();
		} else {
			EditorNode::get_singleton()->show_warning(TTR("Couldn't find a solid floor to snap the selection to."));
		}
	}
}

void Node3DEditor::shortcut_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!is_visible_in_tree()) {
		return;
	}

	snap_key_enabled = Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL);
}

void Node3DEditor::_sun_environ_settings_pressed() {
	Vector2 pos = sun_environ_settings->get_screen_position() + sun_environ_settings->get_size();
	sun_environ_popup->set_position(pos - Vector2(sun_environ_popup->get_contents_minimum_size().width / 2, 0));
	sun_environ_popup->reset_size();
	sun_environ_popup->popup();
	// Grabbing the focus is required for Shift modifier checking to be functional
	// (when the Add sun/environment buttons are pressed).
	sun_environ_popup->grab_focus();
}

void Node3DEditor::_add_sun_to_scene(bool p_already_added_environment) {
	sun_environ_popup->hide();

	if (!p_already_added_environment && world_env_count == 0 && Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		// Prevent infinite feedback loop between the sun and environment methods.
		_add_environment_to_scene(true);
	}

	Node *base = get_tree()->get_edited_scene_root();
	if (!base) {
		// Create a root node so we can add child nodes to it.
		SceneTreeDock::get_singleton()->add_root_node(memnew(Node3D));
		base = get_tree()->get_edited_scene_root();
	}
	ERR_FAIL_NULL(base);
	Node *new_sun = preview_sun->duplicate();

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Preview Sun to Scene"));
	undo_redo->add_do_method(base, "add_child", new_sun, true);
	// Move to the beginning of the scene tree since more "global" nodes
	// generally look better when placed at the top.
	undo_redo->add_do_method(base, "move_child", new_sun, 0);
	undo_redo->add_do_method(new_sun, "set_owner", base);
	undo_redo->add_undo_method(base, "remove_child", new_sun);
	undo_redo->add_do_reference(new_sun);
	undo_redo->commit_action();
}

void Node3DEditor::_add_environment_to_scene(bool p_already_added_sun) {
	sun_environ_popup->hide();

	if (!p_already_added_sun && directional_light_count == 0 && Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		// Prevent infinite feedback loop between the sun and environment methods.
		_add_sun_to_scene(true);
	}

	Node *base = get_tree()->get_edited_scene_root();
	if (!base) {
		// Create a root node so we can add child nodes to it.
		SceneTreeDock::get_singleton()->add_root_node(memnew(Node3D));
		base = get_tree()->get_edited_scene_root();
	}
	ERR_FAIL_NULL(base);

	WorldEnvironment *new_env = memnew(WorldEnvironment);
	new_env->set_environment(preview_environment->get_environment()->duplicate(true));
	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		new_env->set_camera_attributes(preview_environment->get_camera_attributes()->duplicate(true));
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Add Preview Environment to Scene"));
	undo_redo->add_do_method(base, "add_child", new_env, true);
	// Move to the beginning of the scene tree since more "global" nodes
	// generally look better when placed at the top.
	undo_redo->add_do_method(base, "move_child", new_env, 0);
	undo_redo->add_do_method(new_env, "set_owner", base);
	undo_redo->add_undo_method(base, "remove_child", new_env);
	undo_redo->add_do_reference(new_env);
	undo_redo->commit_action();
}

void Node3DEditor::_update_theme() {
	tool_button[TOOL_MODE_TRANSFORM]->set_button_icon(get_editor_theme_icon(SNAME("ToolTransform")));
	tool_button[TOOL_MODE_MOVE]->set_button_icon(get_editor_theme_icon(SNAME("ToolMove")));
	tool_button[TOOL_MODE_ROTATE]->set_button_icon(get_editor_theme_icon(SNAME("ToolRotate")));
	tool_button[TOOL_MODE_SCALE]->set_button_icon(get_editor_theme_icon(SNAME("ToolScale")));
	tool_button[TOOL_MODE_SELECT]->set_button_icon(get_editor_theme_icon(SNAME("ToolSelect")));
	tool_button[TOOL_MODE_LIST_SELECT]->set_button_icon(get_editor_theme_icon(SNAME("ListSelect")));
	tool_button[TOOL_LOCK_SELECTED]->set_button_icon(get_editor_theme_icon(SNAME("Lock")));
	tool_button[TOOL_UNLOCK_SELECTED]->set_button_icon(get_editor_theme_icon(SNAME("Unlock")));
	tool_button[TOOL_GROUP_SELECTED]->set_button_icon(get_editor_theme_icon(SNAME("Group")));
	tool_button[TOOL_UNGROUP_SELECTED]->set_button_icon(get_editor_theme_icon(SNAME("Ungroup")));
	tool_button[TOOL_RULER]->set_button_icon(get_editor_theme_icon(SNAME("Ruler")));

	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_button_icon(get_editor_theme_icon(SNAME("Object")));
	tool_option_button[TOOL_OPT_USE_SNAP]->set_button_icon(get_editor_theme_icon(SNAME("Snap")));
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_button_icon(get_editor_theme_icon(SNAME("Trackball")));
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_button_icon(get_editor_theme_icon(SNAME("Pin")));

	view_layout_menu->get_popup()->set_item_icon(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), get_editor_theme_icon(SNAME("Panels1")));
	view_layout_menu->get_popup()->set_item_icon(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), get_editor_theme_icon(SNAME("Panels2")));
	view_layout_menu->get_popup()->set_item_icon(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), get_editor_theme_icon(SNAME("Panels2Alt")));
	view_layout_menu->get_popup()->set_item_icon(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), get_editor_theme_icon(SNAME("Panels3")));
	view_layout_menu->get_popup()->set_item_icon(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), get_editor_theme_icon(SNAME("Panels3Alt")));
	view_layout_menu->get_popup()->set_item_icon(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), get_editor_theme_icon(SNAME("Panels4")));

	sun_button->set_button_icon(get_editor_theme_icon(SNAME("PreviewSun")));
	environ_button->set_button_icon(get_editor_theme_icon(SNAME("PreviewEnvironment")));
	sun_environ_settings->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));

	sun_title->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("title_font"), SNAME("Window")));
	environ_title->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("title_font"), SNAME("Window")));

	sun_color->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("inspector_property_height"), EditorStringName(Editor))));
	environ_sky_color->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("inspector_property_height"), EditorStringName(Editor))));
	environ_ground_color->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("inspector_property_height"), EditorStringName(Editor))));

	context_toolbar_panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("ContextualToolbar"), EditorStringName(EditorStyles)));
}

void Node3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			const String show_list_tooltip = TTR("Alt+RMB: Show list of all nodes at position clicked, including locked.");
			tool_button[TOOL_MODE_TRANSFORM]->set_tooltip_text(vformat(TTR("%s+Drag: Rotate selected node around pivot."), keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL)) + "\n" + show_list_tooltip);
			tool_button[TOOL_MODE_MOVE]->set_tooltip_text(vformat(TTR("%s+Drag: Use snap."), keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL)) + "\n" + show_list_tooltip);
			tool_button[TOOL_MODE_ROTATE]->set_tooltip_text(vformat(TTR("%s+Drag: Use snap."), keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL)) + "\n" + show_list_tooltip);
			tool_button[TOOL_MODE_SCALE]->set_tooltip_text(vformat(TTR("%s+Drag: Use snap."), keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL)) + "\n" + show_list_tooltip);
			tool_button[TOOL_MODE_SELECT]->set_tooltip_text(show_list_tooltip);
			tool_button[TOOL_MODE_LIST_SELECT]->set_tooltip_text(TTR("Show list of selectable nodes at position clicked.") + "\n" + show_list_tooltip);
			tool_button[TOOL_RULER]->set_tooltip_text(TTR("LMB+Drag: Measure the distance between two points in 3D space.") + "\n" + TTR("Shift+LMB+Drag: Show component measurements.") + "\n" + show_list_tooltip);
			_update_gizmos_menu();
			_update_vertex_snap_tooltips();
		} break;

		case NOTIFICATION_READY: {
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);

			_refresh_menu_icons();

			get_tree()->connect("node_removed", callable_mp(this, &Node3DEditor::_node_removed));
			get_tree()->connect("node_added", callable_mp(this, &Node3DEditor::_node_added));
			SceneTreeDock::get_singleton()->get_tree_editor()->connect("node_changed", callable_mp(this, &Node3DEditor::_refresh_menu_icons));
			editor_selection->connect("selection_changed", callable_mp(this, &Node3DEditor::_selection_changed));

			_update_preview_environment();

			sun_state->set_custom_minimum_size(sun_vb->get_combined_minimum_size());
			environ_state->set_custom_minimum_size(environ_vb->get_combined_minimum_size());

			ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &Node3DEditor::update_all_gizmos).bind(Variant()));
		} break;

		case NOTIFICATION_ENTER_TREE: {
			_update_theme();
			_register_all_gizmos();
			_init_indicators();
			update_all_gizmos();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_finish_indicators();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
			_update_gizmos_menu_theme();
			sun_title->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("title_font"), SNAME("Window")));
			environ_title->add_theme_font_override(SceneStringName(font), get_theme_font(SNAME("title_font"), SNAME("Window")));
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (EditorSettings::get_singleton()->check_changed_settings_in_group("editors/3d")) {
				const Color selection_box_color = EDITOR_GET("editors/3d/selection_box_color");
				const Color active_selection_box_color = EDITOR_GET("editors/3d/active_selection_box_color");

				if (selection_box_color != selection_box_mat->get_albedo()) {
					selection_box_mat->set_albedo(selection_box_color);
					selection_box_mat_xray->set_albedo(selection_box_color * Color(1, 1, 1, 0.15));
				}

				if (active_selection_box_color != active_selection_box_mat->get_albedo()) {
					active_selection_box_mat->set_albedo(active_selection_box_color);
					active_selection_box_mat_xray->set_albedo(active_selection_box_color * Color(1, 1, 1, 0.15));
				}

				gizmo_view_rotation_scale = GIZMO_CIRCLE_SIZE * (float)EDITOR_GET("editors/3d/view_plane_rotation_gizmo_scale");

				// Update grid color by rebuilding grid.
				_finish_grid();
				_init_grid();

				for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
					viewports[i]->update_transform_gizmo_view();
				}
				update_gizmo_opacity();
			}
			if (EditorSettings::get_singleton()->check_changed_settings_in_group("editors/3d_gizmos/gizmo_settings")) {
				CollisionShape3DGizmoPlugin::set_show_only_when_selected(EDITOR_GET("editors/3d_gizmos/gizmo_settings/show_collision_shapes_only_when_selected"));
				update_all_gizmos();
			}
			_update_vertex_snap_tooltips();
		} break;

		case NOTIFICATION_PHYSICS_PROCESS: {
			if (do_snap_selected_nodes_to_floor) {
				_snap_selected_nodes_to_floor();
				do_snap_selected_nodes_to_floor = false;
			}
		}
	}
}

bool Node3DEditor::is_subgizmo_selected(int p_id) {
	Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;
	if (se) {
		return se->subgizmos.has(p_id);
	}
	return false;
}

bool Node3DEditor::is_current_selected_gizmo(const EditorNode3DGizmo *p_gizmo) {
	Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;
	if (se) {
		return se->gizmo == p_gizmo;
	}
	return false;
}

Vector<int> Node3DEditor::get_subgizmo_selection() {
	Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;

	Vector<int> ret;
	if (se) {
		for (const KeyValue<int, Transform3D> &E : se->subgizmos) {
			ret.push_back(E.key);
		}
	}
	return ret;
}

void Node3DEditor::clear_subgizmo_selection(Object *p_obj) {
	_clear_subgizmo_selection(p_obj);
}

void Node3DEditor::add_control_to_menu_panel(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND(p_control->get_parent());

	VSeparator *sep = memnew(VSeparator);
	context_toolbar_hbox->add_child(sep);
	context_toolbar_hbox->add_child(p_control);
	context_toolbar_separators[p_control] = sep;

	p_control->connect(SceneStringName(visibility_changed), callable_mp(this, &Node3DEditor::_update_context_toolbar));

	_update_context_toolbar();
}

void Node3DEditor::remove_control_from_menu_panel(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	ERR_FAIL_COND(p_control->get_parent() != context_toolbar_hbox);

	p_control->disconnect(SceneStringName(visibility_changed), callable_mp(this, &Node3DEditor::_update_context_toolbar));

	VSeparator *sep = context_toolbar_separators[p_control];
	context_toolbar_hbox->remove_child(sep);
	context_toolbar_hbox->remove_child(p_control);
	context_toolbar_separators.erase(p_control);
	memdelete(sep);

	_update_context_toolbar();
}

void Node3DEditor::_update_context_toolbar() {
	bool has_visible = false;
	bool first_visible = false;

	for (int i = 0; i < context_toolbar_hbox->get_child_count(); i++) {
		Control *child = Object::cast_to<Control>(context_toolbar_hbox->get_child(i));
		if (!child || !context_toolbar_separators.has(child)) {
			continue;
		}
		if (child->is_visible()) {
			first_visible = !has_visible;
			has_visible = true;
		}

		VSeparator *sep = context_toolbar_separators[child];
		sep->set_visible(!first_visible && child->is_visible());
	}

	context_toolbar_panel->set_visible(has_visible);
}

void Node3DEditor::set_can_preview(Camera3D *p_preview) {
	for (int i = 0; i < 4; i++) {
		viewports[i]->set_can_preview(p_preview);
	}

	viewports[last_used_viewport]->switch_preview_camera(p_preview);
}

VSplitContainer *Node3DEditor::get_shader_split() {
	return shader_split;
}

Node3DEditorViewport *Node3DEditor::get_last_used_viewport() {
	return viewports[last_used_viewport];
}

void Node3DEditor::set_freelook_viewport(Node3DEditorViewport *p_viewport) {
	freelook_viewport = p_viewport;
}

Node3DEditorViewport *Node3DEditor::get_freelook_viewport() const {
	return freelook_viewport;
}

void Node3DEditor::add_control_to_left_panel(Control *p_control) {
	left_panel_split->add_child(p_control);
	left_panel_split->move_child(p_control, 0);
}

void Node3DEditor::add_control_to_right_panel(Control *p_control) {
	right_panel_split->add_child(p_control);
	right_panel_split->move_child(p_control, 1);
}

void Node3DEditor::remove_control_from_left_panel(Control *p_control) {
	left_panel_split->remove_child(p_control);
}

void Node3DEditor::remove_control_from_right_panel(Control *p_control) {
	right_panel_split->remove_child(p_control);
}

void Node3DEditor::move_control_to_left_panel(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	if (p_control->get_parent() == left_panel_split) {
		return;
	}

	ERR_FAIL_COND(p_control->get_parent() != right_panel_split);
	right_panel_split->remove_child(p_control);

	add_control_to_left_panel(p_control);
}

void Node3DEditor::move_control_to_right_panel(Control *p_control) {
	ERR_FAIL_NULL(p_control);
	if (p_control->get_parent() == right_panel_split) {
		return;
	}

	ERR_FAIL_COND(p_control->get_parent() != left_panel_split);
	left_panel_split->remove_child(p_control);

	add_control_to_right_panel(p_control);
}

void Node3DEditor::_request_gizmo(Object *p_obj) {
	Node3D *sp = Object::cast_to<Node3D>(p_obj);
	if (!sp) {
		return;
	}

	bool is_selected = (sp == selected);

	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	if (edited_scene && (sp == edited_scene || (sp->get_owner() && edited_scene->is_ancestor_of(sp)))) {
		for (int i = 0; i < gizmo_plugins_by_priority.size(); ++i) {
			Ref<EditorNode3DGizmo> seg = gizmo_plugins_by_priority.write[i]->get_gizmo(sp);

			if (seg.is_valid()) {
				sp->add_gizmo(seg);

				if (is_selected != seg->is_selected()) {
					seg->set_selected(is_selected);
				}
			}
		}
		if (!sp->get_gizmos().is_empty()) {
			sp->update_gizmos();
		}
	}
}

void Node3DEditor::_request_gizmo_for_id(ObjectID p_id) {
	Node3D *node = ObjectDB::get_instance<Node3D>(p_id);
	if (node) {
		_request_gizmo(node);
	}
}

void Node3DEditor::_set_subgizmo_selection(Object *p_obj, Ref<Node3DGizmo> p_gizmo, int p_id, Transform3D p_transform) {
	if (p_id == -1) {
		_clear_subgizmo_selection(p_obj);
		return;
	}

	Node3D *sp = nullptr;
	if (p_obj) {
		sp = Object::cast_to<Node3D>(p_obj);
	} else {
		sp = selected;
	}

	if (!sp) {
		return;
	}

	Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
	if (se) {
		se->subgizmos.clear();
		se->subgizmos.insert(p_id, p_transform);
		se->gizmo = p_gizmo;
		sp->update_gizmos();
		update_transform_gizmo();
	}
}

void Node3DEditor::_clear_subgizmo_selection(Object *p_obj) {
	Node3D *sp = nullptr;
	if (p_obj) {
		sp = Object::cast_to<Node3D>(p_obj);
	} else {
		sp = selected;
	}

	if (!sp) {
		return;
	}

	Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
	if (se) {
		se->subgizmos.clear();
		se->gizmo.unref();
		sp->update_gizmos();
		update_transform_gizmo();
	}
}

void Node3DEditor::_toggle_maximize_view(Object *p_viewport) {
	if (!p_viewport) {
		return;
	}
	Node3DEditorViewport *current_viewport = Object::cast_to<Node3DEditorViewport>(p_viewport);
	if (!current_viewport) {
		return;
	}

	int index = -1;
	bool maximized = false;
	for (int i = 0; i < 4; i++) {
		if (viewports[i] == current_viewport) {
			index = i;
			if (current_viewport->get_global_rect() == viewport_base->get_global_rect()) {
				maximized = true;
			}
			break;
		}
	}
	if (index == -1) {
		return;
	}

	if (!maximized) {
		for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
			if (i == (uint32_t)index) {
				viewports[i]->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
			} else {
				viewports[i]->hide();
			}
		}
	} else {
		for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
			viewports[i]->show();
		}

		if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT))) {
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);
		} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS))) {
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS);
		} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT))) {
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS_ALT);
		} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS))) {
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS);
		} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT))) {
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS_ALT);
		} else if (view_layout_menu->get_popup()->is_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS))) {
			_menu_item_pressed(MENU_VIEW_USE_4_VIEWPORTS);
		}
	}
}

void Node3DEditor::_viewport_clicked(int p_viewport_idx) {
	last_used_viewport = p_viewport_idx;
}

void Node3DEditor::_node_added(Node *p_node) {
	if (EditorNode::get_singleton()->get_scene_root()->is_ancestor_of(p_node)) {
		if (Object::cast_to<WorldEnvironment>(p_node)) {
			world_env_count++;
			if (world_env_count == 1) {
				_update_preview_environment();
			}
		} else if (Object::cast_to<DirectionalLight3D>(p_node)) {
			directional_light_count++;
			if (directional_light_count == 1) {
				_update_preview_environment();
			}
		}
	}
}

void Node3DEditor::_node_removed(Node *p_node) {
	if (EditorNode::get_singleton()->get_scene_root()->is_ancestor_of(p_node)) {
		if (Object::cast_to<WorldEnvironment>(p_node)) {
			world_env_count--;
			if (world_env_count == 0) {
				_update_preview_environment();
			}
		} else if (Object::cast_to<DirectionalLight3D>(p_node)) {
			directional_light_count--;
			if (directional_light_count == 0) {
				_update_preview_environment();
			}
		}
	}

	if (p_node == selected) {
		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected);
		if (se) {
			se->gizmo.unref();
			se->subgizmos.clear();
		}
		selected = nullptr;
		update_transform_gizmo();
	}
}

void Node3DEditor::_register_all_gizmos() {
	add_gizmo_plugin(Ref<Camera3DGizmoPlugin>(memnew(Camera3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Light3DGizmoPlugin>(memnew(Light3DGizmoPlugin)));
	add_gizmo_plugin(Ref<AudioStreamPlayer3DGizmoPlugin>(memnew(AudioStreamPlayer3DGizmoPlugin)));
	add_gizmo_plugin(Ref<AudioListener3DGizmoPlugin>(memnew(AudioListener3DGizmoPlugin)));
	add_gizmo_plugin(Ref<MeshInstance3DGizmoPlugin>(memnew(MeshInstance3DGizmoPlugin)));
	add_gizmo_plugin(Ref<OccluderInstance3DGizmoPlugin>(memnew(OccluderInstance3DGizmoPlugin)));
	add_gizmo_plugin(Ref<SpriteBase3DGizmoPlugin>(memnew(SpriteBase3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Label3DGizmoPlugin>(memnew(Label3DGizmoPlugin)));
	add_gizmo_plugin(Ref<GeometryInstance3DGizmoPlugin>(memnew(GeometryInstance3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Marker3DGizmoPlugin>(memnew(Marker3DGizmoPlugin)));
	add_gizmo_plugin(Ref<SpringBoneCollision3DGizmoPlugin>(memnew(SpringBoneCollision3DGizmoPlugin)));
	add_gizmo_plugin(Ref<SpringBoneSimulator3DGizmoPlugin>(memnew(SpringBoneSimulator3DGizmoPlugin)));
	add_gizmo_plugin(Ref<VisibleOnScreenNotifier3DGizmoPlugin>(memnew(VisibleOnScreenNotifier3DGizmoPlugin)));
	add_gizmo_plugin(Ref<GPUParticles3DGizmoPlugin>(memnew(GPUParticles3DGizmoPlugin)));
	add_gizmo_plugin(Ref<GPUParticlesCollision3DGizmoPlugin>(memnew(GPUParticlesCollision3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Particles3DEmissionShapeGizmoPlugin>(memnew(Particles3DEmissionShapeGizmoPlugin)));
	add_gizmo_plugin(Ref<CPUParticles3DGizmoPlugin>(memnew(CPUParticles3DGizmoPlugin)));
	add_gizmo_plugin(Ref<ReflectionProbeGizmoPlugin>(memnew(ReflectionProbeGizmoPlugin)));
	add_gizmo_plugin(Ref<DecalGizmoPlugin>(memnew(DecalGizmoPlugin)));
	add_gizmo_plugin(Ref<VoxelGIGizmoPlugin>(memnew(VoxelGIGizmoPlugin)));
	add_gizmo_plugin(Ref<LightmapGIGizmoPlugin>(memnew(LightmapGIGizmoPlugin)));
	add_gizmo_plugin(Ref<LightmapProbeGizmoPlugin>(memnew(LightmapProbeGizmoPlugin)));
	add_gizmo_plugin(Ref<FogVolumeGizmoPlugin>(memnew(FogVolumeGizmoPlugin)));
	add_gizmo_plugin(Ref<TwoBoneIK3DGizmoPlugin>(memnew(TwoBoneIK3DGizmoPlugin)));
	add_gizmo_plugin(Ref<ChainIK3DGizmoPlugin>(memnew(ChainIK3DGizmoPlugin)));
	// Physics gizmo plugins.
	add_gizmo_plugin(Ref<CollisionObject3DGizmoPlugin>(memnew(CollisionObject3DGizmoPlugin)));
	add_gizmo_plugin(Ref<CollisionShape3DGizmoPlugin>(memnew(CollisionShape3DGizmoPlugin)));
	add_gizmo_plugin(Ref<CollisionPolygon3DGizmoPlugin>(memnew(CollisionPolygon3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Joint3DGizmoPlugin>(memnew(Joint3DGizmoPlugin)));
	add_gizmo_plugin(Ref<SoftBody3DGizmoPlugin>(memnew(SoftBody3DGizmoPlugin)));
	add_gizmo_plugin(Ref<ShapeCast3DGizmoPlugin>(memnew(ShapeCast3DGizmoPlugin)));
	add_gizmo_plugin(Ref<SpringArm3DGizmoPlugin>(memnew(SpringArm3DGizmoPlugin)));
	add_gizmo_plugin(Ref<PhysicalBone3DGizmoPlugin>(memnew(PhysicalBone3DGizmoPlugin)));
	add_gizmo_plugin(Ref<VehicleWheel3DGizmoPlugin>(memnew(VehicleWheel3DGizmoPlugin)));
	add_gizmo_plugin(Ref<RayCast3DGizmoPlugin>(memnew(RayCast3DGizmoPlugin)));
}

void Node3DEditor::_bind_methods() {
	ClassDB::bind_method("_get_editor_data", &Node3DEditor::_get_editor_data);
	ClassDB::bind_method("_request_gizmo", &Node3DEditor::_request_gizmo);
	ClassDB::bind_method("_request_gizmo_for_id", &Node3DEditor::_request_gizmo_for_id);
	ClassDB::bind_method("_set_subgizmo_selection", &Node3DEditor::_set_subgizmo_selection);
	ClassDB::bind_method("_clear_subgizmo_selection", &Node3DEditor::_clear_subgizmo_selection);
	ClassDB::bind_method("_refresh_menu_icons", &Node3DEditor::_refresh_menu_icons);
	ClassDB::bind_method("_preview_settings_changed", &Node3DEditor::_preview_settings_changed);

	ClassDB::bind_method("update_all_gizmos", &Node3DEditor::update_all_gizmos);
	ClassDB::bind_method("update_transform_gizmo", &Node3DEditor::update_transform_gizmo);

	ADD_SIGNAL(MethodInfo("transform_3d_key_request"));
	ADD_SIGNAL(MethodInfo("item_lock_status_changed"));
	ADD_SIGNAL(MethodInfo("item_group_status_changed"));
}

void Node3DEditor::clear() {
	settings_fov->set_value(EDITOR_GET("editors/3d/default_fov"));
	settings_znear->set_value(EDITOR_GET("editors/3d/default_z_near"));
	settings_zfar->set_value(EDITOR_GET("editors/3d/default_z_far"));

	snap_translate_value = EditorSettings::get_singleton()->get_project_metadata("3d_editor", "snap_translate_value", 1);
	snap_rotate_value = EditorSettings::get_singleton()->get_project_metadata("3d_editor", "snap_rotate_value", 15);
	snap_scale_value = EditorSettings::get_singleton()->get_project_metadata("3d_editor", "snap_scale_value", 10);
	_snap_update();

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->reset();
	}

	if (origin_instance.is_valid()) {
		RenderingServer::get_singleton()->instance_set_visible(origin_instance, true);
	}

	view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN), true);
	for (int i = 0; i < 3; ++i) {
		if (grid_enable[i]) {
			grid_visible[i] = true;
		}
	}

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->view_display_menu->get_popup()->set_item_checked(viewports[i]->view_display_menu->get_popup()->get_item_index(Node3DEditorViewport::VIEW_AUDIO_LISTENER), i == 0);
		viewports[i]->viewport->set_as_audio_listener_3d(i == 0);
	}

	view_layout_menu->get_popup()->set_item_checked(view_layout_menu->get_popup()->get_item_index(MENU_VIEW_GRID), true);
	grid_enabled = true;
	grid_init_draw = false;
}

void Node3DEditor::_sun_direction_draw() {
	sun_direction->draw_rect(Rect2(Vector2(), sun_direction->get_size()), Color(1, 1, 1, 1));
	Vector3 z_axis = preview_sun->get_transform().basis.get_column(Vector3::AXIS_Z);
	z_axis = get_editor_viewport(0)->camera->get_camera_transform().basis.xform_inv(z_axis);
	sun_direction_material->set_shader_parameter("sun_direction", Vector3(z_axis.x, -z_axis.y, z_axis.z));
	Color color = sun_color->get_pick_color() * sun_energy->get_value();
	sun_direction_material->set_shader_parameter("sun_color", Vector3(color.r, color.g, color.b));
}

void Node3DEditor::_preview_settings_changed() {
	if (sun_environ_updating) {
		return;
	}

	{ // preview sun
		sun_rotation.x = Math::deg_to_rad(-sun_angle_altitude->get_value());
		sun_rotation.y = Math::deg_to_rad(180.0 - sun_angle_azimuth->get_value());
		Transform3D t;
		t.basis = Basis::from_euler(Vector3(sun_rotation.x, sun_rotation.y, 0));
		preview_sun->set_transform(t);
		sun_direction->queue_redraw();
		preview_sun->set_param(Light3D::PARAM_ENERGY, sun_energy->get_value());
		preview_sun->set_param(Light3D::PARAM_SHADOW_MAX_DISTANCE, sun_shadow_max_distance->get_value());
		preview_sun->set_color(sun_color->get_pick_color());
	}

	{ //preview env
		sky_material->set_energy_multiplier(environ_energy->get_value());
		Color hz_color = environ_sky_color->get_pick_color().lerp(environ_ground_color->get_pick_color(), 0.5);
		float hz_lum = hz_color.get_luminance() * 3.333;
		hz_color = hz_color.lerp(Color(hz_lum, hz_lum, hz_lum), 0.5);
		sky_material->set_sky_top_color(environ_sky_color->get_pick_color());
		sky_material->set_sky_horizon_color(hz_color);
		sky_material->set_ground_bottom_color(environ_ground_color->get_pick_color());
		sky_material->set_ground_horizon_color(hz_color);

		environment->set_ssao_enabled(environ_ao_button->is_pressed());
		environment->set_glow_enabled(environ_glow_button->is_pressed());
		environment->set_sdfgi_enabled(environ_gi_button->is_pressed());
		environment->set_tonemapper(environ_tonemap_button->is_pressed() ? Environment::TONE_MAPPER_FILMIC : Environment::TONE_MAPPER_LINEAR);
	}
}

void Node3DEditor::_load_default_preview_settings() {
	sun_environ_updating = true;

	// These default rotations place the preview sun at an angular altitude
	// of 60 degrees (must be negative) and an azimuth of 30 degrees clockwise
	// from north (or 150 CCW from south), from north east, facing south west.
	// On any not-tidally-locked planet, a sun would have an angular altitude
	// of 60 degrees as the average of all points on the sphere at noon.
	// The azimuth choice is arbitrary, but ideally shouldn't be on an axis.
	sun_rotation = Vector2(-Math::deg_to_rad(60.0), Math::deg_to_rad(150.0));

	sun_angle_altitude->set_value_no_signal(-Math::rad_to_deg(sun_rotation.x));
	sun_angle_azimuth->set_value_no_signal(180.0 - Math::rad_to_deg(sun_rotation.y));
	sun_direction->queue_redraw();
	environ_sky_color->set_pick_color(Color(0.385, 0.454, 0.55));
	environ_ground_color->set_pick_color(Color(0.2, 0.169, 0.133));
	environ_energy->set_value_no_signal(1.0);
	if (OS::get_singleton()->get_current_rendering_method() != "gl_compatibility" && OS::get_singleton()->get_current_rendering_method() != "dummy") {
		environ_glow_button->set_pressed_no_signal(true);
	}
	environ_tonemap_button->set_pressed_no_signal(false);
	environ_ao_button->set_pressed_no_signal(false);
	environ_gi_button->set_pressed_no_signal(false);
	sun_shadow_max_distance->set_value_no_signal(100);

	sun_color->set_pick_color(Color(1, 1, 1));
	sun_energy->set_value_no_signal(1.0);

	sun_environ_updating = false;
}

void Node3DEditor::_update_preview_environment() {
	bool disable_light = directional_light_count > 0 || !sun_button->is_pressed();

	sun_button->set_disabled(directional_light_count > 0);

	if (disable_light) {
		if (preview_sun->get_parent()) {
			preview_sun->get_parent()->remove_child(preview_sun);
			sun_state->show();
			sun_vb->hide();
			preview_sun_dangling = true;
		}

		if (directional_light_count > 0) {
			sun_state->set_text(TTRC("Scene contains\nDirectionalLight3D.\nPreview disabled."));
		} else {
			sun_state->set_text(TTRC("Preview disabled."));
		}

	} else {
		if (!preview_sun->get_parent()) {
			add_child(preview_sun, true);
			sun_state->hide();
			sun_vb->show();
			preview_sun_dangling = false;
		}
	}

	sun_angle_altitude->set_value_no_signal(-Math::rad_to_deg(sun_rotation.x));
	sun_angle_azimuth->set_value_no_signal(180.0 - Math::rad_to_deg(sun_rotation.y));

	bool disable_env = world_env_count > 0 || !environ_button->is_pressed();

	environ_button->set_disabled(world_env_count > 0);

	if (disable_env) {
		if (preview_environment->get_parent()) {
			preview_environment->get_parent()->remove_child(preview_environment);
			environ_state->show();
			environ_vb->hide();
			preview_env_dangling = true;
		}
		if (world_env_count > 0) {
			environ_state->set_text(TTRC("Scene contains\nWorldEnvironment.\nPreview disabled."));
		} else {
			environ_state->set_text(TTRC("Preview disabled."));
		}

	} else {
		if (!preview_environment->get_parent()) {
			add_child(preview_environment);
			environ_state->hide();
			environ_vb->show();
			preview_env_dangling = false;
		}
	}
}

void Node3DEditor::_sun_direction_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
		sun_rotation.x += mm->get_relative().y * (0.02 * EDSCALE);
		sun_rotation.y -= mm->get_relative().x * (0.02 * EDSCALE);
		sun_rotation.x = CLAMP(sun_rotation.x, -Math::TAU / 4, Math::TAU / 4);

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Set Preview Sun Direction"), UndoRedo::MergeMode::MERGE_ENDS);
		undo_redo->add_do_method(sun_angle_altitude, "set_value_no_signal", -Math::rad_to_deg(sun_rotation.x));
		undo_redo->add_undo_method(sun_angle_altitude, "set_value_no_signal", sun_angle_altitude->get_value());
		undo_redo->add_do_method(sun_angle_azimuth, "set_value_no_signal", 180.0 - Math::rad_to_deg(sun_rotation.y));
		undo_redo->add_undo_method(sun_angle_azimuth, "set_value_no_signal", sun_angle_azimuth->get_value());
		undo_redo->add_do_method(this, "_preview_settings_changed");
		undo_redo->add_undo_method(this, "_preview_settings_changed");
		undo_redo->commit_action();
	}
}

void Node3DEditor::_sun_direction_set_altitude(float p_altitude) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Sun Altitude"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(sun_angle_altitude, "set_value_no_signal", p_altitude);
	undo_redo->add_undo_method(sun_angle_altitude, "set_value_no_signal", -Math::rad_to_deg(sun_rotation.x));
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_sun_direction_set_azimuth(float p_azimuth) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Sun Azimuth"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(sun_angle_azimuth, "set_value_no_signal", p_azimuth);
	undo_redo->add_undo_method(sun_angle_azimuth, "set_value_no_signal", 180.0 - Math::rad_to_deg(sun_rotation.y));
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_sun_set_color(const Color &p_color) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Sun Color"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(sun_color, "set_pick_color", p_color);
	undo_redo->add_undo_method(sun_color, "set_pick_color", preview_sun->get_color());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_sun_set_energy(float p_energy) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Sun Energy"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(sun_energy, "set_value_no_signal", p_energy);
	undo_redo->add_undo_method(sun_energy, "set_value_no_signal", preview_sun->get_param(Light3D::PARAM_ENERGY));
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_sun_set_shadow_max_distance(float p_shadow_max_distance) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Sun Max Shadow Distance"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(sun_shadow_max_distance, "set_value_no_signal", p_shadow_max_distance);
	undo_redo->add_undo_method(sun_shadow_max_distance, "set_value_no_signal", preview_sun->get_param(Light3D::PARAM_SHADOW_MAX_DISTANCE));
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_sky_color(const Color &p_color) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Sky Color"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(environ_sky_color, "set_pick_color", p_color);
	undo_redo->add_undo_method(environ_sky_color, "set_pick_color", sky_material->get_sky_top_color());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_ground_color(const Color &p_color) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Ground Color"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(environ_ground_color, "set_pick_color", p_color);
	undo_redo->add_undo_method(environ_ground_color, "set_pick_color", sky_material->get_ground_bottom_color());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_sky_energy(float p_energy) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Energy"), UndoRedo::MergeMode::MERGE_ENDS);
	undo_redo->add_do_method(environ_energy, "set_value_no_signal", p_energy);
	undo_redo->add_undo_method(environ_energy, "set_value_no_signal", sky_material->get_energy_multiplier());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_ao() {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Ambient Occlusion"));
	undo_redo->add_do_method(environ_ao_button, "set_pressed", environ_ao_button->is_pressed());
	undo_redo->add_undo_method(environ_ao_button, "set_pressed", !environ_ao_button->is_pressed());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_glow() {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Glow"));
	undo_redo->add_do_method(environ_glow_button, "set_pressed", environ_glow_button->is_pressed());
	undo_redo->add_undo_method(environ_glow_button, "set_pressed", !environ_glow_button->is_pressed());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_tonemap() {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Tonemap"));
	undo_redo->add_do_method(environ_tonemap_button, "set_pressed", environ_tonemap_button->is_pressed());
	undo_redo->add_undo_method(environ_tonemap_button, "set_pressed", !environ_tonemap_button->is_pressed());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::_environ_set_gi() {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set Preview Environment Global Illumination"));
	undo_redo->add_do_method(environ_gi_button, "set_pressed", environ_gi_button->is_pressed());
	undo_redo->add_undo_method(environ_gi_button, "set_pressed", !environ_gi_button->is_pressed());
	undo_redo->add_do_method(this, "_preview_settings_changed");
	undo_redo->add_undo_method(this, "_preview_settings_changed");
	undo_redo->commit_action();
}

void Node3DEditor::PreviewSunEnvPopup::shortcut_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed()) {
		bool handled = false;

		if (ED_IS_SHORTCUT("ui_undo", p_event)) {
			EditorNode::get_singleton()->undo();
			handled = true;
		}

		if (ED_IS_SHORTCUT("ui_redo", p_event)) {
			EditorNode::get_singleton()->redo();
			handled = true;
		}

		if (handled) {
			set_input_as_handled();
		}
	}
}

Node3DEditor::Node3DEditor() {
	gizmo.visible = true;
	gizmo.scale = 1.0;
	gizmo_view_rotation_scale = GIZMO_CIRCLE_SIZE * (float)EDITOR_GET("editors/3d/view_plane_rotation_gizmo_scale");

	viewport_environment.instantiate();
	VBoxContainer *vbc = this;

	ERR_FAIL_COND_MSG(singleton != nullptr, "A Node3DEditor singleton already exists.");
	singleton = this;
	editor_selection = EditorNode::get_singleton()->get_editor_selection();
	editor_selection->add_editor_plugin(this);

	MarginContainer *toolbar_margin = memnew(MarginContainer);
	toolbar_margin->set_theme_type_variation("MainToolBarMargin");
	vbc->add_child(toolbar_margin);

	// A fluid container for all toolbars.
	HFlowContainer *main_flow = memnew(HFlowContainer);
	toolbar_margin->add_child(main_flow);

	// Main toolbars.
	HBoxContainer *main_menu_hbox = memnew(HBoxContainer);
	main_flow->add_child(main_menu_hbox);

	tool_button[TOOL_MODE_TRANSFORM] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_TRANSFORM]);
	tool_button[TOOL_MODE_TRANSFORM]->set_toggle_mode(true);
	tool_button[TOOL_MODE_TRANSFORM]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_MODE_TRANSFORM]->set_pressed(true);
	tool_button[TOOL_MODE_TRANSFORM]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_TRANSFORM));
	tool_button[TOOL_MODE_TRANSFORM]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_transform", TTRC("Transform Mode"), Key::Q, true));
	tool_button[TOOL_MODE_TRANSFORM]->set_shortcut_context(this);
	tool_button[TOOL_MODE_TRANSFORM]->set_accessibility_name(TTRC("Transform Mode"));

	tool_button[TOOL_MODE_MOVE] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_MOVE]);
	tool_button[TOOL_MODE_MOVE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_MOVE]->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tool_button[TOOL_MODE_MOVE]->set_theme_type_variation(SceneStringName(FlatButton));

	tool_button[TOOL_MODE_MOVE]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_MOVE));
	tool_button[TOOL_MODE_MOVE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_move", TTRC("Move Mode"), Key::W, true));
	tool_button[TOOL_MODE_MOVE]->set_shortcut_context(this);
	tool_button[TOOL_MODE_MOVE]->set_accessibility_name(TTRC("Move Mode"));

	tool_button[TOOL_MODE_ROTATE] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_ROTATE]);
	tool_button[TOOL_MODE_ROTATE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_ROTATE]->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tool_button[TOOL_MODE_ROTATE]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_MODE_ROTATE]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_ROTATE));
	tool_button[TOOL_MODE_ROTATE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_rotate", TTRC("Rotate Mode"), Key::E, true));
	tool_button[TOOL_MODE_ROTATE]->set_shortcut_context(this);
	tool_button[TOOL_MODE_ROTATE]->set_accessibility_name(TTRC("Rotate Mode"));

	tool_button[TOOL_MODE_SCALE] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_SCALE]);
	tool_button[TOOL_MODE_SCALE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_SCALE]->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tool_button[TOOL_MODE_SCALE]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_MODE_SCALE]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_SCALE));
	tool_button[TOOL_MODE_SCALE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_scale", TTRC("Scale Mode"), Key::R, true));
	tool_button[TOOL_MODE_SCALE]->set_shortcut_context(this);
	tool_button[TOOL_MODE_SCALE]->set_accessibility_name(TTRC("Scale Mode"));

	tool_button[TOOL_MODE_SELECT] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_SELECT]);
	tool_button[TOOL_MODE_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_SELECT]->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tool_button[TOOL_MODE_SELECT]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_MODE_SELECT]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_SELECT));
	tool_button[TOOL_MODE_SELECT]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_select", TTRC("Select Mode"), Key::V, true));
	tool_button[TOOL_MODE_SELECT]->set_shortcut_context(this);
	tool_button[TOOL_MODE_SELECT]->set_accessibility_name(TTRC("Select Mode"));

	main_menu_hbox->add_child(memnew(VSeparator));

	tool_button[TOOL_MODE_LIST_SELECT] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_LIST_SELECT]);
	tool_button[TOOL_MODE_LIST_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_LIST_SELECT]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_MODE_LIST_SELECT]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_LIST_SELECT));
	tool_button[TOOL_MODE_LIST_SELECT]->set_tooltip_text(TTR("Show list of selectable nodes at position clicked.") + "\n" + TTR("Alt+RMB: Show list of all nodes at position clicked, including locked."));
	tool_button[TOOL_MODE_LIST_SELECT]->set_accessibility_name(TTRC("Show List of Selectable Nodes"));

	tool_button[TOOL_LOCK_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_LOCK_SELECTED]);
	tool_button[TOOL_LOCK_SELECTED]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_LOCK_SELECTED]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_LOCK_SELECTED));
	tool_button[TOOL_LOCK_SELECTED]->set_tooltip_text(TTRC("Lock selected node, preventing selection and movement."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_LOCK_SELECTED]->set_shortcut(ED_GET_SHORTCUT("editor/lock_selected_nodes"));
	tool_button[TOOL_LOCK_SELECTED]->set_accessibility_name(TTRC("Lock"));

	tool_button[TOOL_UNLOCK_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_UNLOCK_SELECTED]);
	tool_button[TOOL_UNLOCK_SELECTED]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_UNLOCK_SELECTED]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_UNLOCK_SELECTED));
	tool_button[TOOL_UNLOCK_SELECTED]->set_tooltip_text(TTRC("Unlock selected node, allowing selection and movement."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_UNLOCK_SELECTED]->set_shortcut(ED_GET_SHORTCUT("editor/unlock_selected_nodes"));
	tool_button[TOOL_UNLOCK_SELECTED]->set_accessibility_name(TTRC("Unlock"));

	tool_button[TOOL_GROUP_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_GROUP_SELECTED]);
	tool_button[TOOL_GROUP_SELECTED]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_GROUP_SELECTED]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_GROUP_SELECTED));
	tool_button[TOOL_GROUP_SELECTED]->set_tooltip_text(TTRC("Groups the selected node with its children. This selects the parent when any child node is clicked in 2D and 3D view."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_GROUP_SELECTED]->set_shortcut(ED_GET_SHORTCUT("editor/group_selected_nodes"));
	tool_button[TOOL_GROUP_SELECTED]->set_accessibility_name(TTRC("Group"));

	tool_button[TOOL_UNGROUP_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_UNGROUP_SELECTED]);
	tool_button[TOOL_UNGROUP_SELECTED]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_button[TOOL_UNGROUP_SELECTED]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_UNGROUP_SELECTED));
	tool_button[TOOL_UNGROUP_SELECTED]->set_tooltip_text(TTRC("Ungroups the selected node from its children. Child nodes will be individual items in 2D and 3D view."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_UNGROUP_SELECTED]->set_shortcut(ED_GET_SHORTCUT("editor/ungroup_selected_nodes"));
	tool_button[TOOL_UNGROUP_SELECTED]->set_accessibility_name(TTRC("Ungroup"));

	tool_button[TOOL_RULER] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_RULER]);
	tool_button[TOOL_RULER]->set_toggle_mode(true);
	tool_button[TOOL_RULER]->set_theme_type_variation("FlatButton");
	tool_button[TOOL_RULER]->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_RULER));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_RULER]->set_shortcut(ED_SHORTCUT("spatial_editor/measure", TTRC("Ruler Mode"), Key::M));
	tool_button[TOOL_RULER]->set_accessibility_name(TTRC("Ruler Mode"));

	main_menu_hbox->add_child(memnew(VSeparator));

	tool_option_button[TOOL_OPT_LOCAL_COORDS] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_LOCAL_COORDS]);
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->connect(SceneStringName(toggled), callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_LOCAL_COORDS));
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_shortcut(ED_SHORTCUT("spatial_editor/local_coords", TTRC("Use Local Space"), Key::T));
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_shortcut_context(this);
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_accessibility_name(TTRC("Use Local Space"));

	tool_option_button[TOOL_OPT_USE_SNAP] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_USE_SNAP]);
	tool_option_button[TOOL_OPT_USE_SNAP]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_USE_SNAP]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_option_button[TOOL_OPT_USE_SNAP]->connect(SceneStringName(toggled), callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_USE_SNAP));
	tool_option_button[TOOL_OPT_USE_SNAP]->set_shortcut(ED_SHORTCUT("spatial_editor/snap", TTRC("Use Snap"), Key::Y));
	tool_option_button[TOOL_OPT_USE_SNAP]->set_shortcut_context(this);
	tool_option_button[TOOL_OPT_USE_SNAP]->set_accessibility_name(TTRC("Use Snap"));

	tool_option_button[TOOL_OPT_USE_TRACKBALL] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_USE_TRACKBALL]);
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->connect(SceneStringName(toggled), callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_USE_TRACKBALL));
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_shortcut(ED_SHORTCUT("spatial_editor/trackball", TTRC("Use Trackball"), Key::U));
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_shortcut_context(this);
	tool_option_button[TOOL_OPT_USE_TRACKBALL]->set_accessibility_name(TTRC("Use Trackball"));

	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]);
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_theme_type_variation(SceneStringName(FlatButton));
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->connect(SceneStringName(toggled), callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_PRESERVE_CHILDREN_TRANSFORM));
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_shortcut(ED_SHORTCUT("spatial_editor/preserve_children_transform", TTRC("Preserve Children Transform"), Key::P));
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_shortcut_context(this);
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_accessibility_name(TTRC("Preserve Children Transform"));
	tool_option_button[TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->set_tooltip_text(TTRC("When enabled, transforming a node will preserve the global transform of its children.\nThis also applies when editing transform properties in the Inspector."));

	main_menu_hbox->add_child(memnew(VSeparator));
	sun_button = memnew(Button);
	sun_button->set_tooltip_text(TTRC("Toggle preview sunlight.\nIf a DirectionalLight3D node is added to the scene, preview sunlight is disabled."));
	sun_button->set_toggle_mode(true);
	sun_button->set_accessibility_name(TTRC("Toggle preview sunlight."));
	sun_button->set_theme_type_variation(SceneStringName(FlatButton));
	sun_button->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_update_preview_environment), CONNECT_DEFERRED);
	// Preview is enabled by default - ensure this applies on editor startup when there is no state yet.
	sun_button->set_pressed(true);

	main_menu_hbox->add_child(sun_button);

	environ_button = memnew(Button);
	environ_button->set_tooltip_text(TTRC("Toggle preview environment.\nIf a WorldEnvironment node is added to the scene, preview environment is disabled."));
	environ_button->set_toggle_mode(true);
	environ_button->set_accessibility_name(TTRC("Toggle preview environment."));
	environ_button->set_theme_type_variation(SceneStringName(FlatButton));
	environ_button->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_update_preview_environment), CONNECT_DEFERRED);
	// Preview is enabled by default - ensure this applies on editor startup when there is no state yet.
	environ_button->set_pressed(true);

	main_menu_hbox->add_child(environ_button);

	sun_environ_settings = memnew(Button);
	sun_environ_settings->set_tooltip_text(TTRC("Edit Sun and Environment settings."));
	sun_environ_settings->set_theme_type_variation(SceneStringName(FlatButton));
	sun_environ_settings->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_sun_environ_settings_pressed));

	main_menu_hbox->add_child(sun_environ_settings);

	main_menu_hbox->add_child(memnew(VSeparator));

	// Drag and drop support;
	preview_node = memnew(Node3D);
	preview_bounds = AABB();

	ED_SHORTCUT("spatial_editor/bottom_view", TTRC("Bottom View"), KeyModifierMask::ALT + Key::KP_7);
	ED_SHORTCUT("spatial_editor/top_view", TTRC("Top View"), Key::KP_7);
	ED_SHORTCUT("spatial_editor/rear_view", TTRC("Rear View"), KeyModifierMask::ALT + Key::KP_1);
	ED_SHORTCUT("spatial_editor/front_view", TTRC("Front View"), Key::KP_1);
	ED_SHORTCUT("spatial_editor/left_view", TTRC("Left View"), KeyModifierMask::ALT + Key::KP_3);
	ED_SHORTCUT("spatial_editor/right_view", TTRC("Right View"), Key::KP_3);
	ED_SHORTCUT("spatial_editor/orbit_view_down", TTRC("Orbit View Down"), Key::KP_2);
	ED_SHORTCUT("spatial_editor/orbit_view_left", TTRC("Orbit View Left"), Key::KP_4);
	ED_SHORTCUT("spatial_editor/orbit_view_right", TTRC("Orbit View Right"), Key::KP_6);
	ED_SHORTCUT("spatial_editor/orbit_view_up", TTRC("Orbit View Up"), Key::KP_8);
	ED_SHORTCUT("spatial_editor/orbit_view_180", TTRC("Orbit View 180"), Key::KP_9);
	ED_SHORTCUT("spatial_editor/switch_perspective_orthogonal", TTRC("Switch Perspective/Orthogonal View"), Key::KP_5);
	ED_SHORTCUT("spatial_editor/insert_anim_key", TTRC("Insert Animation Key"), Key::K);
	ED_SHORTCUT("spatial_editor/focus_origin", TTRC("Focus Origin"), Key::O);
	ED_SHORTCUT("spatial_editor/focus_selection", TTRC("Focus Selection"), Key::F);
	ED_SHORTCUT_ARRAY("spatial_editor/align_transform_with_view", TTRC("Align Transform with View"),
			{ int32_t(KeyModifierMask::ALT | KeyModifierMask::CTRL | Key::KP_0),
					int32_t(KeyModifierMask::ALT | KeyModifierMask::CTRL | Key::M),
					int32_t(KeyModifierMask::ALT | KeyModifierMask::CTRL | Key::G) });
	ED_SHORTCUT_OVERRIDE_ARRAY("spatial_editor/align_transform_with_view", "macos",
			{ int32_t(KeyModifierMask::ALT | KeyModifierMask::META | Key::KP_0),
					int32_t(KeyModifierMask::ALT | KeyModifierMask::META | Key::G) });
	ED_SHORTCUT("spatial_editor/align_rotation_with_view", TTRC("Align Rotation with View"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::F);
	ED_SHORTCUT("spatial_editor/freelook_toggle", TTRC("Toggle Freelook"), KeyModifierMask::SHIFT + Key::F);
	ED_SHORTCUT("spatial_editor/decrease_fov", TTRC("Decrease Field of View"), KeyModifierMask::CMD_OR_CTRL + Key::EQUAL); // Usually direct access key for `KEY_PLUS`.
	ED_SHORTCUT("spatial_editor/increase_fov", TTRC("Increase Field of View"), KeyModifierMask::CMD_OR_CTRL + Key::MINUS);
	ED_SHORTCUT("spatial_editor/reset_fov", TTRC("Reset Field of View to Default"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_0);

	PopupMenu *p;

	transform_menu = memnew(MenuButton);
	transform_menu->set_flat(false);
	transform_menu->set_theme_type_variation("FlatMenuButton");
	transform_menu->set_text(TTRC("Transform"));
	transform_menu->set_switch_on_hover(true);
	transform_menu->set_shortcut_context(this);
	main_menu_hbox->add_child(transform_menu);

	p = transform_menu->get_popup();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/snap_to_floor", TTRC("Snap Object to Floor"), Key::PAGEDOWN), MENU_SNAP_TO_FLOOR);
	p->add_shortcut(ED_SHORTCUT("spatial_editor/transform_dialog", TTRC("Transform Dialog...")), MENU_TRANSFORM_DIALOG);

	p->add_separator();
	ED_SHORTCUT("spatial_editor/vertex_snap", TTRC("Vertex Snap"), Key::B);
	p->add_radio_check_item(TTRC("Snap Vertex to Vertex"), MENU_VERTEX_SNAP_BASE_VERTEX);
	p->set_item_checked(p->get_item_index(MENU_VERTEX_SNAP_BASE_VERTEX), true);
	p->add_radio_check_item(TTRC("Snap Origin to Vertex"), MENU_VERTEX_SNAP_BASE_ORIGIN);

	p->add_separator();
	p->add_radio_check_item(TTRC("Snap to Mesh Vertices"), MENU_VERTEX_SNAP_SOURCE_MESH);
	p->set_item_checked(p->get_item_index(MENU_VERTEX_SNAP_SOURCE_MESH), true);
	p->add_radio_check_item(TTRC("Snap to Collision Vertices"), MENU_VERTEX_SNAP_SOURCE_COLLISION);
	_update_vertex_snap_tooltips();

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/configure_snap", TTRC("Configure Snap...")), MENU_TRANSFORM_CONFIGURE_SNAP);

	p->connect(SceneStringName(id_pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed));

	view_layout_menu = memnew(MenuButton);
	view_layout_menu->set_flat(false);
	view_layout_menu->set_theme_type_variation("FlatMenuButton");
	// TRANSLATORS: Noun, name of the 2D/3D View menus.
	view_layout_menu->set_text(TTRC("View"));
	view_layout_menu->set_switch_on_hover(true);
	view_layout_menu->set_shortcut_context(this);
	main_menu_hbox->add_child(view_layout_menu);

	main_menu_hbox->add_child(memnew(VSeparator));

	context_toolbar_panel = memnew(PanelContainer);
	context_toolbar_hbox = memnew(HBoxContainer);
	context_toolbar_panel->add_child(context_toolbar_hbox);
	main_flow->add_child(context_toolbar_panel);

	// Get the view menu popup and have it stay open when a checkable item is selected
	p = view_layout_menu->get_popup();
	p->set_hide_on_checkable_item_selection(false);

	accept = memnew(AcceptDialog);
	EditorNode::get_singleton()->get_gui_base()->add_child(accept);

	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/1_viewport", TTRC("1 Viewport"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_1, true), MENU_VIEW_USE_1_VIEWPORT);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/2_viewports", TTRC("2 Viewports"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_2, true), MENU_VIEW_USE_2_VIEWPORTS);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/2_viewports_alt", TTRC("2 Viewports (Alt)"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::KEY_2, true), MENU_VIEW_USE_2_VIEWPORTS_ALT);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/3_viewports", TTRC("3 Viewports"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_3, true), MENU_VIEW_USE_3_VIEWPORTS);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/3_viewports_alt", TTRC("3 Viewports (Alt)"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::KEY_3, true), MENU_VIEW_USE_3_VIEWPORTS_ALT);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/4_viewports", TTRC("4 Viewports"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_4, true), MENU_VIEW_USE_4_VIEWPORTS);
	p->add_separator();

	gizmos_menu = memnew(PopupMenu);
	gizmos_menu->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	gizmos_menu->set_hide_on_checkable_item_selection(false);
	p->add_submenu_node_item(TTRC("Gizmos"), gizmos_menu);
	gizmos_menu->connect(SceneStringName(id_pressed), callable_mp(this, &Node3DEditor::_menu_gizmo_toggled));

	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_origin", TTRC("View Origin")), MENU_VIEW_ORIGIN);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_grid", TTRC("View Grid"), Key::NUMBERSIGN), MENU_VIEW_GRID);

	p->add_separator();
	p->add_submenu_node_item(TTRC("Preview Translation"), memnew(EditorTranslationPreviewMenu));

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/settings", TTRC("Settings...")), MENU_VIEW_CAMERA_SETTINGS);

	p->set_item_checked(p->get_item_index(MENU_VIEW_ORIGIN), true);
	p->set_item_checked(p->get_item_index(MENU_VIEW_GRID), true);

	p->connect(SceneStringName(id_pressed), callable_mp(this, &Node3DEditor::_menu_item_pressed));

	/* REST OF MENU */

	left_panel_split = memnew(HSplitContainer);
	left_panel_split->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(left_panel_split);

	right_panel_split = memnew(HSplitContainer);
	right_panel_split->set_v_size_flags(SIZE_EXPAND_FILL);
	left_panel_split->add_child(right_panel_split);

	shader_split = memnew(VSplitContainer);
	shader_split->set_h_size_flags(SIZE_EXPAND_FILL);
	right_panel_split->add_child(shader_split);
	viewport_base = memnew(Node3DEditorViewportContainer);
	shader_split->add_child(viewport_base);
	viewport_base->set_v_size_flags(SIZE_EXPAND_FILL);
	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i] = memnew(Node3DEditorViewport(this, i));
		viewports[i]->connect("toggle_maximize_view", callable_mp(this, &Node3DEditor::_toggle_maximize_view));
		viewports[i]->connect("clicked", callable_mp(this, &Node3DEditor::_viewport_clicked).bind(i));
		viewports[i]->assign_pending_data_pointers(preview_node, &preview_bounds, accept);
		viewports[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		viewports[i]->set_v_size_flags(SIZE_EXPAND_FILL);
		viewports[i]->set_custom_minimum_size(Size2(39, 39));
		viewport_base->add_viewport(viewports[i], i);
	}

	/* SNAP DIALOG */

	snap_dialog = memnew(ConfirmationDialog);
	snap_dialog->set_title(TTRC("Snap Settings"));
	add_child(snap_dialog);
	snap_dialog->connect(SceneStringName(confirmed), callable_mp(this, &Node3DEditor::_snap_changed));
	snap_dialog->get_cancel_button()->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_snap_update));

	VBoxContainer *snap_dialog_vbc = memnew(VBoxContainer);
	snap_dialog->add_child(snap_dialog_vbc);

	snap_translate = memnew(EditorSpinSlider);
	snap_translate->set_min(0.0);
	snap_translate->set_step(0.001);
	snap_translate->set_max(10.0);
	snap_translate->set_suffix("m");
	snap_translate->set_allow_greater(true);
	snap_translate->set_accessibility_name(TTRC("Translate Snap"));
	snap_dialog_vbc->add_margin_child(TTR("Translate Snap:"), snap_translate);

	snap_rotate = memnew(EditorSpinSlider);
	snap_rotate->set_min(0.0);
	snap_rotate->set_step(0.1);
	snap_rotate->set_max(360);
	snap_rotate->set_suffix(U"°");
	snap_rotate->set_accessibility_name(TTRC("Rotate Snap"));
	snap_dialog_vbc->add_margin_child(TTR("Rotate Snap:"), snap_rotate);

	snap_scale = memnew(EditorSpinSlider);
	snap_scale->set_min(0.0);
	snap_scale->set_step(1.0);
	snap_scale->set_max(100);
	snap_scale->set_suffix("%");
	snap_scale->set_accessibility_name(TTRC("Scale Snap"));
	snap_dialog_vbc->add_margin_child(TTR("Scale Snap:"), snap_scale);

	/* SETTINGS DIALOG */

	settings_dialog = memnew(ConfirmationDialog);
	settings_dialog->set_title(TTRC("Viewport Settings"));
	add_child(settings_dialog);
	settings_vbc = memnew(VBoxContainer);
	settings_vbc->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	settings_dialog->add_child(settings_vbc);

	settings_fov = memnew(SpinBox);
	settings_fov->set_max(MAX_FOV);
	settings_fov->set_min(MIN_FOV);
	settings_fov->set_step(0.1);
	settings_fov->set_value(EDITOR_GET("editors/3d/default_fov"));
	settings_fov->set_select_all_on_focus(true);
	settings_fov->set_tooltip_text(TTRC("FOV is defined as a vertical value, as the editor camera always uses the Keep Height aspect mode."));
	settings_fov->set_accessibility_name(TTRC("Perspective VFOV (deg.):"));
	settings_vbc->add_margin_child(TTRC("Perspective VFOV (deg.):"), settings_fov);

	settings_znear = memnew(SpinBox);
	settings_znear->set_max(MAX_Z);
	settings_znear->set_min(MIN_Z);
	settings_znear->set_step(0.01);
	settings_znear->set_accessibility_name(TTRC("View Z-Near:"));
	settings_znear->set_value(EDITOR_GET("editors/3d/default_z_near"));
	settings_znear->set_select_all_on_focus(true);
	settings_vbc->add_margin_child(TTRC("View Z-Near:"), settings_znear);

	settings_zfar = memnew(SpinBox);
	settings_zfar->set_max(MAX_Z);
	settings_zfar->set_min(MIN_Z);
	settings_zfar->set_step(0.1);
	settings_zfar->set_accessibility_name(TTRC("View Z-Far:"));
	settings_zfar->set_value(EDITOR_GET("editors/3d/default_z_far"));
	settings_zfar->set_select_all_on_focus(true);
	settings_vbc->add_margin_child(TTRC("View Z-Far:"), settings_zfar);

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; ++i) {
		settings_dialog->connect(SceneStringName(confirmed), callable_mp(viewports[i], &Node3DEditorViewport::_view_settings_confirmed).bind(0.0));
	}

	/* XFORM DIALOG */

	xform_dialog = memnew(ConfirmationDialog);
	xform_dialog->set_title(TTRC("Transform Change"));
	add_child(xform_dialog);

	VBoxContainer *xform_vbc = memnew(VBoxContainer);
	xform_dialog->add_child(xform_vbc);

	HBoxContainer *translate_hb = memnew(HBoxContainer);
	xform_vbc->add_margin_child(TTRC("Translate:"), translate_hb);
	HBoxContainer *rotate_hb = memnew(HBoxContainer);
	xform_vbc->add_margin_child(TTRC("Rotate (deg.):"), rotate_hb);
	HBoxContainer *scale_hb = memnew(HBoxContainer);
	xform_vbc->add_margin_child(TTRC("Scale (ratio):"), scale_hb);

	for (int i = 0; i < 3; i++) {
		xform_translate[i] = memnew(LineEdit);
		xform_translate[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_translate[i]->set_select_all_on_focus(true);
		translate_hb->add_child(xform_translate[i]);

		xform_rotate[i] = memnew(LineEdit);
		xform_rotate[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_rotate[i]->set_select_all_on_focus(true);
		rotate_hb->add_child(xform_rotate[i]);

		xform_scale[i] = memnew(LineEdit);
		xform_scale[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_scale[i]->set_select_all_on_focus(true);
		scale_hb->add_child(xform_scale[i]);
	}

	xform_type = memnew(OptionButton);
	xform_type->set_h_size_flags(SIZE_EXPAND_FILL);
	xform_type->set_accessibility_name(TTRC("Transform Type"));
	xform_type->add_item(TTRC("Pre"));
	xform_type->add_item(TTRC("Post"));
	xform_vbc->add_margin_child(TTRC("Transform Type"), xform_type);

	xform_dialog->connect(SceneStringName(confirmed), callable_mp(this, &Node3DEditor::_xform_dialog_action));

	selected = nullptr;

	set_process_shortcut_input(true);
	add_to_group(SceneStringName(_spatial_editor_group));

	current_hover_gizmo_handle = -1;
	current_hover_gizmo_handle_secondary = false;
	{
		// Sun/preview environment popup.
		sun_environ_popup = memnew(PreviewSunEnvPopup);
		add_child(sun_environ_popup);

		HBoxContainer *sun_environ_hb = memnew(HBoxContainer);

		sun_environ_popup->add_child(sun_environ_hb);

		sun_vb = memnew(VBoxContainer);
		sun_environ_hb->add_child(sun_vb);
		sun_vb->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
		sun_vb->hide();

		sun_title = memnew(Label);
		sun_title->set_theme_type_variation("HeaderMedium");
		sun_vb->add_child(sun_title);
		sun_title->set_text(TTRC("Preview Sun"));
		sun_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		CenterContainer *sun_direction_center = memnew(CenterContainer);
		sun_direction = memnew(Control);
		sun_direction->set_custom_minimum_size(Size2(128, 128) * EDSCALE);
		sun_direction_center->add_child(sun_direction);
		sun_vb->add_margin_child(TTRC("Sun Direction"), sun_direction_center);
		sun_direction->connect(SceneStringName(gui_input), callable_mp(this, &Node3DEditor::_sun_direction_input));
		sun_direction->connect(SceneStringName(draw), callable_mp(this, &Node3DEditor::_sun_direction_draw));
		sun_direction->set_default_cursor_shape(CURSOR_MOVE);

		sun_direction_shader.instantiate();
		sun_direction_shader->set_code(R"(
// 3D editor Preview Sun direction shader.

shader_type canvas_item;

uniform vec3 sun_direction;
uniform vec3 sun_color;

void fragment() {
	vec3 n;
	n.xy = UV * 2.0 - 1.0;
	n.z = sqrt(max(0.0, 1.0 - dot(n.xy, n.xy)));
	COLOR.rgb = dot(n, sun_direction) * sun_color;
	COLOR.a = 1.0 - smoothstep(0.99, 1.0, length(n.xy));
}
)");
		sun_direction_material.instantiate();
		sun_direction_material->set_shader(sun_direction_shader);
		sun_direction_material->set_shader_parameter("sun_direction", Vector3(0, 0, 1));
		sun_direction_material->set_shader_parameter("sun_color", Vector3(1, 1, 1));
		sun_direction->set_material(sun_direction_material);

		HBoxContainer *sun_angle_hbox = memnew(HBoxContainer);
		sun_angle_hbox->set_h_size_flags(SIZE_EXPAND_FILL);
		VBoxContainer *sun_angle_altitude_vbox = memnew(VBoxContainer);
		sun_angle_altitude_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		Label *sun_angle_altitude_label = memnew(Label);
		sun_angle_altitude_label->set_text(TTRC("Angular Altitude"));
		sun_angle_altitude_vbox->add_child(sun_angle_altitude_label);
		sun_angle_altitude = memnew(EditorSpinSlider);
		sun_angle_altitude->set_suffix(U"\u00B0");
		sun_angle_altitude->set_max(90);
		sun_angle_altitude->set_min(-90);
		sun_angle_altitude->set_step(0.1);
		sun_angle_altitude->connect(SceneStringName(value_changed), callable_mp(this, &Node3DEditor::_sun_direction_set_altitude));
		sun_angle_altitude_vbox->add_child(sun_angle_altitude);
		sun_angle_hbox->add_child(sun_angle_altitude_vbox);
		VBoxContainer *sun_angle_azimuth_vbox = memnew(VBoxContainer);
		sun_angle_azimuth_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		sun_angle_azimuth_vbox->set_custom_minimum_size(Vector2(100, 0));
		Label *sun_angle_azimuth_label = memnew(Label);
		sun_angle_azimuth_label->set_text(TTRC("Azimuth"));
		sun_angle_azimuth_vbox->add_child(sun_angle_azimuth_label);
		sun_angle_azimuth = memnew(EditorSpinSlider);
		sun_angle_azimuth->set_suffix(U"\u00B0");
		sun_angle_azimuth->set_max(180);
		sun_angle_azimuth->set_min(-180);
		sun_angle_azimuth->set_step(0.1);
		sun_angle_azimuth->set_allow_greater(true);
		sun_angle_azimuth->set_allow_lesser(true);
		sun_angle_azimuth->connect(SceneStringName(value_changed), callable_mp(this, &Node3DEditor::_sun_direction_set_azimuth));
		sun_angle_azimuth_vbox->add_child(sun_angle_azimuth);
		sun_angle_hbox->add_child(sun_angle_azimuth_vbox);
		sun_angle_hbox->add_theme_constant_override("separation", 10);
		sun_vb->add_child(sun_angle_hbox);

		sun_color = memnew(ColorPickerButton);
		sun_color->set_edit_alpha(false);
		sun_vb->add_margin_child(TTRC("Sun Color"), sun_color);
		sun_color->connect("color_changed", callable_mp(this, &Node3DEditor::_sun_set_color));
		sun_color->get_popup()->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(sun_color->get_picker()));

		sun_energy = memnew(EditorSpinSlider);
		sun_energy->set_max(64.0);
		sun_energy->set_min(0);
		sun_energy->set_step(0.05);
		sun_vb->add_margin_child(TTRC("Sun Energy"), sun_energy);
		sun_energy->connect(SceneStringName(value_changed), callable_mp(this, &Node3DEditor::_sun_set_energy));

		sun_shadow_max_distance = memnew(EditorSpinSlider);
		sun_vb->add_margin_child(TTRC("Shadow Max Distance"), sun_shadow_max_distance);
		sun_shadow_max_distance->connect(SceneStringName(value_changed), callable_mp(this, &Node3DEditor::_sun_set_shadow_max_distance));
		sun_shadow_max_distance->set_min(1);
		sun_shadow_max_distance->set_max(4096);

		sun_add_to_scene = memnew(Button);
		sun_add_to_scene->set_text(TTRC("Add Sun to Scene"));
		sun_add_to_scene->set_tooltip_text(TTRC("Adds a DirectionalLight3D node matching the preview sun settings to the current scene.\nHold Shift while clicking to also add the preview environment to the current scene."));
		sun_add_to_scene->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_add_sun_to_scene).bind(false));
		sun_vb->add_spacer();
		sun_vb->add_child(sun_add_to_scene);

		sun_state = memnew(Label);
		sun_environ_hb->add_child(sun_state);
		sun_state->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		sun_state->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
		sun_state->set_h_size_flags(SIZE_EXPAND_FILL);

		VSeparator *sc = memnew(VSeparator);
		sc->set_custom_minimum_size(Size2(10 * EDSCALE, 0));
		sc->set_v_size_flags(SIZE_EXPAND_FILL);
		sun_environ_hb->add_child(sc);

		environ_vb = memnew(VBoxContainer);
		sun_environ_hb->add_child(environ_vb);
		environ_vb->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
		environ_vb->hide();

		environ_title = memnew(Label);
		environ_title->set_theme_type_variation("HeaderMedium");

		environ_vb->add_child(environ_title);
		environ_title->set_text(TTRC("Preview Environment"));
		environ_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		environ_sky_color = memnew(ColorPickerButton);
		environ_sky_color->set_edit_alpha(false);
		environ_sky_color->connect("color_changed", callable_mp(this, &Node3DEditor::_environ_set_sky_color));
		environ_sky_color->get_popup()->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(environ_sky_color->get_picker()));
		environ_vb->add_margin_child(TTRC("Sky Color"), environ_sky_color);
		environ_ground_color = memnew(ColorPickerButton);
		environ_ground_color->connect("color_changed", callable_mp(this, &Node3DEditor::_environ_set_ground_color));
		environ_ground_color->set_edit_alpha(false);
		environ_ground_color->get_popup()->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(environ_ground_color->get_picker()));
		environ_vb->add_margin_child(TTRC("Ground Color"), environ_ground_color);
		environ_energy = memnew(EditorSpinSlider);
		environ_energy->set_max(8.0);
		environ_energy->set_min(0);
		environ_energy->set_step(0.05);
		environ_energy->connect(SceneStringName(value_changed), callable_mp(this, &Node3DEditor::_environ_set_sky_energy));
		environ_vb->add_margin_child(TTRC("Sky Energy"), environ_energy);
		HBoxContainer *fx_vb = memnew(HBoxContainer);
		fx_vb->set_h_size_flags(SIZE_EXPAND_FILL);

		environ_ao_button = memnew(Button);
		environ_ao_button->set_text(TTRC("AO"));
		environ_ao_button->set_h_size_flags(SIZE_EXPAND_FILL);
		environ_ao_button->set_toggle_mode(true);
		environ_ao_button->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_environ_set_ao), CONNECT_DEFERRED);
		fx_vb->add_child(environ_ao_button);
		environ_glow_button = memnew(Button);
		environ_glow_button->set_text(TTRC("Glow"));
		environ_glow_button->set_h_size_flags(SIZE_EXPAND_FILL);
		environ_glow_button->set_toggle_mode(true);
		environ_glow_button->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_environ_set_glow), CONNECT_DEFERRED);
		fx_vb->add_child(environ_glow_button);
		environ_tonemap_button = memnew(Button);
		environ_tonemap_button->set_text(TTRC("Tonemap"));
		environ_tonemap_button->set_h_size_flags(SIZE_EXPAND_FILL);
		environ_tonemap_button->set_toggle_mode(true);
		environ_tonemap_button->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_environ_set_tonemap), CONNECT_DEFERRED);
		fx_vb->add_child(environ_tonemap_button);
		environ_gi_button = memnew(Button);
		environ_gi_button->set_text(TTRC("GI"));
		environ_gi_button->set_h_size_flags(SIZE_EXPAND_FILL);
		environ_gi_button->set_toggle_mode(true);
		environ_gi_button->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_environ_set_gi), CONNECT_DEFERRED);
		fx_vb->add_child(environ_gi_button);
		environ_vb->add_margin_child(TTRC("Post Process"), fx_vb);

		environ_add_to_scene = memnew(Button);
		environ_add_to_scene->set_text(TTRC("Add Environment to Scene"));
		environ_add_to_scene->set_tooltip_text(TTRC("Adds a WorldEnvironment node matching the preview environment settings to the current scene.\nHold Shift while clicking to also add the preview sun to the current scene."));
		environ_add_to_scene->connect(SceneStringName(pressed), callable_mp(this, &Node3DEditor::_add_environment_to_scene).bind(false));
		environ_vb->add_spacer();
		environ_vb->add_child(environ_add_to_scene);

		environ_state = memnew(Label);
		sun_environ_hb->add_child(environ_state);
		environ_state->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		environ_state->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
		environ_state->set_h_size_flags(SIZE_EXPAND_FILL);

		preview_sun = memnew(DirectionalLight3D);
		preview_sun->set_shadow(true);
		preview_sun->set_shadow_mode(DirectionalLight3D::SHADOW_PARALLEL_4_SPLITS);
		preview_environment = memnew(WorldEnvironment);
		environment.instantiate();
		preview_environment->set_environment(environment);
		if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
			camera_attributes.instantiate();
			preview_environment->set_camera_attributes(camera_attributes);
		}
		Ref<Sky> sky;
		sky.instantiate();
		sky_material.instantiate();
		sky->set_material(sky_material);
		environment->set_sky(sky);
		environment->set_background(Environment::BG_SKY);

		sun_environ_popup->set_process_shortcut_input(true);

		_load_default_preview_settings();
		_preview_settings_changed();
	}
	clear(); // Make sure values are initialized. Will call _snap_update() for us.
}
Node3DEditor::~Node3DEditor() {
	singleton = nullptr;
	memdelete(preview_node);
	if (preview_sun_dangling && preview_sun) {
		memdelete(preview_sun);
	}
	if (preview_env_dangling && preview_environment) {
		memdelete(preview_environment);
	}
}

void Node3DEditorPlugin::edited_scene_changed() {
	for (uint32_t i = 0; i < Node3DEditor::VIEWPORTS_COUNT; i++) {
		Node3DEditorViewport *viewport = Node3DEditor::get_singleton()->get_editor_viewport(i);
		if (viewport->is_visible()) {
			viewport->notification(Control::NOTIFICATION_VISIBILITY_CHANGED);
		}
	}
}

void Node3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		spatial_editor->show();
		spatial_editor->set_process(true);
		spatial_editor->set_physics_process(true);
		spatial_editor->refresh_dirty_gizmos();
	} else {
		spatial_editor->hide();
		spatial_editor->set_process(false);
		spatial_editor->set_physics_process(false);
	}
}

void Node3DEditorPlugin::edit(Object *p_object) {
	spatial_editor->edit(Object::cast_to<Node3D>(p_object));
}

bool Node3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Node3D");
}

Dictionary Node3DEditorPlugin::get_state() const {
	return spatial_editor->get_state();
}

void Node3DEditorPlugin::set_state(const Dictionary &p_state) {
	spatial_editor->set_state(p_state);
}

Size2i Node3DEditor::get_camera_viewport_size(Camera3D *p_camera) {
	Viewport *viewport = p_camera->get_viewport();

	Window *window = Object::cast_to<Window>(viewport);
	if (window) {
		return window->get_size();
	}

	SubViewport *sub_viewport = Object::cast_to<SubViewport>(viewport);
	ERR_FAIL_NULL_V(sub_viewport, Size2i());

	if (sub_viewport == EditorNode::get_singleton()->get_scene_root()) {
		return Size2(GLOBAL_GET("display/window/size/viewport_width"), GLOBAL_GET("display/window/size/viewport_height"));
	}

	return sub_viewport->get_size();
}

Vector3 Node3DEditor::snap_point(Vector3 p_target, Vector3 p_start) const {
	if (is_snap_enabled()) {
		real_t snap = get_translate_snap();
		p_target.snapf(snap);
	}
	return p_target;
}

float Node3DEditor::get_znear() const {
	return settings_znear->get_value();
}

float Node3DEditor::get_zfar() const {
	return settings_zfar->get_value();
}

float Node3DEditor::get_fov() const {
	return settings_fov->get_value();
}

bool Node3DEditor::is_gizmo_visible() const {
	if (selected) {
		return gizmo.visible && selected->is_transform_gizmo_visible();
	}
	return gizmo.visible;
}

bool Node3DEditor::are_local_coords_enabled() const {
	return tool_option_button[Node3DEditor::TOOL_OPT_LOCAL_COORDS]->is_pressed();
}

void Node3DEditor::set_local_coords_enabled(bool on) const {
	tool_option_button[Node3DEditor::TOOL_OPT_LOCAL_COORDS]->set_pressed(on);
}

bool Node3DEditor::is_preserve_children_transform_enabled() const {
	return tool_option_button[Node3DEditor::TOOL_OPT_PRESERVE_CHILDREN_TRANSFORM]->is_pressed();
}

bool Node3DEditor::is_vertex_snap_use_collision() const {
	return vertex_snap_use_collision != Input::get_singleton()->is_key_pressed(Key::SHIFT);
}

real_t Node3DEditor::get_translate_snap() const {
	real_t snap_value = snap_translate_value;
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		snap_value /= 10.0f;
	}
	return snap_value;
}

real_t Node3DEditor::get_rotate_snap() const {
	real_t snap_value = snap_rotate_value;
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		snap_value /= 3.0f;
	}
	return snap_value;
}

real_t Node3DEditor::get_scale_snap() const {
	real_t snap_value = snap_scale_value;
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		snap_value /= 2.0f;
	}
	return snap_value;
}

struct _GizmoPluginPriorityComparator {
	bool operator()(const Ref<EditorNode3DGizmoPlugin> &p_a, const Ref<EditorNode3DGizmoPlugin> &p_b) const {
		if (p_a->get_priority() == p_b->get_priority()) {
			return p_a->get_gizmo_name() < p_b->get_gizmo_name();
		}
		return p_a->get_priority() > p_b->get_priority();
	}
};

struct _GizmoPluginNameComparator {
	bool operator()(const Ref<EditorNode3DGizmoPlugin> &p_a, const Ref<EditorNode3DGizmoPlugin> &p_b) const {
		return p_a->get_gizmo_name() < p_b->get_gizmo_name();
	}
};

void Node3DEditor::add_gizmo_plugin(Ref<EditorNode3DGizmoPlugin> p_plugin) {
	ERR_FAIL_COND(p_plugin.is_null());

	gizmo_plugins_by_priority.push_back(p_plugin);
	gizmo_plugins_by_priority.sort_custom<_GizmoPluginPriorityComparator>();

	gizmo_plugins_by_name.push_back(p_plugin);
	gizmo_plugins_by_name.sort_custom<_GizmoPluginNameComparator>();

	_update_gizmos_menu();
}

void Node3DEditor::remove_gizmo_plugin(Ref<EditorNode3DGizmoPlugin> p_plugin) {
	gizmo_plugins_by_priority.erase(p_plugin);
	gizmo_plugins_by_name.erase(p_plugin);
	_update_gizmos_menu();
}

DynamicBVH::ID Node3DEditor::insert_gizmo_bvh_node(Node3D *p_node, const AABB &p_aabb) {
	return gizmo_bvh.insert(p_aabb, p_node);
}

void Node3DEditor::update_gizmo_bvh_node(DynamicBVH::ID p_id, const AABB &p_aabb) {
	gizmo_bvh.update(p_id, p_aabb);
	gizmo_bvh.optimize_incremental(1);
}

void Node3DEditor::remove_gizmo_bvh_node(DynamicBVH::ID p_id) {
	gizmo_bvh.remove(p_id);
}

Vector<Node3D *> Node3DEditor::gizmo_bvh_ray_query(const Vector3 &p_ray_start, const Vector3 &p_ray_end) {
	struct Result {
		Vector<Node3D *> nodes;
		bool operator()(void *p_data) {
			nodes.append((Node3D *)p_data);
			return false;
		}
	} result;

	gizmo_bvh.ray_query(p_ray_start, p_ray_end, result);

	return result.nodes;
}

Vector<Node3D *> Node3DEditor::gizmo_bvh_frustum_query(const Vector<Plane> &p_frustum) {
	Vector<Vector3> points = Geometry3D::compute_convex_mesh_points(&p_frustum[0], p_frustum.size());

	struct Result {
		Vector<Node3D *> nodes;
		bool operator()(void *p_data) {
			nodes.append((Node3D *)p_data);
			return false;
		}
	} result;

	gizmo_bvh.convex_query(p_frustum.ptr(), p_frustum.size(), points.ptr(), points.size(), result);

	return result.nodes;
}

Node3DEditorPlugin::Node3DEditorPlugin() {
	spatial_editor = memnew(Node3DEditor);
	spatial_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(spatial_editor);

	spatial_editor->hide();
}
