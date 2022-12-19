/*************************************************************************/
/*  node_3d_editor.cpp                                                   */
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

#include "node_3d_editor.h"

#include "core/config/project_settings.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene_tree_dock.h"
#include "node_3d_editor_selected_item.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/gui/center_container.h"
#include "scene/gui/flow_container.h"
#include "scene/resources/surface_tool.h"

Node3DEditor *Node3DEditor::singleton = nullptr;

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
	ERR_FAIL_NULL(p_plugin.ptr());

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

Vector3 Node3DEditor::snap_point(Vector3 p_target, Vector3 p_start) const {
	if (is_snap_enabled()) {
		p_target.x = Math::snap_scalar(0.0, get_translate_snap(), p_target.x);
		p_target.y = Math::snap_scalar(0.0, get_translate_snap(), p_target.y);
		p_target.z = Math::snap_scalar(0.0, get_translate_snap(), p_target.z);
	}
	return p_target;
}

bool Node3DEditor::is_gizmo_visible() const {
	if (selected) {
		return gizmo.visible && selected->is_transform_gizmo_visible();
	}
	return gizmo.visible;
}

double Node3DEditor::get_translate_snap() const {
	double snap_value;
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		snap_value = snap_translate->get_text().to_float() / 10.0;
	} else {
		snap_value = snap_translate->get_text().to_float();
	}

	return snap_value;
}

double Node3DEditor::get_rotate_snap() const {
	double snap_value;
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		snap_value = snap_rotate->get_text().to_float() / 3.0;
	} else {
		snap_value = snap_rotate->get_text().to_float();
	}

	return snap_value;
}

double Node3DEditor::get_scale_snap() const {
	double snap_value;
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		snap_value = snap_scale->get_text().to_float() / 2.0;
	} else {
		snap_value = snap_scale->get_text().to_float();
	}

	return snap_value;
}

void Node3DEditor::select_gizmo_highlight_axis(int p_axis) {
	for (int i = 0; i < 3; i++) {
		move_gizmo[i]->surface_set_material(0, i == p_axis ? gizmo_color_hl[i] : gizmo_color[i]);
		move_plane_gizmo[i]->surface_set_material(0, (i + 6) == p_axis ? plane_gizmo_color_hl[i] : plane_gizmo_color[i]);
		rotate_gizmo[i]->surface_set_material(0, (i + 3) == p_axis ? rotate_gizmo_color_hl[i] : rotate_gizmo_color[i]);
		scale_gizmo[i]->surface_set_material(0, (i + 9) == p_axis ? gizmo_color_hl[i] : gizmo_color[i]);
		scale_plane_gizmo[i]->surface_set_material(0, (i + 12) == p_axis ? plane_gizmo_color_hl[i] : plane_gizmo_color[i]);
	}
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
			gizmo_center += xf.origin;
			if (count == 0 && local_gizmo_coords) {
				gizmo_basis = xf.basis;
			}
			count++;
		}
	} else {
		List<Node *> &selection = editor_selection->get_selected_node_list();
		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
			Node3D *sp = Object::cast_to<Node3D>(E->get());
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
			gizmo_center += xf.origin;
			if (count == 0 && local_gizmo_coords) {
				gizmo_basis = xf.basis;
			}
			count++;
		}
	}

	gizmo.visible = count > 0;
	gizmo.transform.origin = (count > 0) ? gizmo_center / count : Vector3();
	gizmo.transform.basis = (count == 1) ? gizmo_basis : Basis();

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
			RS::SHADOW_CASTING_SETTING_OFF);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance_offset,
			RS::SHADOW_CASTING_SETTING_OFF);
	// Use the Edit layer to hide the selection box when View Gizmos is disabled, since it is a bit distracting.
	// It's still possible to approximately guess what is selected by looking at the manipulation gizmo position.
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance_offset, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_offset, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_offset, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	si->sbox_instance_xray = RenderingServer::get_singleton()->instance_create2(
			selection_box_xray->get_rid(),
			sp->get_world_3d()->get_scenario());
	si->sbox_instance_xray_offset = RenderingServer::get_singleton()->instance_create2(
			selection_box_xray->get_rid(),
			sp->get_world_3d()->get_scenario());
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance_xray,
			RS::SHADOW_CASTING_SETTING_OFF);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(
			si->sbox_instance_xray_offset,
			RS::SHADOW_CASTING_SETTING_OFF);
	// Use the Edit layer to hide the selection box when View Gizmos is disabled, since it is a bit distracting.
	// It's still possible to approximately guess what is selected by looking at the manipulation gizmo position.
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance_xray, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_set_layer_mask(si->sbox_instance_xray_offset, 1 << Node3DEditorViewport::GIZMO_EDIT_LAYER);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray_offset, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(si->sbox_instance_xray_offset, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

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

	st->begin(Mesh::PRIMITIVE_LINES);
	st_xray->begin(Mesh::PRIMITIVE_LINES);
	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);

		st->add_vertex(a);
		st->add_vertex(b);
		st_xray->add_vertex(a);
		st_xray->add_vertex(b);
	}

	Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
	mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	const Color selection_box_color = EDITOR_GET("editors/3d/selection_box_color");
	mat->set_albedo(selection_box_color);
	mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st->set_material(mat);
	selection_box = st->commit();

	Ref<StandardMaterial3D> mat_xray = memnew(StandardMaterial3D);
	mat_xray->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	mat_xray->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);
	mat_xray->set_albedo(selection_box_color * Color(1, 1, 1, 0.15));
	mat_xray->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	st_xray->set_material(mat_xray);
	selection_box_xray = st_xray->commit();
}

Dictionary Node3DEditor::get_state() const {
	Dictionary d;

	d["snap_enabled"] = snap_enabled;
	d["translate_snap"] = get_translate_snap();
	d["rotate_snap"] = get_rotate_snap();
	d["scale_snap"] = get_scale_snap();

	d["local_coords"] = tool_option_button[TOOL_OPT_LOCAL_COORDS]->is_pressed();

	int vc = 0;
	if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT))) {
		vc = 1;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS))) {
		vc = 2;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS))) {
		vc = 3;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS))) {
		vc = 4;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT))) {
		vc = 5;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT))) {
		vc = 6;
	}

	d["viewport_mode"] = vc;
	Array vpdata;
	for (int i = 0; i < 4; i++) {
		vpdata.push_back(viewports[i]->get_state());
	}

	d["viewports"] = vpdata;

	d["show_grid"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_GRID));
	d["show_origin"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN));
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
		pd["sun_max_distance"] = sun_max_distance->get_value();

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

	if (d.has("zfar")) {
		settings_zfar->set_value(double(d["zfar"]));
	}
	if (d.has("znear")) {
		settings_znear->set_value(double(d["znear"]));
	}
	if (d.has("fov")) {
		settings_fov->set_value(double(d["fov"]));
	}
	if (d.has("show_grid")) {
		bool use = d["show_grid"];

		if (use != view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_GRID))) {
			_menu_item_pressed(MENU_VIEW_GRID);
		}
	}

	if (d.has("show_origin")) {
		bool use = d["show_origin"];

		if (use != view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN))) {
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN), use);
			RenderingServer::get_singleton()->instance_set_visible(origin_instance, use);
		}
	}

	if (d.has("gizmos_status")) {
		Dictionary gizmos_status = d["gizmos_status"];
		List<Variant> keys;
		gizmos_status.get_key_list(&keys);

		for (int j = 0; j < gizmo_plugins_by_name.size(); ++j) {
			if (!gizmo_plugins_by_name[j]->can_be_hidden()) {
				continue;
			}
			int state = EditorNode3DGizmoPlugin::VISIBLE;
			for (int i = 0; i < keys.size(); i++) {
				if (gizmo_plugins_by_name.write[j]->get_gizmo_name() == String(keys[i])) {
					state = gizmos_status[keys[i]];
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
		environ_energy->set_value(pd["environ_energy"]);
		environ_glow_button->set_pressed(pd["environ_glow_enabled"]);
		environ_tonemap_button->set_pressed(pd["environ_tonemap_enabled"]);
		environ_ao_button->set_pressed(pd["environ_ao_enabled"]);
		environ_gi_button->set_pressed(pd["environ_gi_enabled"]);
		sun_max_distance->set_value(pd["sun_max_distance"]);

		sun_color->set_pick_color(pd["sun_color"]);
		sun_energy->set_value(pd["sun_energy"]);

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
				if (!seg.is_valid()) {
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
				if (!seg.is_valid()) {
					continue;
				}
				seg->set_selected(true);
			}
			selected->update_gizmos();
		}
	}
}

void Node3DEditor::_snap_changed() {
	snap_translate_value = snap_translate->get_text().to_float();
	snap_rotate_value = snap_rotate->get_text().to_float();
	snap_scale_value = snap_scale->get_text().to_float();
}

void Node3DEditor::_snap_update() {
	snap_translate->set_text(String::num(snap_translate_value));
	snap_rotate->set_text(String::num(snap_rotate_value));
	snap_scale->set_text(String::num(snap_scale_value));
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

	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
	undo_redo->create_action(TTR("XForm Dialog"));

	const List<Node *> &selection = editor_selection->get_selected_node_list();

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

		undo_redo->add_do_method(sp, "set_global_transform", tr);
		undo_redo->add_undo_method(sp, "set_global_transform", sp->get_global_gizmo_transform());
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

		case MENU_TOOL_OVERRIDE_CAMERA: {
			EditorDebuggerNode *const debugger = EditorDebuggerNode::get_singleton();

			using Override = EditorDebuggerNode::CameraOverride;
			if (pressed) {
				debugger->set_camera_override((Override)(Override::OVERRIDE_3D_1 + camera_override_viewport_id));
			} else {
				debugger->set_camera_override(Override::OVERRIDE_NONE);
			}

		} break;
	}
}

void Node3DEditor::_menu_gizmo_toggled(int p_option) {
	const int idx = gizmos_menu->get_item_index(p_option);
	gizmos_menu->toggle_item_multistate(idx);

	// Change icon
	const int state = gizmos_menu->get_item_state(idx);
	switch (state) {
		case EditorNode3DGizmoPlugin::VISIBLE:
			gizmos_menu->set_item_icon(idx, view_menu->get_popup()->get_theme_icon(SNAME("visibility_visible")));
			break;
		case EditorNode3DGizmoPlugin::ON_TOP:
			gizmos_menu->set_item_icon(idx, view_menu->get_popup()->get_theme_icon(SNAME("visibility_xray")));
			break;
		case EditorNode3DGizmoPlugin::HIDDEN:
			gizmos_menu->set_item_icon(idx, view_menu->get_popup()->get_theme_icon(SNAME("visibility_hidden")));
			break;
	}

	gizmo_plugins_by_name.write[p_option]->set_state(state);

	update_all_gizmos();
}

void Node3DEditor::_update_camera_override_button(bool p_game_running) {
	Button *const button = tool_option_button[TOOL_OPT_OVERRIDE_CAMERA];

	if (p_game_running) {
		button->set_disabled(false);
		button->set_tooltip_text(TTR("Project Camera Override\nOverrides the running project's camera with the editor viewport camera."));
	} else {
		button->set_disabled(true);
		button->set_pressed(false);
		button->set_tooltip_text(TTR("Project Camera Override\nNo project instance running. Run the project from the editor to use this feature."));
	}
}

void Node3DEditor::_update_camera_override_viewport(Object *p_viewport) {
	Node3DEditorViewport *current_viewport = Object::cast_to<Node3DEditorViewport>(p_viewport);

	if (!current_viewport) {
		return;
	}

	EditorDebuggerNode *const debugger = EditorDebuggerNode::get_singleton();

	camera_override_viewport_id = current_viewport->index;
	if (debugger->get_camera_override() >= EditorDebuggerNode::OVERRIDE_3D_1) {
		using Override = EditorDebuggerNode::CameraOverride;

		debugger->set_camera_override((Override)(Override::OVERRIDE_3D_1 + camera_override_viewport_id));
	}
}

void Node3DEditor::_menu_item_pressed(int p_option) {
	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
	switch (p_option) {
		case MENU_TOOL_SELECT:
		case MENU_TOOL_MOVE:
		case MENU_TOOL_ROTATE:
		case MENU_TOOL_SCALE:
		case MENU_TOOL_LIST_SELECT: {
			for (int i = 0; i < TOOL_MAX; i++) {
				tool_button[i]->set_pressed(i == p_option);
			}
			tool_mode = (ToolMode)p_option;
			update_transform_gizmo();

		} break;
		case MENU_TRANSFORM_CONFIGURE_SNAP: {
			snap_dialog->popup_centered(Size2(200, 180));
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

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_2_VIEWPORTS: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_2_VIEWPORTS);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_2_VIEWPORTS_ALT: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_2_VIEWPORTS_ALT);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_3_VIEWPORTS: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_3_VIEWPORTS);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_3_VIEWPORTS_ALT: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_3_VIEWPORTS_ALT);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), true);

		} break;
		case MENU_VIEW_USE_4_VIEWPORTS: {
			viewport_base->set_view(Node3DEditorViewportContainer::VIEW_USE_4_VIEWPORTS);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_ORIGIN: {
			bool is_checked = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(p_option));

			origin_enabled = !is_checked;
			RenderingServer::get_singleton()->instance_set_visible(origin_instance, origin_enabled);
			// Update the grid since its appearance depends on whether the origin is enabled
			_finish_grid();
			_init_grid();

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(p_option), origin_enabled);
		} break;
		case MENU_VIEW_GRID: {
			bool is_checked = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(p_option));

			grid_enabled = !is_checked;

			for (int i = 0; i < 3; ++i) {
				if (grid_enable[i]) {
					grid_visible[i] = grid_enabled;
				}
			}
			_finish_grid();
			_init_grid();

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(p_option), grid_enabled);

		} break;
		case MENU_VIEW_CAMERA_SETTINGS: {
			settings_dialog->popup_centered(settings_vbc->get_combined_minimum_size() + Size2(50, 50));
		} break;
		case MENU_SNAP_TO_FLOOR: {
			snap_selected_nodes_to_floor();
		} break;
		case MENU_LOCK_SELECTED: {
			undo_redo->create_action(TTR("Lock Selected"));

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				if (spatial->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				undo_redo->add_do_method(spatial, "set_meta", "_edit_lock_", true);
				undo_redo->add_undo_method(spatial, "remove_meta", "_edit_lock_");
				undo_redo->add_do_method(this, "emit_signal", "item_lock_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_lock_status_changed");
			}

			undo_redo->add_do_method(this, "_refresh_menu_icons");
			undo_redo->add_undo_method(this, "_refresh_menu_icons");
			undo_redo->commit_action();
		} break;
		case MENU_UNLOCK_SELECTED: {
			undo_redo->create_action(TTR("Unlock Selected"));

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				if (spatial->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
					continue;
				}

				undo_redo->add_do_method(spatial, "remove_meta", "_edit_lock_");
				undo_redo->add_undo_method(spatial, "set_meta", "_edit_lock_", true);
				undo_redo->add_do_method(this, "emit_signal", "item_lock_status_changed");
				undo_redo->add_undo_method(this, "emit_signal", "item_lock_status_changed");
			}

			undo_redo->add_do_method(this, "_refresh_menu_icons");
			undo_redo->add_undo_method(this, "_refresh_menu_icons");
			undo_redo->commit_action();
		} break;
		case MENU_GROUP_SELECTED: {
			undo_redo->create_action(TTR("Group Selected"));

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				if (spatial->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
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
			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (Node *E : selection) {
				Node3D *spatial = Object::cast_to<Node3D>(E);
				if (!spatial || !spatial->is_inside_tree()) {
					continue;
				}

				if (spatial->get_viewport() != EditorNode::get_singleton()->get_scene_root()) {
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
	}
}

void Node3DEditor::_init_indicators() {
	{
		origin_enabled = true;
		grid_enabled = true;

		indicator_mat.instantiate();
		indicator_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		indicator_mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		indicator_mat->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
		indicator_mat->set_transparency(StandardMaterial3D::Transparency::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS);

		Vector<Color> origin_colors;
		Vector<Vector3> origin_points;

		const int count_of_elements = 3 * 6;
		origin_colors.resize(count_of_elements);
		origin_points.resize(count_of_elements);

		int x = 0;

		for (int i = 0; i < 3; i++) {
			Vector3 axis;
			axis[i] = 1;
			Color origin_color;
			switch (i) {
				case 0:
					origin_color = get_theme_color(SNAME("axis_x_color"), SNAME("Editor"));
					break;
				case 1:
					origin_color = get_theme_color(SNAME("axis_y_color"), SNAME("Editor"));
					break;
				case 2:
					origin_color = get_theme_color(SNAME("axis_z_color"), SNAME("Editor"));
					break;
				default:
					origin_color = Color();
					break;
			}

			grid_enable[i] = false;
			grid_visible[i] = false;

			origin_colors.set(x, origin_color);
			origin_colors.set(x + 1, origin_color);
			origin_colors.set(x + 2, origin_color);
			origin_colors.set(x + 3, origin_color);
			origin_colors.set(x + 4, origin_color);
			origin_colors.set(x + 5, origin_color);
			// To both allow having a large origin size and avoid jitter
			// at small scales, we should segment the line into pieces.
			// 3 pieces seems to do the trick, and let's use powers of 2.
			origin_points.set(x, axis * 1048576);
			origin_points.set(x + 1, axis * 1024);
			origin_points.set(x + 2, axis * 1024);
			origin_points.set(x + 3, axis * -1024);
			origin_points.set(x + 4, axis * -1024);
			origin_points.set(x + 5, axis * -1048576);
			x += 6;
		}

		Ref<Shader> grid_shader = memnew(Shader);
		grid_shader->set_code(R"(
// 3D editor grid shader.

shader_type spatial;

render_mode unshaded;

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

		origin = RenderingServer::get_singleton()->mesh_create();
		Array d;
		d.resize(RS::ARRAY_MAX);
		d[RenderingServer::ARRAY_VERTEX] = origin_points;
		d[RenderingServer::ARRAY_COLOR] = origin_colors;

		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(origin, RenderingServer::PRIMITIVE_LINES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(origin, 0, indicator_mat->get_rid());

		origin_instance = RenderingServer::get_singleton()->instance_create2(origin, get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_layer_mask(origin_instance, 1 << Node3DEditorViewport::GIZMO_GRID_LAYER);
		RS::get_singleton()->instance_geometry_set_flag(origin_instance, RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(origin_instance, RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		RenderingServer::get_singleton()->instance_geometry_set_cast_shadows_setting(origin_instance, RS::SHADOW_CASTING_SETTING_OFF);
	}

	{
		//move gizmo

		// Inverted zxy.
		Vector3 ivec = Vector3(0, 0, -1);
		Vector3 nivec = Vector3(-1, -1, 0);
		Vector3 ivec2 = Vector3(-1, 0, 0);
		Vector3 ivec3 = Vector3(0, -1, 0);

		for (int i = 0; i < 3; i++) {
			Color col;
			switch (i) {
				case 0:
					col = get_theme_color(SNAME("axis_x_color"), SNAME("Editor"));
					break;
				case 1:
					col = get_theme_color(SNAME("axis_y_color"), SNAME("Editor"));
					break;
				case 2:
					col = get_theme_color(SNAME("axis_z_color"), SNAME("Editor"));
					break;
				default:
					col = Color();
					break;
			}

			col.a = EDITOR_GET("editors/3d/manipulator_gizmo_opacity");

			move_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			move_plane_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			rotate_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			scale_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			scale_plane_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			axis_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));

			Ref<StandardMaterial3D> mat = memnew(StandardMaterial3D);
			mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
			mat->set_on_top_of_alpha();
			mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			mat->set_albedo(col);
			gizmo_color[i] = mat;

			Ref<StandardMaterial3D> mat_hl = mat->duplicate();
			const Color albedo = col.from_hsv(col.get_h(), 0.25, 1.0, 1);
			mat_hl->set_albedo(albedo);
			gizmo_color_hl[i] = mat_hl;

			//translate
			{
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
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

				const real_t arrow_sides_step = Math_TAU / arrow_sides;
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
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				Vector3 vec = ivec2 - ivec3;
				Vector3 plane[4] = {
					vec * GIZMO_PLANE_DST,
					vec * GIZMO_PLANE_DST + ivec2 * GIZMO_PLANE_SIZE,
					vec * (GIZMO_PLANE_DST + GIZMO_PLANE_SIZE),
					vec * GIZMO_PLANE_DST - ivec3 * GIZMO_PLANE_SIZE
				};

				Basis ma(ivec, Math_PI / 2);

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

				Ref<StandardMaterial3D> plane_mat = memnew(StandardMaterial3D);
				plane_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
				plane_mat->set_on_top_of_alpha();
				plane_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
				plane_mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
				plane_mat->set_albedo(col);
				plane_gizmo_color[i] = plane_mat; // needed, so we can draw planes from both sides
				surftool->set_material(plane_mat);
				surftool->commit(move_plane_gizmo[i]);

				Ref<StandardMaterial3D> plane_mat_hl = plane_mat->duplicate();
				plane_mat_hl->set_albedo(albedo);
				plane_gizmo_color_hl[i] = plane_mat_hl; // needed, so we can draw planes from both sides
			}

			// Rotate
			{
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				int n = 128; // number of circle segments
				int m = 3; // number of thickness segments

				real_t step = Math_TAU / n;
				for (int j = 0; j < n; ++j) {
					Basis basis = Basis(ivec, j * step);

					Vector3 vertex = basis.xform(ivec2 * GIZMO_CIRCLE_SIZE);

					for (int k = 0; k < m; ++k) {
						Vector2 ofs = Vector2(Math::cos((Math_TAU * k) / m), Math::sin((Math_TAU * k) / m));
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

				rotate_shader->set_code(R"(
// 3D editor rotation manipulator gizmo shader.

shader_type spatial;

render_mode unshaded, depth_test_disabled;

uniform vec4 albedo;

mat3 orthonormalize(mat3 m) {
	vec3 x = normalize(m[0]);
	vec3 y = normalize(m[1] - x * dot(x, m[1]));
	vec3 z = m[2] - x * dot(x, m[2]);
	z = normalize(z - y * (dot(y,m[2])));
	return mat3(x,y,z);
}

void vertex() {
	mat3 mv = orthonormalize(mat3(MODELVIEW_MATRIX));
	vec3 n = mv * VERTEX;
	float orientation = dot(vec3(0, 0, -1), n);
	if (orientation <= 0.005) {
		VERTEX += NORMAL * 0.02;
	}
}

void fragment() {
	ALBEDO = albedo.rgb;
	ALPHA = albedo.a;
}
)");

				Ref<ShaderMaterial> rotate_mat = memnew(ShaderMaterial);
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

				if (i == 2) { // Rotation white outline
					Ref<ShaderMaterial> border_mat = rotate_mat->duplicate();

					Ref<Shader> border_shader = memnew(Shader);
					border_shader->set_code(R"(
// 3D editor rotation manipulator gizmo shader (white outline).

shader_type spatial;

render_mode unshaded, depth_test_disabled;

uniform vec4 albedo;

mat3 orthonormalize(mat3 m) {
	vec3 x = normalize(m[0]);
	vec3 y = normalize(m[1] - x * dot(x, m[1]));
	vec3 z = m[2] - x * dot(x, m[2]);
	z = normalize(z - y * (dot(y,m[2])));
	return mat3(x,y,z);
}

void vertex() {
	mat3 mv = orthonormalize(mat3(MODELVIEW_MATRIX));
	mv = inverse(mv);
	VERTEX += NORMAL*0.008;
	vec3 camera_dir_local = mv * vec3(0,0,1);
	vec3 camera_up_local = mv * vec3(0,1,0);
	mat3 rotation_matrix = mat3(cross(camera_dir_local, camera_up_local), camera_up_local, camera_dir_local);
	VERTEX = rotation_matrix * VERTEX;
}

void fragment() {
	ALBEDO = albedo.rgb;
	ALPHA = albedo.a;
}
)");

					border_mat->set_shader(border_shader);
					border_mat->set_shader_parameter("albedo", Color(0.75, 0.75, 0.75, col.a / 3.0));

					rotate_gizmo[3] = Ref<ArrayMesh>(memnew(ArrayMesh));
					rotate_gizmo[3]->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);
					rotate_gizmo[3]->surface_set_material(0, border_mat);
				}
			}

			// Scale
			{
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				// Cube arrow profile
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

				const real_t arrow_sides_step = Math_TAU / arrow_sides;
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

			// Plane Scale
			{
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				Vector3 vec = ivec2 - ivec3;
				Vector3 plane[4] = {
					vec * GIZMO_PLANE_DST,
					vec * GIZMO_PLANE_DST + ivec2 * GIZMO_PLANE_SIZE,
					vec * (GIZMO_PLANE_DST + GIZMO_PLANE_SIZE),
					vec * GIZMO_PLANE_DST - ivec3 * GIZMO_PLANE_SIZE
				};

				Basis ma(ivec, Math_PI / 2);

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

				Ref<StandardMaterial3D> plane_mat = memnew(StandardMaterial3D);
				plane_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
				plane_mat->set_on_top_of_alpha();
				plane_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
				plane_mat->set_cull_mode(StandardMaterial3D::CULL_DISABLED);
				plane_mat->set_albedo(col);
				plane_gizmo_color[i] = plane_mat; // needed, so we can draw planes from both sides
				surftool->set_material(plane_mat);
				surftool->commit(scale_plane_gizmo[i]);

				Ref<StandardMaterial3D> plane_mat_hl = plane_mat->duplicate();
				plane_mat_hl->set_albedo(col.from_hsv(col.get_h(), 0.25, 1.0, 1));
				plane_gizmo_color_hl[i] = plane_mat_hl; // needed, so we can draw planes from both sides
			}

			// Lines to visualize transforms locked to an axis/plane
			{
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_LINE_STRIP);

				Vector3 vec;
				vec[i] = 1;

				// line extending through infinity(ish)
				surftool->add_vertex(vec * -1048576);
				surftool->add_vertex(Vector3());
				surftool->add_vertex(vec * 1048576);
				surftool->set_material(mat_hl);
				surftool->commit(axis_gizmo[i]);
			}
		}
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
				gizmos_menu->set_item_icon(idx, gizmos_menu->get_theme_icon(SNAME("visibility_visible")));
				break;
			case EditorNode3DGizmoPlugin::ON_TOP:
				gizmos_menu->set_item_icon(idx, gizmos_menu->get_theme_icon(SNAME("visibility_xray")));
				break;
			case EditorNode3DGizmoPlugin::HIDDEN:
				gizmos_menu->set_item_icon(idx, gizmos_menu->get_theme_icon(SNAME("visibility_hidden")));
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
				gizmos_menu->set_item_icon(idx, gizmos_menu->get_theme_icon(SNAME("visibility_visible")));
				break;
			case EditorNode3DGizmoPlugin::ON_TOP:
				gizmos_menu->set_item_icon(idx, gizmos_menu->get_theme_icon(SNAME("visibility_xray")));
				break;
			case EditorNode3DGizmoPlugin::HIDDEN:
				gizmos_menu->set_item_icon(idx, gizmos_menu->get_theme_icon(SNAME("visibility_hidden")));
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

	Vector<Color> grid_colors[3];
	Vector<Vector3> grid_points[3];
	Vector<Vector3> grid_normals[3];

	Color primary_grid_color = EDITOR_GET("editors/3d/primary_grid_color");
	Color secondary_grid_color = EDITOR_GET("editors/3d/secondary_grid_color");
	int grid_size = EDITOR_GET("editors/3d/grid_size");
	int primary_grid_steps = EDITOR_GET("editors/3d/primary_grid_steps");

	// Which grid planes are enabled? Which should we generate?
	grid_enable[0] = grid_visible[0] = EDITOR_GET("editors/3d/grid_xy_plane");
	grid_enable[1] = grid_visible[1] = EDITOR_GET("editors/3d/grid_yz_plane");
	grid_enable[2] = grid_visible[2] = EDITOR_GET("editors/3d/grid_xz_plane");

	// Offsets division_level for bigger or smaller grids.
	// Default value is -0.2. -1.0 gives Blender-like behavior, 0.5 gives huge grids.
	real_t division_level_bias = EDITOR_GET("editors/3d/grid_division_level_bias");
	// Default largest grid size is 8^2 when primary_grid_steps is 8 (64m apart, so primary grid lines are 512m apart).
	int division_level_max = EDITOR_GET("editors/3d/grid_division_level_max");
	// Default smallest grid size is 1cm, 10^-2 (default value is -2).
	int division_level_min = EDITOR_GET("editors/3d/grid_division_level_min");
	ERR_FAIL_COND_MSG(division_level_max < division_level_min, "The 3D grid's maximum division level cannot be lower than its minimum division level.");

	if (primary_grid_steps != 10) { // Log10 of 10 is 1.
		// Change of base rule, divide by ln(10).
		real_t div = Math::log((real_t)primary_grid_steps) / (real_t)2.302585092994045901094;
		// Trucation (towards zero) is intentional.
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

		// Cache these so we don't have to re-access memory.
		Vector<Vector3> &ref_grid = grid_points[c];
		Vector<Vector3> &ref_grid_normals = grid_normals[c];
		Vector<Color> &ref_grid_colors = grid_colors[c];

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
				ref_grid.set(idx, line_bgn);
				ref_grid.set(idx + 1, line_end);
				ref_grid_colors.set(idx, line_color);
				ref_grid_colors.set(idx + 1, line_color);
				ref_grid_normals.set(idx, normal);
				ref_grid_normals.set(idx + 1, normal);
				idx += 2;
			}

			if (!(origin_enabled && Math::is_zero_approx(position_b))) {
				Vector3 line_bgn;
				Vector3 line_end;
				line_bgn[b] = position_b;
				line_end[b] = position_b;
				line_bgn[a] = bgn_a;
				line_end[a] = end_a;
				ref_grid.set(idx, line_bgn);
				ref_grid.set(idx + 1, line_end);
				ref_grid_colors.set(idx, line_color);
				ref_grid_colors.set(idx + 1, line_color);
				ref_grid_normals.set(idx, normal);
				ref_grid_normals.set(idx + 1, normal);
				idx += 2;
			}
		}

		// Create a mesh from the pushed vector points and colors.
		grid[c] = RenderingServer::get_singleton()->mesh_create();
		Array d;
		d.resize(RS::ARRAY_MAX);
		d[RenderingServer::ARRAY_VERTEX] = grid_points[c];
		d[RenderingServer::ARRAY_COLOR] = grid_colors[c];
		d[RenderingServer::ARRAY_NORMAL] = grid_normals[c];
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(grid[c], RenderingServer::PRIMITIVE_LINES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(grid[c], 0, grid_mat[c]->get_rid());
		grid_instance[c] = RenderingServer::get_singleton()->instance_create2(grid[c], get_tree()->get_root()->get_world_3d()->get_scenario());

		// Yes, the end of this line is supposed to be a.
		RenderingServer::get_singleton()->instance_set_visible(grid_instance[c], grid_visible[a]);
		RenderingServer::get_singleton()->instance_geometry_set_cast_shadows_setting(grid_instance[c], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(grid_instance[c], 1 << Node3DEditorViewport::GIZMO_GRID_LAYER);
		RS::get_singleton()->instance_geometry_set_flag(grid_instance[c], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(grid_instance[c], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	}
}

void Node3DEditor::_finish_indicators() {
	RenderingServer::get_singleton()->free(origin_instance);
	RenderingServer::get_singleton()->free(origin);

	_finish_grid();
}

void Node3DEditor::_finish_grid() {
	for (int i = 0; i < 3; i++) {
		RenderingServer::get_singleton()->free(grid_instance[i]);
		RenderingServer::get_singleton()->free(grid[i]);
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
	if (selected && editor_selection->get_selected_node_list().size() != 1) {
		Vector<Ref<Node3DGizmo>> gizmos = selected->get_gizmos();
		for (int i = 0; i < gizmos.size(); i++) {
			Ref<EditorNode3DGizmo> seg = gizmos[i];
			if (!seg.is_valid()) {
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
	update_transform_gizmo();
}

void Node3DEditor::_refresh_menu_icons() {
	bool all_locked = true;
	bool all_grouped = true;

	List<Node *> &selection = editor_selection->get_selected_node_list();

	if (selection.is_empty()) {
		all_locked = false;
		all_grouped = false;
	} else {
		for (Node *E : selection) {
			if (Object::cast_to<Node3D>(E) && !Object::cast_to<Node3D>(E)->has_meta("_edit_lock_")) {
				all_locked = false;
				break;
			}
		}
		for (Node *E : selection) {
			if (Object::cast_to<Node3D>(E) && !Object::cast_to<Node3D>(E)->has_meta("_edit_group_")) {
				all_grouped = false;
				break;
			}
		}
	}

	tool_button[TOOL_LOCK_SELECTED]->set_visible(!all_locked);
	tool_button[TOOL_LOCK_SELECTED]->set_disabled(selection.is_empty());
	tool_button[TOOL_UNLOCK_SELECTED]->set_visible(all_locked);

	tool_button[TOOL_GROUP_SELECTED]->set_visible(!all_grouped);
	tool_button[TOOL_GROUP_SELECTED]->set_disabled(selection.is_empty());
	tool_button[TOOL_UNGROUP_SELECTED]->set_visible(all_grouped);
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
	const List<Node *> &selection = editor_selection->get_selected_node_list();
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
	PhysicsDirectSpaceState3D::RayResult result;

	Array keys = snap_data.keys();

	// The maximum height an object can travel to be snapped
	const float max_snap_height = 500.0;

	// Will be set to `true` if at least one node from the selection was successfully snapped
	bool snapped_to_floor = false;

	if (keys.size()) {
		// For snapping to be performed, there must be solid geometry under at least one of the selected nodes.
		// We need to check this before snapping to register the undo/redo action only if needed.
		for (int i = 0; i < keys.size(); i++) {
			Node *node = Object::cast_to<Node>(keys[i]);
			Node3D *sp = Object::cast_to<Node3D>(node);
			Dictionary d = snap_data[node];
			Vector3 from = d["from"];
			Vector3 to = from - Vector3(0.0, max_snap_height, 0.0);
			HashSet<RID> excluded = _get_physics_bodies_rid(sp);

			PhysicsDirectSpaceState3D::RayParameters ray_params;
			ray_params.from = from;
			ray_params.to = to;
			ray_params.exclude = excluded;

			if (ss->intersect_ray(ray_params, result)) {
				snapped_to_floor = true;
			}
		}

		if (snapped_to_floor) {
			Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
			undo_redo->create_action(TTR("Snap Nodes to Floor"));

			// Perform snapping if at least one node can be snapped
			for (int i = 0; i < keys.size(); i++) {
				Node *node = Object::cast_to<Node>(keys[i]);
				Node3D *sp = Object::cast_to<Node3D>(node);
				Dictionary d = snap_data[node];
				Vector3 from = d["from"];
				Vector3 to = from - Vector3(0.0, max_snap_height, 0.0);
				HashSet<RID> excluded = _get_physics_bodies_rid(sp);

				PhysicsDirectSpaceState3D::RayParameters ray_params;
				ray_params.from = from;
				ray_params.to = to;
				ray_params.exclude = excluded;

				if (ss->intersect_ray(ray_params, result)) {
					Vector3 position_offset = d["position_offset"];
					Transform3D new_transform = sp->get_global_transform();

					new_transform.origin.y = result.position.y;
					new_transform.origin = new_transform.origin - position_offset;

					undo_redo->add_do_method(sp, "set_global_transform", new_transform);
					undo_redo->add_undo_method(sp, "set_global_transform", sp->get_global_transform());
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

	snap_key_enabled = Input::get_singleton()->is_key_pressed(Key::CTRL);
}

void Node3DEditor::_sun_environ_settings_pressed() {
	Vector2 pos = sun_environ_settings->get_screen_position() + sun_environ_settings->get_size();
	sun_environ_popup->set_position(pos - Vector2(sun_environ_popup->get_contents_minimum_size().width / 2, 0));
	sun_environ_popup->reset_size();
	sun_environ_popup->popup();
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
	ERR_FAIL_COND(!base);
	Node *new_sun = preview_sun->duplicate();

	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
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
	ERR_FAIL_COND(!base);

	WorldEnvironment *new_env = memnew(WorldEnvironment);
	new_env->set_environment(preview_environment->get_environment()->duplicate(true));
	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		new_env->set_camera_attributes(preview_environment->get_camera_attributes()->duplicate(true));
	}

	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
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
	tool_button[TOOL_MODE_SELECT]->set_icon(get_theme_icon(SNAME("ToolSelect"), SNAME("EditorIcons")));
	tool_button[TOOL_MODE_MOVE]->set_icon(get_theme_icon(SNAME("ToolMove"), SNAME("EditorIcons")));
	tool_button[TOOL_MODE_ROTATE]->set_icon(get_theme_icon(SNAME("ToolRotate"), SNAME("EditorIcons")));
	tool_button[TOOL_MODE_SCALE]->set_icon(get_theme_icon(SNAME("ToolScale"), SNAME("EditorIcons")));
	tool_button[TOOL_MODE_LIST_SELECT]->set_icon(get_theme_icon(SNAME("ListSelect"), SNAME("EditorIcons")));
	tool_button[TOOL_LOCK_SELECTED]->set_icon(get_theme_icon(SNAME("Lock"), SNAME("EditorIcons")));
	tool_button[TOOL_UNLOCK_SELECTED]->set_icon(get_theme_icon(SNAME("Unlock"), SNAME("EditorIcons")));
	tool_button[TOOL_GROUP_SELECTED]->set_icon(get_theme_icon(SNAME("Group"), SNAME("EditorIcons")));
	tool_button[TOOL_UNGROUP_SELECTED]->set_icon(get_theme_icon(SNAME("Ungroup"), SNAME("EditorIcons")));

	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_icon(get_theme_icon(SNAME("Object"), SNAME("EditorIcons")));
	tool_option_button[TOOL_OPT_USE_SNAP]->set_icon(get_theme_icon(SNAME("Snap"), SNAME("EditorIcons")));
	tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->set_icon(get_theme_icon(SNAME("Camera3D"), SNAME("EditorIcons")));

	view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), get_theme_icon(SNAME("Panels1"), SNAME("EditorIcons")));
	view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), get_theme_icon(SNAME("Panels2"), SNAME("EditorIcons")));
	view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), get_theme_icon(SNAME("Panels2Alt"), SNAME("EditorIcons")));
	view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), get_theme_icon(SNAME("Panels3"), SNAME("EditorIcons")));
	view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), get_theme_icon(SNAME("Panels3Alt"), SNAME("EditorIcons")));
	view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), get_theme_icon(SNAME("Panels4"), SNAME("EditorIcons")));

	sun_button->set_icon(get_theme_icon(SNAME("PreviewSun"), SNAME("EditorIcons")));
	environ_button->set_icon(get_theme_icon(SNAME("PreviewEnvironment"), SNAME("EditorIcons")));
	sun_environ_settings->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));

	sun_title->add_theme_font_override("font", get_theme_font(SNAME("title_font"), SNAME("Window")));
	environ_title->add_theme_font_override("font", get_theme_font(SNAME("title_font"), SNAME("Window")));

	sun_color->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("color_picker_button_height"), SNAME("Editor"))));
	environ_sky_color->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("color_picker_button_height"), SNAME("Editor"))));
	environ_ground_color->set_custom_minimum_size(Size2(0, get_theme_constant(SNAME("color_picker_button_height"), SNAME("Editor"))));

	context_menu_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("ContextualToolbar"), SNAME("EditorStyles")));
}

void Node3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);

			_refresh_menu_icons();

			get_tree()->connect("node_removed", callable_mp(this, &Node3DEditor::_node_removed));
			get_tree()->connect("node_added", callable_mp(this, &Node3DEditor::_node_added));
			SceneTreeDock::get_singleton()->get_tree_editor()->connect("node_changed", callable_mp(this, &Node3DEditor::_refresh_menu_icons));
			editor_selection->connect("selection_changed", callable_mp(this, &Node3DEditor::_selection_changed));

			EditorNode::get_singleton()->connect("stop_pressed", callable_mp(this, &Node3DEditor::_update_camera_override_button).bind(false));
			EditorNode::get_singleton()->connect("play_pressed", callable_mp(this, &Node3DEditor::_update_camera_override_button).bind(true));

			_update_preview_environment();

			sun_state->set_custom_minimum_size(sun_vb->get_combined_minimum_size());
			environ_state->set_custom_minimum_size(environ_vb->get_combined_minimum_size());

			EditorNode::get_singleton()->connect("project_settings_changed", callable_mp(this, &Node3DEditor::update_all_gizmos).bind(Variant()));
		} break;

		case NOTIFICATION_ENTER_TREE: {
			_update_theme();
			_register_all_gizmos();
			_update_gizmos_menu();
			_init_indicators();
			update_all_gizmos();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_finish_indicators();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
			_update_gizmos_menu_theme();
			sun_title->add_theme_font_override("font", get_theme_font(SNAME("title_font"), SNAME("Window")));
			environ_title->add_theme_font_override("font", get_theme_font(SNAME("title_font"), SNAME("Window")));
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			// Update grid color by rebuilding grid.
			_finish_grid();
			_init_grid();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible() && tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->is_pressed()) {
				EditorDebuggerNode *debugger = EditorDebuggerNode::get_singleton();

				debugger->set_camera_override(EditorDebuggerNode::OVERRIDE_NONE);
				tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->set_pressed(false);
			}
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

void Node3DEditor::add_control_to_menu_panel(Control *p_control) {
	context_menu_hbox->add_child(p_control);
}

void Node3DEditor::remove_control_from_menu_panel(Control *p_control) {
	context_menu_hbox->remove_child(p_control);
}

void Node3DEditor::set_can_preview(Camera3D *p_preview) {
	for (int i = 0; i < 4; i++) {
		viewports[i]->set_can_preview(p_preview);
	}
}

VSplitContainer *Node3DEditor::get_shader_split() {
	return shader_split;
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
	Node3D *node = Object::cast_to<Node3D>(ObjectDB::get_instance(p_id));
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

		if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT))) {
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);
		} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS))) {
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS);
		} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT))) {
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS_ALT);
		} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS))) {
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS);
		} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT))) {
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS_ALT);
		} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS))) {
			_menu_item_pressed(MENU_VIEW_USE_4_VIEWPORTS);
		}
	}
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
	add_gizmo_plugin(Ref<SoftBody3DGizmoPlugin>(memnew(SoftBody3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Sprite3DGizmoPlugin>(memnew(Sprite3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Label3DGizmoPlugin>(memnew(Label3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Marker3DGizmoPlugin>(memnew(Marker3DGizmoPlugin)));
	add_gizmo_plugin(Ref<RayCast3DGizmoPlugin>(memnew(RayCast3DGizmoPlugin)));
	add_gizmo_plugin(Ref<ShapeCast3DGizmoPlugin>(memnew(ShapeCast3DGizmoPlugin)));
	add_gizmo_plugin(Ref<SpringArm3DGizmoPlugin>(memnew(SpringArm3DGizmoPlugin)));
	add_gizmo_plugin(Ref<VehicleWheel3DGizmoPlugin>(memnew(VehicleWheel3DGizmoPlugin)));
	add_gizmo_plugin(Ref<VisibleOnScreenNotifier3DGizmoPlugin>(memnew(VisibleOnScreenNotifier3DGizmoPlugin)));
	add_gizmo_plugin(Ref<GPUParticles3DGizmoPlugin>(memnew(GPUParticles3DGizmoPlugin)));
	add_gizmo_plugin(Ref<GPUParticlesCollision3DGizmoPlugin>(memnew(GPUParticlesCollision3DGizmoPlugin)));
	add_gizmo_plugin(Ref<CPUParticles3DGizmoPlugin>(memnew(CPUParticles3DGizmoPlugin)));
	add_gizmo_plugin(Ref<ReflectionProbeGizmoPlugin>(memnew(ReflectionProbeGizmoPlugin)));
	add_gizmo_plugin(Ref<DecalGizmoPlugin>(memnew(DecalGizmoPlugin)));
	add_gizmo_plugin(Ref<VoxelGIGizmoPlugin>(memnew(VoxelGIGizmoPlugin)));
	add_gizmo_plugin(Ref<LightmapGIGizmoPlugin>(memnew(LightmapGIGizmoPlugin)));
	add_gizmo_plugin(Ref<LightmapProbeGizmoPlugin>(memnew(LightmapProbeGizmoPlugin)));
	add_gizmo_plugin(Ref<CollisionObject3DGizmoPlugin>(memnew(CollisionObject3DGizmoPlugin)));
	add_gizmo_plugin(Ref<CollisionShape3DGizmoPlugin>(memnew(CollisionShape3DGizmoPlugin)));
	add_gizmo_plugin(Ref<CollisionPolygon3DGizmoPlugin>(memnew(CollisionPolygon3DGizmoPlugin)));
	add_gizmo_plugin(Ref<NavigationLink3DGizmoPlugin>(memnew(NavigationLink3DGizmoPlugin)));
	add_gizmo_plugin(Ref<NavigationRegion3DGizmoPlugin>(memnew(NavigationRegion3DGizmoPlugin)));
	add_gizmo_plugin(Ref<Joint3DGizmoPlugin>(memnew(Joint3DGizmoPlugin)));
	add_gizmo_plugin(Ref<PhysicalBone3DGizmoPlugin>(memnew(PhysicalBone3DGizmoPlugin)));
	add_gizmo_plugin(Ref<FogVolumeGizmoPlugin>(memnew(FogVolumeGizmoPlugin)));
}

void Node3DEditor::_bind_methods() {
	ClassDB::bind_method("_get_editor_data", &Node3DEditor::_get_editor_data);
	ClassDB::bind_method("_request_gizmo", &Node3DEditor::_request_gizmo);
	ClassDB::bind_method("_request_gizmo_for_id", &Node3DEditor::_request_gizmo_for_id);
	ClassDB::bind_method("_set_subgizmo_selection", &Node3DEditor::_set_subgizmo_selection);
	ClassDB::bind_method("_clear_subgizmo_selection", &Node3DEditor::_clear_subgizmo_selection);
	ClassDB::bind_method("_refresh_menu_icons", &Node3DEditor::_refresh_menu_icons);

	ADD_SIGNAL(MethodInfo("transform_key_request"));
	ADD_SIGNAL(MethodInfo("item_lock_status_changed"));
	ADD_SIGNAL(MethodInfo("item_group_status_changed"));
}

void Node3DEditor::clear() {
	settings_fov->set_value(EDITOR_GET("editors/3d/default_fov"));
	settings_znear->set_value(EDITOR_GET("editors/3d/default_z_near"));
	settings_zfar->set_value(EDITOR_GET("editors/3d/default_z_far"));

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->reset();
	}

	RenderingServer::get_singleton()->instance_set_visible(origin_instance, true);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN), true);
	for (int i = 0; i < 3; ++i) {
		if (grid_enable[i]) {
			grid_visible[i] = true;
		}
	}

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(Node3DEditorViewport::VIEW_AUDIO_LISTENER), i == 0);
		viewports[i]->viewport->set_as_audio_listener_3d(i == 0);
	}

	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_GRID), true);
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
		Transform3D t;
		t.basis = Basis::from_euler(Vector3(sun_rotation.x, sun_rotation.y, 0));
		preview_sun->set_transform(t);
		sun_direction->queue_redraw();
		preview_sun->set_param(Light3D::PARAM_ENERGY, sun_energy->get_value());
		preview_sun->set_param(Light3D::PARAM_SHADOW_MAX_DISTANCE, sun_max_distance->get_value());
		preview_sun->set_color(sun_color->get_pick_color());
	}

	{ //preview env
		sky_material->set_sky_energy_multiplier(environ_energy->get_value());
		Color hz_color = environ_sky_color->get_pick_color().lerp(environ_ground_color->get_pick_color(), 0.5).lerp(Color(1, 1, 1), 0.5);
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

	sun_angle_altitude->set_value(-Math::rad_to_deg(sun_rotation.x));
	sun_angle_azimuth->set_value(180.0 - Math::rad_to_deg(sun_rotation.y));
	sun_direction->queue_redraw();
	environ_sky_color->set_pick_color(Color(0.385, 0.454, 0.55));
	environ_ground_color->set_pick_color(Color(0.2, 0.169, 0.133));
	environ_energy->set_value(1.0);
	environ_glow_button->set_pressed(true);
	environ_tonemap_button->set_pressed(true);
	environ_ao_button->set_pressed(false);
	environ_gi_button->set_pressed(false);
	sun_max_distance->set_value(100);

	sun_color->set_pick_color(Color(1, 1, 1));
	sun_energy->set_value(1.0);

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
			sun_state->set_text(TTR("Scene contains\nDirectionalLight3D.\nPreview disabled."));
		} else {
			sun_state->set_text(TTR("Preview disabled."));
		}

	} else {
		if (!preview_sun->get_parent()) {
			add_child(preview_sun, true);
			sun_state->hide();
			sun_vb->show();
			preview_sun_dangling = false;
		}
	}

	sun_angle_altitude->set_value(-Math::rad_to_deg(sun_rotation.x));
	sun_angle_azimuth->set_value(180.0 - Math::rad_to_deg(sun_rotation.y));

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
			environ_state->set_text(TTR("Scene contains\nWorldEnvironment.\nPreview disabled."));
		} else {
			environ_state->set_text(TTR("Preview disabled."));
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
	if (mm.is_valid() && (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
		sun_rotation.x += mm->get_relative().y * (0.02 * EDSCALE);
		sun_rotation.y -= mm->get_relative().x * (0.02 * EDSCALE);
		sun_rotation.x = CLAMP(sun_rotation.x, -Math_TAU / 4, Math_TAU / 4);
		sun_angle_altitude->set_value(-Math::rad_to_deg(sun_rotation.x));
		sun_angle_azimuth->set_value(180.0 - Math::rad_to_deg(sun_rotation.y));
		_preview_settings_changed();
	}
}

void Node3DEditor::_sun_direction_angle_set() {
	sun_rotation.x = Math::deg_to_rad(-sun_angle_altitude->get_value());
	sun_rotation.y = Math::deg_to_rad(180.0 - sun_angle_azimuth->get_value());
	_preview_settings_changed();
}

Node3DEditor::Node3DEditor() {
	gizmo.visible = true;
	gizmo.scale = 1.0;

	viewport_environment = Ref<Environment>(memnew(Environment));
	VBoxContainer *vbc = this;

	custom_camera = nullptr;
	singleton = this;
	editor_selection = EditorNode::get_singleton()->get_editor_selection();
	editor_selection->add_editor_plugin(this);

	snap_enabled = false;
	snap_key_enabled = false;
	tool_mode = TOOL_MODE_SELECT;

	camera_override_viewport_id = 0;

	// A fluid container for all toolbars.
	HFlowContainer *main_flow = memnew(HFlowContainer);
	vbc->add_child(main_flow);

	// Main toolbars.
	HBoxContainer *main_menu_hbox = memnew(HBoxContainer);
	main_flow->add_child(main_menu_hbox);

	String sct;

	// Add some margin to the left for better aesthetics.
	// This prevents the first button's hover/pressed effect from "touching" the panel's border,
	// which looks ugly.
	Control *margin_left = memnew(Control);
	main_menu_hbox->add_child(margin_left);
	margin_left->set_custom_minimum_size(Size2(2, 0) * EDSCALE);

	tool_button[TOOL_MODE_SELECT] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_SELECT]);
	tool_button[TOOL_MODE_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_SELECT]->set_flat(true);
	tool_button[TOOL_MODE_SELECT]->set_pressed(true);
	tool_button[TOOL_MODE_SELECT]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_SELECT));
	tool_button[TOOL_MODE_SELECT]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_select", TTR("Select Mode"), Key::Q));
	tool_button[TOOL_MODE_SELECT]->set_shortcut_context(this);
	tool_button[TOOL_MODE_SELECT]->set_tooltip_text(keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL) + TTR("Drag: Rotate selected node around pivot.") + "\n" + TTR("Alt+RMB: Show list of all nodes at position clicked, including locked."));
	main_menu_hbox->add_child(memnew(VSeparator));

	tool_button[TOOL_MODE_MOVE] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_MOVE]);
	tool_button[TOOL_MODE_MOVE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_MOVE]->set_flat(true);

	tool_button[TOOL_MODE_MOVE]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_MOVE));
	tool_button[TOOL_MODE_MOVE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_move", TTR("Move Mode"), Key::W));
	tool_button[TOOL_MODE_MOVE]->set_shortcut_context(this);

	tool_button[TOOL_MODE_ROTATE] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_ROTATE]);
	tool_button[TOOL_MODE_ROTATE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_ROTATE]->set_flat(true);
	tool_button[TOOL_MODE_ROTATE]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_ROTATE));
	tool_button[TOOL_MODE_ROTATE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_rotate", TTR("Rotate Mode"), Key::E));
	tool_button[TOOL_MODE_ROTATE]->set_shortcut_context(this);

	tool_button[TOOL_MODE_SCALE] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_SCALE]);
	tool_button[TOOL_MODE_SCALE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_SCALE]->set_flat(true);
	tool_button[TOOL_MODE_SCALE]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_SCALE));
	tool_button[TOOL_MODE_SCALE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_scale", TTR("Scale Mode"), Key::R));
	tool_button[TOOL_MODE_SCALE]->set_shortcut_context(this);

	main_menu_hbox->add_child(memnew(VSeparator));

	tool_button[TOOL_MODE_LIST_SELECT] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_MODE_LIST_SELECT]);
	tool_button[TOOL_MODE_LIST_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_LIST_SELECT]->set_flat(true);
	tool_button[TOOL_MODE_LIST_SELECT]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_TOOL_LIST_SELECT));
	tool_button[TOOL_MODE_LIST_SELECT]->set_tooltip_text(TTR("Show list of selectable nodes at position clicked."));

	tool_button[TOOL_LOCK_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_LOCK_SELECTED]);
	tool_button[TOOL_LOCK_SELECTED]->set_flat(true);
	tool_button[TOOL_LOCK_SELECTED]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_LOCK_SELECTED));
	tool_button[TOOL_LOCK_SELECTED]->set_tooltip_text(TTR("Lock selected node, preventing selection and movement."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_LOCK_SELECTED]->set_shortcut(ED_SHORTCUT("editor/lock_selected_nodes", TTR("Lock Selected Node(s)"), KeyModifierMask::CMD_OR_CTRL | Key::L));

	tool_button[TOOL_UNLOCK_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_UNLOCK_SELECTED]);
	tool_button[TOOL_UNLOCK_SELECTED]->set_flat(true);
	tool_button[TOOL_UNLOCK_SELECTED]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_UNLOCK_SELECTED));
	tool_button[TOOL_UNLOCK_SELECTED]->set_tooltip_text(TTR("Unlock selected node, allowing selection and movement."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_UNLOCK_SELECTED]->set_shortcut(ED_SHORTCUT("editor/unlock_selected_nodes", TTR("Unlock Selected Node(s)"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::L));

	tool_button[TOOL_GROUP_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_GROUP_SELECTED]);
	tool_button[TOOL_GROUP_SELECTED]->set_flat(true);
	tool_button[TOOL_GROUP_SELECTED]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_GROUP_SELECTED));
	tool_button[TOOL_GROUP_SELECTED]->set_tooltip_text(TTR("Make selected node's children not selectable."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_GROUP_SELECTED]->set_shortcut(ED_SHORTCUT("editor/group_selected_nodes", TTR("Group Selected Node(s)"), KeyModifierMask::CMD_OR_CTRL | Key::G));

	tool_button[TOOL_UNGROUP_SELECTED] = memnew(Button);
	main_menu_hbox->add_child(tool_button[TOOL_UNGROUP_SELECTED]);
	tool_button[TOOL_UNGROUP_SELECTED]->set_flat(true);
	tool_button[TOOL_UNGROUP_SELECTED]->connect("pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed).bind(MENU_UNGROUP_SELECTED));
	tool_button[TOOL_UNGROUP_SELECTED]->set_tooltip_text(TTR("Make selected node's children selectable."));
	// Define the shortcut globally (without a context) so that it works if the Scene tree dock is currently focused.
	tool_button[TOOL_UNGROUP_SELECTED]->set_shortcut(ED_SHORTCUT("editor/ungroup_selected_nodes", TTR("Ungroup Selected Node(s)"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::G));

	main_menu_hbox->add_child(memnew(VSeparator));

	tool_option_button[TOOL_OPT_LOCAL_COORDS] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_LOCAL_COORDS]);
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_flat(true);
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->connect("toggled", callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_LOCAL_COORDS));
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_shortcut(ED_SHORTCUT("spatial_editor/local_coords", TTR("Use Local Space"), Key::T));
	tool_option_button[TOOL_OPT_LOCAL_COORDS]->set_shortcut_context(this);

	tool_option_button[TOOL_OPT_USE_SNAP] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_USE_SNAP]);
	tool_option_button[TOOL_OPT_USE_SNAP]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_USE_SNAP]->set_flat(true);
	tool_option_button[TOOL_OPT_USE_SNAP]->connect("toggled", callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_USE_SNAP));
	tool_option_button[TOOL_OPT_USE_SNAP]->set_shortcut(ED_SHORTCUT("spatial_editor/snap", TTR("Use Snap"), Key::Y));
	tool_option_button[TOOL_OPT_USE_SNAP]->set_shortcut_context(this);

	main_menu_hbox->add_child(memnew(VSeparator));

	tool_option_button[TOOL_OPT_OVERRIDE_CAMERA] = memnew(Button);
	main_menu_hbox->add_child(tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]);
	tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->set_toggle_mode(true);
	tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->set_flat(true);
	tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->set_disabled(true);
	tool_option_button[TOOL_OPT_OVERRIDE_CAMERA]->connect("toggled", callable_mp(this, &Node3DEditor::_menu_item_toggled).bind(MENU_TOOL_OVERRIDE_CAMERA));
	_update_camera_override_button(false);

	main_menu_hbox->add_child(memnew(VSeparator));
	sun_button = memnew(Button);
	sun_button->set_tooltip_text(TTR("Toggle preview sunlight.\nIf a DirectionalLight3D node is added to the scene, preview sunlight is disabled."));
	sun_button->set_toggle_mode(true);
	sun_button->set_flat(true);
	sun_button->connect("pressed", callable_mp(this, &Node3DEditor::_update_preview_environment), CONNECT_DEFERRED);
	// Preview is enabled by default - ensure this applies on editor startup when there is no state yet.
	sun_button->set_pressed(true);

	main_menu_hbox->add_child(sun_button);

	environ_button = memnew(Button);
	environ_button->set_tooltip_text(TTR("Toggle preview environment.\nIf a WorldEnvironment node is added to the scene, preview environment is disabled."));
	environ_button->set_toggle_mode(true);
	environ_button->set_flat(true);
	environ_button->connect("pressed", callable_mp(this, &Node3DEditor::_update_preview_environment), CONNECT_DEFERRED);
	// Preview is enabled by default - ensure this applies on editor startup when there is no state yet.
	environ_button->set_pressed(true);

	main_menu_hbox->add_child(environ_button);

	sun_environ_settings = memnew(Button);
	sun_environ_settings->set_tooltip_text(TTR("Edit Sun and Environment settings."));
	sun_environ_settings->set_flat(true);
	sun_environ_settings->connect("pressed", callable_mp(this, &Node3DEditor::_sun_environ_settings_pressed));

	main_menu_hbox->add_child(sun_environ_settings);

	main_menu_hbox->add_child(memnew(VSeparator));

	// Drag and drop support;
	preview_node = memnew(Node3D);
	preview_bounds = AABB();

	ED_SHORTCUT("spatial_editor/bottom_view", TTR("Bottom View"), KeyModifierMask::ALT + Key::KP_7);
	ED_SHORTCUT("spatial_editor/top_view", TTR("Top View"), Key::KP_7);
	ED_SHORTCUT("spatial_editor/rear_view", TTR("Rear View"), KeyModifierMask::ALT + Key::KP_1);
	ED_SHORTCUT("spatial_editor/front_view", TTR("Front View"), Key::KP_1);
	ED_SHORTCUT("spatial_editor/left_view", TTR("Left View"), KeyModifierMask::ALT + Key::KP_3);
	ED_SHORTCUT("spatial_editor/right_view", TTR("Right View"), Key::KP_3);
	ED_SHORTCUT("spatial_editor/orbit_view_down", TTR("Orbit View Down"), Key::KP_2);
	ED_SHORTCUT("spatial_editor/orbit_view_left", TTR("Orbit View Left"), Key::KP_4);
	ED_SHORTCUT("spatial_editor/orbit_view_right", TTR("Orbit View Right"), Key::KP_6);
	ED_SHORTCUT("spatial_editor/orbit_view_up", TTR("Orbit View Up"), Key::KP_8);
	ED_SHORTCUT("spatial_editor/orbit_view_180", TTR("Orbit View 180"), Key::KP_9);
	ED_SHORTCUT("spatial_editor/switch_perspective_orthogonal", TTR("Switch Perspective/Orthogonal View"), Key::KP_5);
	ED_SHORTCUT("spatial_editor/insert_anim_key", TTR("Insert Animation Key"), Key::K);
	ED_SHORTCUT("spatial_editor/focus_origin", TTR("Focus Origin"), Key::O);
	ED_SHORTCUT("spatial_editor/focus_selection", TTR("Focus Selection"), Key::F);
	ED_SHORTCUT("spatial_editor/align_transform_with_view", TTR("Align Transform with View"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::M);
	ED_SHORTCUT("spatial_editor/align_rotation_with_view", TTR("Align Rotation with View"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::F);
	ED_SHORTCUT("spatial_editor/freelook_toggle", TTR("Toggle Freelook"), KeyModifierMask::SHIFT + Key::F);
	ED_SHORTCUT("spatial_editor/decrease_fov", TTR("Decrease Field of View"), KeyModifierMask::CMD_OR_CTRL + Key::EQUAL); // Usually direct access key for `KEY_PLUS`.
	ED_SHORTCUT("spatial_editor/increase_fov", TTR("Increase Field of View"), KeyModifierMask::CMD_OR_CTRL + Key::MINUS);
	ED_SHORTCUT("spatial_editor/reset_fov", TTR("Reset Field of View to Default"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_0);

	PopupMenu *p;

	transform_menu = memnew(MenuButton);
	transform_menu->set_text(TTR("Transform"));
	transform_menu->set_switch_on_hover(true);
	transform_menu->set_shortcut_context(this);
	main_menu_hbox->add_child(transform_menu);

	p = transform_menu->get_popup();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/snap_to_floor", TTR("Snap Object to Floor"), Key::PAGEDOWN), MENU_SNAP_TO_FLOOR);
	p->add_shortcut(ED_SHORTCUT("spatial_editor/transform_dialog", TTR("Transform Dialog...")), MENU_TRANSFORM_DIALOG);

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/configure_snap", TTR("Configure Snap...")), MENU_TRANSFORM_CONFIGURE_SNAP);

	p->connect("id_pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed));

	view_menu = memnew(MenuButton);
	// TRANSLATORS: Noun, name of the 2D/3D View menus.
	view_menu->set_text(TTR("View"));
	view_menu->set_switch_on_hover(true);
	view_menu->set_shortcut_context(this);
	main_menu_hbox->add_child(view_menu);

	main_menu_hbox->add_child(memnew(VSeparator));

	context_menu_panel = memnew(PanelContainer);
	context_menu_hbox = memnew(HBoxContainer);
	context_menu_panel->add_child(context_menu_hbox);
	main_flow->add_child(context_menu_panel);

	// Get the view menu popup and have it stay open when a checkable item is selected
	p = view_menu->get_popup();
	p->set_hide_on_checkable_item_selection(false);

	accept = memnew(AcceptDialog);
	EditorNode::get_singleton()->get_gui_base()->add_child(accept);

	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/1_viewport", TTR("1 Viewport"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_1), MENU_VIEW_USE_1_VIEWPORT);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/2_viewports", TTR("2 Viewports"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_2), MENU_VIEW_USE_2_VIEWPORTS);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/2_viewports_alt", TTR("2 Viewports (Alt)"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::KEY_2), MENU_VIEW_USE_2_VIEWPORTS_ALT);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/3_viewports", TTR("3 Viewports"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_3), MENU_VIEW_USE_3_VIEWPORTS);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/3_viewports_alt", TTR("3 Viewports (Alt)"), KeyModifierMask::ALT + KeyModifierMask::CMD_OR_CTRL + Key::KEY_3), MENU_VIEW_USE_3_VIEWPORTS_ALT);
	p->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/4_viewports", TTR("4 Viewports"), KeyModifierMask::CMD_OR_CTRL + Key::KEY_4), MENU_VIEW_USE_4_VIEWPORTS);
	p->add_separator();

	p->add_submenu_item(TTR("Gizmos"), "GizmosMenu");

	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_origin", TTR("View Origin")), MENU_VIEW_ORIGIN);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_grid", TTR("View Grid"), Key::NUMBERSIGN), MENU_VIEW_GRID);

	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/settings", TTR("Settings...")), MENU_VIEW_CAMERA_SETTINGS);

	p->set_item_checked(p->get_item_index(MENU_VIEW_ORIGIN), true);
	p->set_item_checked(p->get_item_index(MENU_VIEW_GRID), true);

	p->connect("id_pressed", callable_mp(this, &Node3DEditor::_menu_item_pressed));

	gizmos_menu = memnew(PopupMenu);
	p->add_child(gizmos_menu);
	gizmos_menu->set_name("GizmosMenu");
	gizmos_menu->set_hide_on_checkable_item_selection(false);
	gizmos_menu->connect("id_pressed", callable_mp(this, &Node3DEditor::_menu_gizmo_toggled));

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
		viewports[i]->connect("clicked", callable_mp(this, &Node3DEditor::_update_camera_override_viewport));
		viewports[i]->assign_pending_data_pointers(preview_node, &preview_bounds, accept);
		viewport_base->add_child(viewports[i]);
	}

	/* SNAP DIALOG */

	snap_translate_value = 1;
	snap_rotate_value = 15;
	snap_scale_value = 10;

	snap_dialog = memnew(ConfirmationDialog);
	snap_dialog->set_title(TTR("Snap Settings"));
	add_child(snap_dialog);
	snap_dialog->connect("confirmed", callable_mp(this, &Node3DEditor::_snap_changed));
	snap_dialog->get_cancel_button()->connect("pressed", callable_mp(this, &Node3DEditor::_snap_update));

	VBoxContainer *snap_dialog_vbc = memnew(VBoxContainer);
	snap_dialog->add_child(snap_dialog_vbc);

	snap_translate = memnew(LineEdit);
	snap_translate->set_select_all_on_focus(true);
	snap_dialog_vbc->add_margin_child(TTR("Translate Snap:"), snap_translate);

	snap_rotate = memnew(LineEdit);
	snap_rotate->set_select_all_on_focus(true);
	snap_dialog_vbc->add_margin_child(TTR("Rotate Snap (deg.):"), snap_rotate);

	snap_scale = memnew(LineEdit);
	snap_scale->set_select_all_on_focus(true);
	snap_dialog_vbc->add_margin_child(TTR("Scale Snap (%):"), snap_scale);

	_snap_update();

	/* SETTINGS DIALOG */

	settings_dialog = memnew(ConfirmationDialog);
	settings_dialog->set_title(TTR("Viewport Settings"));
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
	settings_vbc->add_margin_child(TTR("Perspective FOV (deg.):"), settings_fov);

	settings_znear = memnew(SpinBox);
	settings_znear->set_max(MAX_Z);
	settings_znear->set_min(MIN_Z);
	settings_znear->set_step(0.01);
	settings_znear->set_value(EDITOR_GET("editors/3d/default_z_near"));
	settings_znear->set_select_all_on_focus(true);
	settings_vbc->add_margin_child(TTR("View Z-Near:"), settings_znear);

	settings_zfar = memnew(SpinBox);
	settings_zfar->set_max(MAX_Z);
	settings_zfar->set_min(MIN_Z);
	settings_zfar->set_step(0.1);
	settings_zfar->set_value(EDITOR_GET("editors/3d/default_z_far"));
	settings_zfar->set_select_all_on_focus(true);
	settings_vbc->add_margin_child(TTR("View Z-Far:"), settings_zfar);

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; ++i) {
		settings_dialog->connect("confirmed", callable_mp(viewports[i], &Node3DEditorViewport::_view_settings_confirmed).bind(0.0));
	}

	/* XFORM DIALOG */

	xform_dialog = memnew(ConfirmationDialog);
	xform_dialog->set_title(TTR("Transform Change"));
	add_child(xform_dialog);

	VBoxContainer *xform_vbc = memnew(VBoxContainer);
	xform_dialog->add_child(xform_vbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Translate:"));
	xform_vbc->add_child(l);

	HBoxContainer *xform_hbc = memnew(HBoxContainer);
	xform_vbc->add_child(xform_hbc);

	for (int i = 0; i < 3; i++) {
		xform_translate[i] = memnew(LineEdit);
		xform_translate[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_translate[i]->set_select_all_on_focus(true);
		xform_hbc->add_child(xform_translate[i]);
	}

	l = memnew(Label);
	l->set_text(TTR("Rotate (deg.):"));
	xform_vbc->add_child(l);

	xform_hbc = memnew(HBoxContainer);
	xform_vbc->add_child(xform_hbc);

	for (int i = 0; i < 3; i++) {
		xform_rotate[i] = memnew(LineEdit);
		xform_rotate[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_rotate[i]->set_select_all_on_focus(true);
		xform_hbc->add_child(xform_rotate[i]);
	}

	l = memnew(Label);
	l->set_text(TTR("Scale (ratio):"));
	xform_vbc->add_child(l);

	xform_hbc = memnew(HBoxContainer);
	xform_vbc->add_child(xform_hbc);

	for (int i = 0; i < 3; i++) {
		xform_scale[i] = memnew(LineEdit);
		xform_scale[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_scale[i]->set_select_all_on_focus(true);
		xform_hbc->add_child(xform_scale[i]);
	}

	l = memnew(Label);
	l->set_text(TTR("Transform Type"));
	xform_vbc->add_child(l);

	xform_type = memnew(OptionButton);
	xform_type->set_h_size_flags(SIZE_EXPAND_FILL);
	xform_type->add_item(TTR("Pre"));
	xform_type->add_item(TTR("Post"));
	xform_vbc->add_child(xform_type);

	xform_dialog->connect("confirmed", callable_mp(this, &Node3DEditor::_xform_dialog_action));

	selected = nullptr;

	set_process_shortcut_input(true);
	add_to_group("_spatial_editor_group");

	EDITOR_DEF("editors/3d/manipulator_gizmo_size", 80);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "editors/3d/manipulator_gizmo_size", PROPERTY_HINT_RANGE, "16,160,1"));
	EDITOR_DEF("editors/3d/manipulator_gizmo_opacity", 0.9);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::FLOAT, "editors/3d/manipulator_gizmo_opacity", PROPERTY_HINT_RANGE, "0,1,0.01"));
	EDITOR_DEF("editors/3d/navigation/show_viewport_rotation_gizmo", true);
	EDITOR_DEF("editors/3d/navigation/show_viewport_navigation_gizmo", DisplayServer::get_singleton()->is_touchscreen_available());

	current_hover_gizmo_handle = -1;
	current_hover_gizmo_handle_secondary = false;
	{
		//sun popup

		sun_environ_popup = memnew(PopupPanel);
		add_child(sun_environ_popup);

		HBoxContainer *sun_environ_hb = memnew(HBoxContainer);

		sun_environ_popup->add_child(sun_environ_hb);

		sun_vb = memnew(VBoxContainer);
		sun_environ_hb->add_child(sun_vb);
		sun_vb->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
		sun_vb->hide();

		sun_title = memnew(Label);
		sun_title->set_theme_type_variation("HeaderSmall");
		sun_vb->add_child(sun_title);
		sun_title->set_text(TTR("Preview Sun"));
		sun_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		CenterContainer *sun_direction_center = memnew(CenterContainer);
		sun_direction = memnew(Control);
		sun_direction->set_custom_minimum_size(Size2(128, 128) * EDSCALE);
		sun_direction_center->add_child(sun_direction);
		sun_vb->add_margin_child(TTR("Sun Direction"), sun_direction_center);
		sun_direction->connect("gui_input", callable_mp(this, &Node3DEditor::_sun_direction_input));
		sun_direction->connect("draw", callable_mp(this, &Node3DEditor::_sun_direction_draw));
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
		VBoxContainer *sun_angle_altitude_vbox = memnew(VBoxContainer);
		Label *sun_angle_altitude_label = memnew(Label);
		sun_angle_altitude_label->set_text(TTR("Angular Altitude"));
		sun_angle_altitude_vbox->add_child(sun_angle_altitude_label);
		sun_angle_altitude = memnew(EditorSpinSlider);
		sun_angle_altitude->set_max(90);
		sun_angle_altitude->set_min(-90);
		sun_angle_altitude->set_step(0.1);
		sun_angle_altitude->connect("value_changed", callable_mp(this, &Node3DEditor::_sun_direction_angle_set).unbind(1));
		sun_angle_altitude_vbox->add_child(sun_angle_altitude);
		sun_angle_hbox->add_child(sun_angle_altitude_vbox);
		VBoxContainer *sun_angle_azimuth_vbox = memnew(VBoxContainer);
		sun_angle_azimuth_vbox->set_custom_minimum_size(Vector2(100, 0));
		Label *sun_angle_azimuth_label = memnew(Label);
		sun_angle_azimuth_label->set_text(TTR("Azimuth"));
		sun_angle_azimuth_vbox->add_child(sun_angle_azimuth_label);
		sun_angle_azimuth = memnew(EditorSpinSlider);
		sun_angle_azimuth->set_max(180);
		sun_angle_azimuth->set_min(-180);
		sun_angle_azimuth->set_step(0.1);
		sun_angle_azimuth->set_allow_greater(true);
		sun_angle_azimuth->set_allow_lesser(true);
		sun_angle_azimuth->connect("value_changed", callable_mp(this, &Node3DEditor::_sun_direction_angle_set).unbind(1));
		sun_angle_azimuth_vbox->add_child(sun_angle_azimuth);
		sun_angle_hbox->add_child(sun_angle_azimuth_vbox);
		sun_angle_hbox->add_theme_constant_override("separation", 10);
		sun_vb->add_child(sun_angle_hbox);

		sun_color = memnew(ColorPickerButton);
		sun_color->set_edit_alpha(false);
		sun_vb->add_margin_child(TTR("Sun Color"), sun_color);
		sun_color->connect("color_changed", callable_mp(this, &Node3DEditor::_preview_settings_changed).unbind(1));
		sun_color->get_popup()->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(sun_color->get_picker()));

		sun_energy = memnew(EditorSpinSlider);
		sun_vb->add_margin_child(TTR("Sun Energy"), sun_energy);
		sun_energy->connect("value_changed", callable_mp(this, &Node3DEditor::_preview_settings_changed).unbind(1));
		sun_energy->set_max(64.0);

		sun_max_distance = memnew(EditorSpinSlider);
		sun_vb->add_margin_child(TTR("Shadow Max Distance"), sun_max_distance);
		sun_max_distance->connect("value_changed", callable_mp(this, &Node3DEditor::_preview_settings_changed).unbind(1));
		sun_max_distance->set_min(1);
		sun_max_distance->set_max(4096);

		sun_add_to_scene = memnew(Button);
		sun_add_to_scene->set_text(TTR("Add Sun to Scene"));
		sun_add_to_scene->set_tooltip_text(TTR("Adds a DirectionalLight3D node matching the preview sun settings to the current scene.\nHold Shift while clicking to also add the preview environment to the current scene."));
		sun_add_to_scene->connect("pressed", callable_mp(this, &Node3DEditor::_add_sun_to_scene).bind(false));
		sun_vb->add_spacer();
		sun_vb->add_child(sun_add_to_scene);

		sun_state = memnew(Label);
		sun_environ_hb->add_child(sun_state);
		sun_state->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		sun_state->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
		sun_state->set_h_size_flags(SIZE_EXPAND_FILL);

		VSeparator *sc = memnew(VSeparator);
		sc->set_custom_minimum_size(Size2(50 * EDSCALE, 0));
		sc->set_v_size_flags(SIZE_EXPAND_FILL);
		sun_environ_hb->add_child(sc);

		environ_vb = memnew(VBoxContainer);
		sun_environ_hb->add_child(environ_vb);
		environ_vb->set_custom_minimum_size(Size2(200 * EDSCALE, 0));
		environ_vb->hide();

		environ_title = memnew(Label);
		environ_title->set_theme_type_variation("HeaderSmall");

		environ_vb->add_child(environ_title);
		environ_title->set_text(TTR("Preview Environment"));
		environ_title->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);

		environ_sky_color = memnew(ColorPickerButton);
		environ_sky_color->set_edit_alpha(false);
		environ_sky_color->connect("color_changed", callable_mp(this, &Node3DEditor::_preview_settings_changed).unbind(1));
		environ_sky_color->get_popup()->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(environ_sky_color->get_picker()));
		environ_vb->add_margin_child(TTR("Sky Color"), environ_sky_color);
		environ_ground_color = memnew(ColorPickerButton);
		environ_ground_color->connect("color_changed", callable_mp(this, &Node3DEditor::_preview_settings_changed).unbind(1));
		environ_ground_color->set_edit_alpha(false);
		environ_ground_color->get_popup()->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(environ_ground_color->get_picker()));
		environ_vb->add_margin_child(TTR("Ground Color"), environ_ground_color);
		environ_energy = memnew(EditorSpinSlider);
		environ_energy->connect("value_changed", callable_mp(this, &Node3DEditor::_preview_settings_changed).unbind(1));
		environ_energy->set_max(8.0);
		environ_vb->add_margin_child(TTR("Sky Energy"), environ_energy);
		HBoxContainer *fx_vb = memnew(HBoxContainer);
		fx_vb->set_h_size_flags(SIZE_EXPAND_FILL);

		environ_ao_button = memnew(Button);
		environ_ao_button->set_text(TTR("AO"));
		environ_ao_button->set_toggle_mode(true);
		environ_ao_button->connect("pressed", callable_mp(this, &Node3DEditor::_preview_settings_changed), CONNECT_DEFERRED);
		fx_vb->add_child(environ_ao_button);
		environ_glow_button = memnew(Button);
		environ_glow_button->set_text(TTR("Glow"));
		environ_glow_button->set_toggle_mode(true);
		environ_glow_button->connect("pressed", callable_mp(this, &Node3DEditor::_preview_settings_changed), CONNECT_DEFERRED);
		fx_vb->add_child(environ_glow_button);
		environ_tonemap_button = memnew(Button);
		environ_tonemap_button->set_text(TTR("Tonemap"));
		environ_tonemap_button->set_toggle_mode(true);
		environ_tonemap_button->connect("pressed", callable_mp(this, &Node3DEditor::_preview_settings_changed), CONNECT_DEFERRED);
		fx_vb->add_child(environ_tonemap_button);
		environ_gi_button = memnew(Button);
		environ_gi_button->set_text(TTR("GI"));
		environ_gi_button->set_toggle_mode(true);
		environ_gi_button->connect("pressed", callable_mp(this, &Node3DEditor::_preview_settings_changed), CONNECT_DEFERRED);
		fx_vb->add_child(environ_gi_button);
		environ_vb->add_margin_child(TTR("Post Process"), fx_vb);

		environ_add_to_scene = memnew(Button);
		environ_add_to_scene->set_text(TTR("Add Environment to Scene"));
		environ_add_to_scene->set_tooltip_text(TTR("Adds a WorldEnvironment node matching the preview environment settings to the current scene.\nHold Shift while clicking to also add the preview sun to the current scene."));
		environ_add_to_scene->connect("pressed", callable_mp(this, &Node3DEditor::_add_environment_to_scene).bind(false));
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

		_load_default_preview_settings();
		_preview_settings_changed();
	}
}
Node3DEditor::~Node3DEditor() {
	memdelete(preview_node);
	if (preview_sun_dangling && preview_sun) {
		memdelete(preview_sun);
	}
	if (preview_env_dangling && preview_environment) {
		memdelete(preview_environment);
	}
}
