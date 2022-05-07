/*************************************************************************/
/*  room_manager_editor_plugin.cpp                                       */
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

#include "room_manager_editor_plugin.h"

#include "editor/spatial_editor_gizmos.h"

void RoomManagerEditorPlugin::_flip_portals() {
	if (_room_manager) {
		_room_manager->rooms_flip_portals();
	}
}

void RoomManagerEditorPlugin::edit(Object *p_object) {
	RoomManager *s = Object::cast_to<RoomManager>(p_object);
	if (!s) {
		return;
	}

	_room_manager = s;
}

bool RoomManagerEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("RoomManager");
}

void RoomManagerEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button_flip_portals->show();
	} else {
		button_flip_portals->hide();
	}

	SpatialEditor::get_singleton()->show_advanced_portal_tools(p_visible);
}

void RoomManagerEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_flip_portals", &RoomManagerEditorPlugin::_flip_portals);
}

RoomManagerEditorPlugin::RoomManagerEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	button_flip_portals = memnew(ToolButton);
	button_flip_portals->set_icon(editor->get_gui_base()->get_icon("Portal", "EditorIcons"));
	button_flip_portals->set_text(TTR("Flip Portals"));
	button_flip_portals->hide();
	button_flip_portals->connect("pressed", this, "_flip_portals");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, button_flip_portals);

	_room_manager = nullptr;

	Ref<RoomGizmoPlugin> room_gizmo_plugin = Ref<RoomGizmoPlugin>(memnew(RoomGizmoPlugin));
	SpatialEditor::get_singleton()->add_gizmo_plugin(room_gizmo_plugin);

	Ref<PortalGizmoPlugin> portal_gizmo_plugin = Ref<PortalGizmoPlugin>(memnew(PortalGizmoPlugin));
	SpatialEditor::get_singleton()->add_gizmo_plugin(portal_gizmo_plugin);

	Ref<OccluderGizmoPlugin> occluder_gizmo_plugin = Ref<OccluderGizmoPlugin>(memnew(OccluderGizmoPlugin));
	SpatialEditor::get_singleton()->add_gizmo_plugin(occluder_gizmo_plugin);
}

RoomManagerEditorPlugin::~RoomManagerEditorPlugin() {
}

///////////////////////

void RoomEditorPlugin::_generate_points() {
	if (_room) {
		PoolVector<Vector3> old_pts = _room->get_points();

		// only generate points if none already exist
		if (_room->_bound_pts.size()) {
			_room->set_points(PoolVector<Vector3>());
		}

		PoolVector<Vector3> pts = _room->generate_points();

		// allow the user to undo generating points, because it is
		// frustrating to lose old data
		undo_redo->create_action(TTR("Room Generate Points"));
		undo_redo->add_do_property(_room, "points", pts);
		undo_redo->add_undo_property(_room, "points", old_pts);
		undo_redo->commit_action();
	}
}

void RoomEditorPlugin::edit(Object *p_object) {
	Room *s = Object::cast_to<Room>(p_object);
	if (!s) {
		return;
	}

	_room = s;

	if (SpatialEditor::get_singleton()->is_visible() && s->_planes.size()) {
		String string = String(s->get_name()) + " [" + itos(s->_planes.size()) + " planes]";
		SpatialEditor::get_singleton()->set_message(string);
	}
}

bool RoomEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Room");
}

void RoomEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button_generate->show();
	} else {
		button_generate->hide();
	}
}

void RoomEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_generate_points", &RoomEditorPlugin::_generate_points);
}

RoomEditorPlugin::RoomEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	button_generate = memnew(ToolButton);
	button_generate->set_icon(editor->get_gui_base()->get_icon("Room", "EditorIcons"));
	button_generate->set_text(TTR("Generate Points"));
	button_generate->hide();
	button_generate->connect("pressed", this, "_generate_points");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, button_generate);

	_room = nullptr;

	undo_redo = EditorNode::get_undo_redo();
}

RoomEditorPlugin::~RoomEditorPlugin() {
}

///////////////////////

void PortalEditorPlugin::_flip_portal() {
	if (_portal) {
		_portal->flip();
		_portal->_changed();
	}
}

void PortalEditorPlugin::edit(Object *p_object) {
	Portal *p = Object::cast_to<Portal>(p_object);
	if (!p) {
		return;
	}

	_portal = p;
}

bool PortalEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Portal");
}

void PortalEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button_flip->show();
	} else {
		button_flip->hide();
	}
}

void PortalEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_flip_portal", &PortalEditorPlugin::_flip_portal);
}

PortalEditorPlugin::PortalEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	button_flip = memnew(ToolButton);
	button_flip->set_icon(editor->get_gui_base()->get_icon("Portal", "EditorIcons"));
	button_flip->set_text(TTR("Flip Portal"));
	button_flip->hide();
	button_flip->connect("pressed", this, "_flip_portal");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, button_flip);

	_portal = nullptr;
}

PortalEditorPlugin::~PortalEditorPlugin() {
}

///////////////////////

void OccluderEditorPlugin::_center() {
	if (_occluder && _occluder->is_inside_tree()) {
		Ref<OccluderShape> ref = _occluder->get_shape();

		if (ref.is_valid()) {
			Spatial *parent = Object::cast_to<Spatial>(_occluder->get_parent());
			if (parent) {
				real_t snap = 0.0;

				if (Engine::get_singleton()->is_editor_hint()) {
					if (SpatialEditor::get_singleton() && SpatialEditor::get_singleton()->is_snap_enabled()) {
						snap = SpatialEditor::get_singleton()->get_translate_snap();
					}
				}

				Transform old_local_xform = _occluder->get_transform();
				Transform new_local_xform = ref->center_node(_occluder->get_global_transform(), parent->get_global_transform(), snap);
				_occluder->property_list_changed_notify();

				undo_redo->create_action(TTR("Occluder Set Transform"));
				undo_redo->add_do_method(_occluder, "set_transform", new_local_xform);
				undo_redo->add_undo_method(_occluder, "set_transform", old_local_xform);
				undo_redo->commit_action();

				_occluder->update_gizmo();
			}
		}
	}
}

void OccluderEditorPlugin::edit(Object *p_object) {
	Occluder *p = Object::cast_to<Occluder>(p_object);
	if (!p) {
		return;
	}

	_occluder = p;
}

bool OccluderEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Occluder");
}

void OccluderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button_center->show();
	} else {
		button_center->hide();
	}
}

void OccluderEditorPlugin::_bind_methods() {
	ClassDB::bind_method("_center", &OccluderEditorPlugin::_center);
}

OccluderEditorPlugin::OccluderEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	button_center = memnew(ToolButton);
	button_center->set_icon(editor->get_gui_base()->get_icon("EditorPosition", "EditorIcons"));
	button_center->set_text(TTR("Center Node"));
	button_center->hide();
	button_center->connect("pressed", this, "_center");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, button_center);

	undo_redo = EditorNode::get_undo_redo();

	_occluder = nullptr;
}

OccluderEditorPlugin::~OccluderEditorPlugin() {
}
