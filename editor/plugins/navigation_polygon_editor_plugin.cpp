/*************************************************************************/
/*  navigation_polygon_editor_plugin.cpp                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "navigation_polygon_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_settings.h"
#include "os/file_access.h"

void NavigationPolygonEditor::_enter_edit_mode() {

	mode = MODE_EDIT;
	button_edit->set_pressed(true);
	button_create->set_pressed(false);
}

bool NavigationPolygonEditor::_is_in_create_mode() const {

	return mode == MODE_CREATE;
}

bool NavigationPolygonEditor::_is_in_edit_mode() const {

	return mode == MODE_EDIT;
}

Node2D *NavigationPolygonEditor::_get_node() const {

	return node;
}

void NavigationPolygonEditor::_set_node(Node *p_node) {

	node = Object::cast_to<NavigationPolygonInstance>(p_node);
}

int NavigationPolygonEditor::_get_polygon_count() const {

	Ref<NavigationPolygon> poly = node->get_navigation_polygon();
	return !poly.is_null() ? poly->get_outline_count() : 0;
}

Vector<Vector2> NavigationPolygonEditor::_get_polygon(int p_polygon) const {

	return Variant(node->get_navigation_polygon()->get_outline(p_polygon));
}

void NavigationPolygonEditor::_set_polygon(int p_polygon, const Vector<Vector2> &p_points) const {

	node->get_navigation_polygon()->set_outline(p_polygon, Variant(p_points));
	node->get_navigation_polygon()->make_polygons_from_outlines();
}

Vector2 NavigationPolygonEditor::_get_offset() const {

	return Vector2(0, 0);
}

bool NavigationPolygonEditor::_is_wip_destructive() const {

	return false;
}

void NavigationPolygonEditor::_create_wip_close_action(const Vector<Vector2> &p_wip) {

	undo_redo->create_action(TTR("Create Poly"));
	undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "remove_outline", node->get_navigation_polygon()->get_outline_count());
	undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "add_outline", p_wip);
	undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "make_polygons_from_outlines");
	undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "make_polygons_from_outlines");
}

void NavigationPolygonEditor::_create_edit_poly_action(int p_polygon, const Vector<Vector2> &p_before, const Vector<Vector2> &p_after) {

	undo_redo->create_action(TTR("Edit Poly"));
	undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "set_outline", p_polygon, p_after);
	undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "set_outline", p_polygon, p_before);
	undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "make_polygons_from_outlines");
	undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "make_polygons_from_outlines");
}

void NavigationPolygonEditor::_create_remove_point_action(int p_polygon, int p_point) {

	Vector<Vector2> poly = _get_polygon(p_polygon);
	if (poly.size() > 3) {
		undo_redo->create_action(TTR("Edit Poly (Remove Point)"));
		undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "set_outline", p_polygon, poly);
		poly.remove(p_point);
		undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "set_outline", p_polygon, poly);
	} else {

		undo_redo->create_action(TTR("Remove Poly And Point"));
		undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "add_outline_at_index", poly, p_polygon);
		undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "remove_outline", p_polygon);
	}

	undo_redo->add_do_method(node->get_navigation_polygon().ptr(), "make_polygons_from_outlines");
	undo_redo->add_undo_method(node->get_navigation_polygon().ptr(), "make_polygons_from_outlines");
}

Color NavigationPolygonEditor::_get_previous_outline_color() const {

	return Color(1.0, 1.0, 0.0);
}

bool NavigationPolygonEditor::_can_input(const Ref<InputEvent> &p_event, bool &p_ret) const {

	if (node->get_navigation_polygon().is_null()) {

		Ref<InputEventMouseButton> mb = p_event;

		if (mb.is_valid() && mb->get_button_index() == 1 && mb->is_pressed()) {
			create_nav->set_text("No NavigationPolygon resource on this node.\nCreate and assign one?");
			create_nav->popup_centered_minsize();
		}
		p_ret = (mb.is_valid() && mb->get_button_index() == 1);
		return false;
	}

	return true;
}

bool NavigationPolygonEditor::_can_draw() const {

	return !node->get_navigation_polygon().is_null();
}

void NavigationPolygonEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			button_create->set_icon(get_icon("Edit", "EditorIcons"));
			button_edit->set_icon(get_icon("MovePoint", "EditorIcons"));
			button_edit->set_pressed(true);
			create_nav->connect("confirmed", this, "_create_nav");

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}

void NavigationPolygonEditor::_create_nav() {

	if (!node)
		return;

	undo_redo->create_action(TTR("Create Navigation Polygon"));
	undo_redo->add_do_method(node, "set_navigation_polygon", Ref<NavigationPolygon>(memnew(NavigationPolygon)));
	undo_redo->add_undo_method(node, "set_navigation_polygon", Variant(REF()));
	undo_redo->commit_action();
	_menu_option(MODE_CREATE);
}

void NavigationPolygonEditor::_menu_option(int p_option) {

	switch (p_option) {

		case MODE_CREATE: {

			mode = MODE_CREATE;
			button_create->set_pressed(true);
			button_edit->set_pressed(false);
		} break;
		case MODE_EDIT: {

			mode = MODE_EDIT;
			button_create->set_pressed(false);
			button_edit->set_pressed(true);
		} break;
	}
}

void NavigationPolygonEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_menu_option"), &NavigationPolygonEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_create_nav"), &NavigationPolygonEditor::_create_nav);
}

NavigationPolygonEditor::NavigationPolygonEditor(EditorNode *p_editor) : AbstractPolygon2DEditor(p_editor) {

	node = NULL;

	add_child(memnew(VSeparator));
	button_create = memnew(ToolButton);
	add_child(button_create);
	button_create->connect("pressed", this, "_menu_option", varray(MODE_CREATE));
	button_create->set_toggle_mode(true);
	button_create->set_tooltip(TTR("Create a new polygon from scratch."));

	button_edit = memnew(ToolButton);
	add_child(button_edit);
	button_edit->connect("pressed", this, "_menu_option", varray(MODE_EDIT));
	button_edit->set_toggle_mode(true);
	button_edit->set_tooltip(TTR("Edit existing polygon:") + "\n" + TTR("LMB: Move Point.") + "\n" + TTR("Ctrl+LMB: Split Segment.") + "\n" + TTR("RMB: Erase Point."));
	create_nav = memnew(ConfirmationDialog);
	add_child(create_nav);
	create_nav->get_ok()->set_text(TTR("Create"));

	mode = MODE_EDIT;
}

void NavigationPolygonEditorPlugin::edit(Object *p_object) {

	collision_polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool NavigationPolygonEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("NavigationPolygonInstance");
}

void NavigationPolygonEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		collision_polygon_editor->show();
	} else {

		collision_polygon_editor->hide();
		collision_polygon_editor->edit(NULL);
	}
}

NavigationPolygonEditorPlugin::NavigationPolygonEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	collision_polygon_editor = memnew(NavigationPolygonEditor(p_node));
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(collision_polygon_editor);

	collision_polygon_editor->hide();
}

NavigationPolygonEditorPlugin::~NavigationPolygonEditorPlugin() {
}
