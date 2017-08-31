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

EditablePolygon2D *NavigationPolygonEditor::_get_editable(Node *p_node) const {

	return Object::cast_to<NavigationPolygonInstance>(p_node)->get_navigation_polygon().ptr();
}

void NavigationPolygonEditor::_create_res() {

	if (!node)
		return;

	undo_redo->create_action(TTR("Create Navigation Polygon"));
	undo_redo->add_do_method(node, "set_navigation_polygon", Ref<NavigationPolygon>(memnew(NavigationPolygon)));
	undo_redo->add_undo_method(node, "set_navigation_polygon", Variant(REF()));
	undo_redo->commit_action();
	_menu_option(MODE_CREATE);

	editable = _get_editable(node);
}

NavigationPolygonEditor::NavigationPolygonEditor(EditorNode *p_editor) : AbstractPolygon2DEditor(p_editor) {

}

NavigationPolygonEditorPlugin::NavigationPolygonEditorPlugin(EditorNode *p_node) :

	AbstractPolygon2DEditorPlugin(p_node, memnew(NavigationPolygonEditor(p_node)), "NavigationPolygonInstance") {

}

