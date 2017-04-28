/*************************************************************************/
/*  color_ramp_editor_plugin.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "color_ramp_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "spatial_editor_plugin.h"

ColorRampEditorPlugin::ColorRampEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	ramp_editor = memnew(ColorRampEdit);

	add_control_to_container(CONTAINER_PROPERTY_EDITOR_BOTTOM, ramp_editor);

	ramp_editor->set_custom_minimum_size(Size2(100, 48));
	ramp_editor->hide();
	ramp_editor->connect("ramp_changed", this, "ramp_changed");
}

void ColorRampEditorPlugin::edit(Object *p_object) {

	ColorRamp *color_ramp = p_object->cast_to<ColorRamp>();
	if (!color_ramp)
		return;
	color_ramp_ref = Ref<ColorRamp>(color_ramp);
	ramp_editor->set_points(color_ramp_ref->get_points());
}

bool ColorRampEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("ColorRamp");
}

void ColorRampEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		ramp_editor->show();
	} else {
		ramp_editor->hide();
	}
}

void ColorRampEditorPlugin::_ramp_changed() {

	if (color_ramp_ref.is_valid()) {

		UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

		//Not sure if I should convert this data to PoolVector
		Vector<float> new_offsets = ramp_editor->get_offsets();
		Vector<Color> new_colors = ramp_editor->get_colors();
		Vector<float> old_offsets = color_ramp_ref->get_offsets();
		Vector<Color> old_colors = color_ramp_ref->get_colors();

		if (old_offsets.size() != new_offsets.size())
			ur->create_action(TTR("Add/Remove Color Ramp Point"));
		else
			ur->create_action(TTR("Modify Color Ramp"), UndoRedo::MERGE_ENDS);
		ur->add_do_method(this, "undo_redo_color_ramp", new_offsets, new_colors);
		ur->add_undo_method(this, "undo_redo_color_ramp", old_offsets, old_colors);
		ur->commit_action();

		//color_ramp_ref->set_points(ramp_editor->get_points());
	}
}

void ColorRampEditorPlugin::_undo_redo_color_ramp(const Vector<float> &offsets,
		const Vector<Color> &colors) {

	color_ramp_ref->set_offsets(offsets);
	color_ramp_ref->set_colors(colors);
	ramp_editor->set_points(color_ramp_ref->get_points());
	ramp_editor->update();
}

ColorRampEditorPlugin::~ColorRampEditorPlugin() {
}

void ColorRampEditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("ramp_changed"), &ColorRampEditorPlugin::_ramp_changed);
	ClassDB::bind_method(D_METHOD("undo_redo_color_ramp", "offsets", "colors"), &ColorRampEditorPlugin::_undo_redo_color_ramp);
}
