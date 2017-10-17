/*************************************************************************/
/*  gradient_editor_plugin.cpp                                           */
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
#include "gradient_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "spatial_editor_plugin.h"

GradientEditorPlugin::GradientEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	ramp_editor = memnew(GradientEdit);

	add_control_to_container(CONTAINER_PROPERTY_EDITOR_BOTTOM, ramp_editor);

	ramp_editor->set_custom_minimum_size(Size2(100, 48));
	ramp_editor->hide();
	ramp_editor->connect("ramp_changed", this, "ramp_changed");
}

void GradientEditorPlugin::edit(Object *p_object) {

	Gradient *gradient = Object::cast_to<Gradient>(p_object);
	if (!gradient)
		return;
	gradient_ref = Ref<Gradient>(gradient);
	ramp_editor->set_points(gradient_ref->get_points());
}

bool GradientEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Gradient");
}

void GradientEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		ramp_editor->show();
	} else {
		ramp_editor->hide();
	}
}

void GradientEditorPlugin::_ramp_changed() {

	if (gradient_ref.is_valid()) {

		UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

		//Not sure if I should convert this data to PoolVector
		Vector<float> new_offsets = ramp_editor->get_offsets();
		Vector<Color> new_colors = ramp_editor->get_colors();
		Vector<float> old_offsets = gradient_ref->get_offsets();
		Vector<Color> old_colors = gradient_ref->get_colors();

		if (old_offsets.size() != new_offsets.size())
			ur->create_action(TTR("Add/Remove Color Ramp Point"));
		else
			ur->create_action(TTR("Modify Color Ramp"), UndoRedo::MERGE_ENDS);
		ur->add_do_method(this, "undo_redo_gradient", new_offsets, new_colors);
		ur->add_undo_method(this, "undo_redo_gradient", old_offsets, old_colors);
		ur->commit_action();

		//color_ramp_ref->set_points(ramp_editor->get_points());
	}
}

void GradientEditorPlugin::_undo_redo_gradient(const Vector<float> &offsets, const Vector<Color> &colors) {

	gradient_ref->set_offsets(offsets);
	gradient_ref->set_colors(colors);
	ramp_editor->set_points(gradient_ref->get_points());
	ramp_editor->update();
}

GradientEditorPlugin::~GradientEditorPlugin() {
}

void GradientEditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("ramp_changed"), &GradientEditorPlugin::_ramp_changed);
	ClassDB::bind_method(D_METHOD("undo_redo_gradient", "offsets", "colors"), &GradientEditorPlugin::_undo_redo_gradient);
}
