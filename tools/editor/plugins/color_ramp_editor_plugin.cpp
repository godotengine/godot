/*
 * color_ramp_editor_plugin.cpp
 */

#include "color_ramp_editor_plugin.h"
#include "spatial_editor_plugin.h"
#include "canvas_item_editor_plugin.h"

ColorRampEditorPlugin::ColorRampEditorPlugin(EditorNode *p_node, bool p_2d) {

	editor=p_node;
	ramp_editor = memnew( ColorRampEdit );

	_2d=p_2d;
	if (p_2d)
		add_control_to_container(CONTAINER_CANVAS_EDITOR_BOTTOM,ramp_editor);
	else
		add_control_to_container(CONTAINER_SPATIAL_EDITOR_BOTTOM,ramp_editor);

	ramp_editor->set_custom_minimum_size(Size2(100, 48));
	ramp_editor->hide();
	ramp_editor->connect("ramp_changed", this, "ramp_changed");
}

void ColorRampEditorPlugin::edit(Object *p_object) {

	ColorRamp* color_ramp = p_object->cast_to<ColorRamp>();
	if (!color_ramp)
		return;
	color_ramp_ref = Ref<ColorRamp>(color_ramp);
	ramp_editor->set_points(color_ramp_ref->get_points());
}

bool ColorRampEditorPlugin::handles(Object *p_object) const {

	if (_2d)
		return p_object->is_type("ColorRamp") && CanvasItemEditor::get_singleton()->is_visible() == true;
	else
		return p_object->is_type("ColorRamp") && SpatialEditor::get_singleton()->is_visible() == true;
}

void ColorRampEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		ramp_editor->show();
	} else {
		ramp_editor->hide();
	}

}

void ColorRampEditorPlugin::_ramp_changed() {

	if(color_ramp_ref.is_valid())
	{

		UndoRedo *ur=EditorNode::get_singleton()->get_undo_redo();

		//Not sure if I should convert this data to DVector
		Vector<float> new_offsets=ramp_editor->get_offsets();
		Vector<Color> new_colors=ramp_editor->get_colors();
		Vector<float> old_offsets=color_ramp_ref->get_offsets();
		Vector<Color> old_colors=color_ramp_ref->get_colors();

		if (old_offsets.size()!=new_offsets.size())
			ur->create_action("Add/Remove Color Ramp Point");
		else
			ur->create_action("Modify Color Ramp",true);
		ur->add_do_method(this,"undo_redo_color_ramp",new_offsets,new_colors);
		ur->add_undo_method(this,"undo_redo_color_ramp",old_offsets,old_colors);
		ur->commit_action();

		//color_ramp_ref->set_points(ramp_editor->get_points());
	}
}

void ColorRampEditorPlugin::_undo_redo_color_ramp(const Vector<float>& offsets,
		const Vector<Color>& colors) {

	color_ramp_ref->set_offsets(offsets);
	color_ramp_ref->set_colors(colors);
	ramp_editor->set_points(color_ramp_ref->get_points());
	ramp_editor->update();
}

ColorRampEditorPlugin::~ColorRampEditorPlugin(){
}

void ColorRampEditorPlugin::_bind_methods() {
	ObjectTypeDB::bind_method(_MD("ramp_changed"),&ColorRampEditorPlugin::_ramp_changed);
	ObjectTypeDB::bind_method(_MD("undo_redo_color_ramp","offsets","colors"),&ColorRampEditorPlugin::_undo_redo_color_ramp);
}
