/**************************************************************************/
/*  soft_body_3d_editor_plugin.cpp                                       */
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

#include "soft_body_3d_editor_plugin.h"

#include "core/object/callable_mp.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "scene/gui/label.h"

void SoftBody3DEditor::_menu_option(int p_option) {
	if (!node) {
		return;
	}

	switch (p_option) {
		case MENU_OPTION_SET_PIN_WEIGHTS_FROM_VERTEX_COLORS: {
			Ref<Mesh> mesh = node->get_mesh();
			if (mesh.is_null()) {
				err_dialog->set_text(TTR("SoftBody3D does not have a mesh assigned."));
				err_dialog->popup_centered();
				return;
			}
			weight_dialog->popup_centered(Size2i(340, 0));
		} break;
		case MENU_OPTION_CLEAR_ALL_PINNED_POINTS: {
			_clear_all_pinned_points();
		} break;
	}
}

void SoftBody3DEditor::_apply_pin_weights_from_vertex_colors() {
	if (!node) {
		return;
	}

	Ref<Mesh> mesh = node->get_mesh();
	if (mesh.is_null()) {
		err_dialog->set_text(TTR("SoftBody3D does not have a mesh assigned."));
		err_dialog->popup_centered();
		return;
	}

	int surface_count = mesh->get_surface_count();
	if (surface_count == 0) {
		err_dialog->set_text(TTR("Mesh has no surfaces."));
		err_dialog->popup_centered();
		return;
	}

	bool found_colors = false;
	for (int s = 0; s < surface_count; s++) {
		Array arrays = mesh->surface_get_arrays(s);
		if (arrays.size() > Mesh::ARRAY_COLOR && arrays[Mesh::ARRAY_COLOR].get_type() == Variant::PACKED_COLOR_ARRAY) {
			PackedColorArray colors = arrays[Mesh::ARRAY_COLOR];
			if (colors.size() > 0) {
				found_colors = true;
				break;
			}
		}
	}

	if (!found_colors) {
		err_dialog->set_text(TTR("Mesh does not contain vertex colors on any surface."));
		err_dialog->popup_centered();
		return;
	}

	int channel = channel_option->get_selected_id();
	real_t threshold = (real_t)threshold_spin->get_value();
	bool invert = invert_check->is_pressed();
	int mode = mode_option->get_selected_id();

	struct PinData {
		int point_index = 0;
		NodePath spatial_attachment_path;
		Vector3 offset;
		real_t weight = 1.0;
	};

	HashMap<int, PinData> target_pins;

	// If mode is merge (1), preload existing pins
	if (mode == 1) {
		Array cur_indices = node->get("pinned_points");
		for (int i = 0; i < cur_indices.size(); i++) {
			int p_idx = cur_indices[i];
			String prefix = vformat("attachments/%d/", i);
			PinData pd;
			pd.point_index = p_idx;
			pd.spatial_attachment_path = node->get(prefix + "spatial_attachment_path");
			pd.offset = node->get(prefix + "offset");
			pd.weight = node->get(prefix + "weight");
			target_pins[p_idx] = pd;
		}
	}

	int global_vertex_offset = 0;
	for (int s = 0; s < surface_count; s++) {
		Array arrays = mesh->surface_get_arrays(s);
		if (arrays.size() <= Mesh::ARRAY_VERTEX) {
			continue;
		}
		PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
		int v_count = vertices.size();
		PackedColorArray colors;
		if (arrays.size() > Mesh::ARRAY_COLOR && arrays[Mesh::ARRAY_COLOR].get_type() == Variant::PACKED_COLOR_ARRAY) {
			colors = arrays[Mesh::ARRAY_COLOR];
		}
		bool has_color = (colors.size() == v_count);

		if (!has_color) {
			global_vertex_offset += v_count;
			continue;
		}

		for (int v = 0; v < v_count; v++) {
			Color c = colors[v];
			real_t val = 1.0;
			switch (channel) {
				case 0: val = c.r; break;
				case 1: val = c.g; break;
				case 2: val = c.b; break;
				case 3: val = c.a; break;
				case 4: val = c.get_luminance(); break;
			}
			if (invert) {
				val = (real_t)1.0 - val;
			}
			val = CLAMP(val, (real_t)0.0, (real_t)1.0);

			int global_idx = global_vertex_offset + v;
			if (val >= threshold && val > (real_t)0.0) {
				PinData pd;
				if (target_pins.has(global_idx)) {
					pd = target_pins[global_idx];
				} else {
					pd.point_index = global_idx;
				}
				pd.weight = val;
				target_pins[global_idx] = pd;
			} else {
				// Weight is 0 or below threshold -> remove from target_pins in both replace and merge modes
				target_pins.erase(global_idx);
			}
		}
		global_vertex_offset += v_count;
	}

	// Prepare undo state
	Array prev_indices = node->get("pinned_points");
	Dictionary prev_props;
	for (int i = 0; i < prev_indices.size(); i++) {
		String prefix = vformat("attachments/%d/", i);
		prev_props[prefix + "point_index"] = node->get(prefix + "point_index");
		prev_props[prefix + "spatial_attachment_path"] = node->get(prefix + "spatial_attachment_path");
		prev_props[prefix + "offset"] = node->get(prefix + "offset");
		prev_props[prefix + "weight"] = node->get(prefix + "weight");
	}

	// Prepare do state
	Array new_indices;
	Dictionary new_props;
	int pin_idx = 0;
	for (const KeyValue<int, PinData> &E : target_pins) {
		if (E.value.point_index < 0 || E.value.weight <= (real_t)0.0) {
			continue;
		}
		new_indices.push_back(E.value.point_index);
		String prefix = vformat("attachments/%d/", pin_idx);
		new_props[prefix + "point_index"] = E.value.point_index;
		new_props[prefix + "spatial_attachment_path"] = E.value.spatial_attachment_path;
		new_props[prefix + "offset"] = E.value.offset;
		new_props[prefix + "weight"] = E.value.weight;
		pin_idx++;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Set SoftBody3D Pin Weights from Vertex Colors"));

	ur->add_do_property(node, "pinned_points", new_indices);
	for (const Variant *E = new_props.next(nullptr); E; E = new_props.next(E)) {
		ur->add_do_property(node, *E, new_props[*E]);
	}

	ur->add_undo_property(node, "pinned_points", prev_indices);
	for (const Variant *E = prev_props.next(nullptr); E; E = prev_props.next(E)) {
		ur->add_undo_property(node, *E, prev_props[*E]);
	}

	ur->add_do_method(node, "notify_property_list_changed");
	ur->add_undo_method(node, "notify_property_list_changed");
	ur->commit_action();
}

void SoftBody3DEditor::_clear_all_pinned_points() {
	if (!node) {
		return;
	}

	Array prev_indices = node->get("pinned_points");
	if (prev_indices.is_empty()) {
		return;
	}

	Dictionary prev_props;
	for (int i = 0; i < prev_indices.size(); i++) {
		String prefix = vformat("attachments/%d/", i);
		prev_props[prefix + "point_index"] = node->get(prefix + "point_index");
		prev_props[prefix + "spatial_attachment_path"] = node->get(prefix + "spatial_attachment_path");
		prev_props[prefix + "offset"] = node->get(prefix + "offset");
		prev_props[prefix + "weight"] = node->get(prefix + "weight");
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Clear SoftBody3D Pinned Points"));

	ur->add_do_property(node, "pinned_points", Array());
	ur->add_undo_property(node, "pinned_points", prev_indices);
	for (const Variant *E = prev_props.next(nullptr); E; E = prev_props.next(E)) {
		ur->add_undo_property(node, *E, prev_props[*E]);
	}

	ur->add_do_method(node, "notify_property_list_changed");
	ur->add_undo_method(node, "notify_property_list_changed");
	ur->commit_action();
}

void SoftBody3DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			if (options) {
				options->set_button_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("SoftBody3D"), EditorStringName(EditorIcons)));
			}
		} break;
	}
}

void SoftBody3DEditor::edit(SoftBody3D *p_soft_body) {
	node = p_soft_body;
}

SoftBody3DEditor::SoftBody3DEditor() {
	options = memnew(MenuButton);
	options->set_switch_on_hover(true);
	options->set_flat(false);
	options->set_theme_type_variation("FlatMenuButtonNoIconTint");
	options->set_text(TTR("SoftBody3D"));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(options);

	options->get_popup()->add_item(TTR("Set Pin Weights from Vertex Colors..."), MENU_OPTION_SET_PIN_WEIGHTS_FROM_VERTEX_COLORS);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Clear All Pinned Points"), MENU_OPTION_CLEAR_ALL_PINNED_POINTS);
	options->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &SoftBody3DEditor::_menu_option));

	weight_dialog = memnew(ConfirmationDialog);
	weight_dialog->set_title(TTR("Set Pin Weights from Vertex Colors"));
	weight_dialog->set_ok_button_text(TTR("Apply"));
	add_child(weight_dialog);
	weight_dialog->connect(SceneStringName(confirmed), callable_mp(this, &SoftBody3DEditor::_apply_pin_weights_from_vertex_colors));

	VBoxContainer *vbc = memnew(VBoxContainer);
	weight_dialog->add_child(vbc);

	Label *lbl_channel = memnew(Label(TTR("Color Channel:")));
	vbc->add_child(lbl_channel);

	channel_option = memnew(OptionButton);
	channel_option->add_item(TTR("Red (R)"), 0);
	channel_option->add_item(TTR("Green (G)"), 1);
	channel_option->add_item(TTR("Blue (B)"), 2);
	channel_option->add_item(TTR("Alpha (A)"), 3);
	channel_option->add_item(TTR("Luminance (Grayscale)"), 4);
	channel_option->select(0);
	vbc->add_child(channel_option);

	Label *lbl_thresh = memnew(Label(TTR("Min Weight Threshold:")));
	vbc->add_child(lbl_thresh);

	threshold_spin = memnew(SpinBox);
	threshold_spin->set_min(0.0);
	threshold_spin->set_max(1.0);
	threshold_spin->set_step(0.001);
	threshold_spin->set_value(0.01);
	vbc->add_child(threshold_spin);

	invert_check = memnew(CheckBox(TTR("Invert Weights")));
	vbc->add_child(invert_check);

	Label *lbl_mode = memnew(Label(TTR("Assignment Mode:")));
	vbc->add_child(lbl_mode);

	mode_option = memnew(OptionButton);
	mode_option->add_item(TTR("Replace All Pins"), 0);
	mode_option->add_item(TTR("Merge / Update Existing Pins"), 1);
	mode_option->select(0);
	vbc->add_child(mode_option);

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);

	options->hide();
}

SoftBody3DEditor::~SoftBody3DEditor() {
}

void SoftBody3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		soft_body_editor->options->show();
	} else {
		soft_body_editor->options->hide();
		soft_body_editor->edit(nullptr);
	}
}

void SoftBody3DEditorPlugin::edit(Object *p_object) {
	SoftBody3D *sb = Object::cast_to<SoftBody3D>(p_object);
	if (sb) {
		soft_body_editor->edit(sb);
	}
}

bool SoftBody3DEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<SoftBody3D>(p_object) != nullptr;
}

SoftBody3DEditorPlugin::SoftBody3DEditorPlugin() {
	soft_body_editor = memnew(SoftBody3DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(soft_body_editor);
}
