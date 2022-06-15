/*************************************************************************/
/*  editor_properties_array_dict.cpp                                     */
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

#include "editor_properties_array_dict.h"

#include "core/io/marshalls.h"
#include "core/os/input.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor_properties.h"
#include "scene/main/viewport.h"

bool EditorPropertyArrayObject::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name.begins_with("indices")) {
		int index = name.get_slicec('/', 1).to_int();
		array.set(index, p_value);
		return true;
	}

	return false;
}

bool EditorPropertyArrayObject::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name.begins_with("indices")) {
		int index = name.get_slicec('/', 1).to_int();
		bool valid;
		r_ret = array.get(index, &valid);
		if (r_ret.get_type() == Variant::OBJECT && Object::cast_to<EncodedObjectAsID>(r_ret)) {
			r_ret = Object::cast_to<EncodedObjectAsID>(r_ret)->get_object_id();
		}

		return valid;
	}

	return false;
}

void EditorPropertyArrayObject::set_array(const Variant &p_array) {
	array = p_array;
}

Variant EditorPropertyArrayObject::get_array() {
	return array;
}

EditorPropertyArrayObject::EditorPropertyArrayObject() {
}

///////////////////

bool EditorPropertyDictionaryObject::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (name == "new_item_key") {
		new_item_key = p_value;
		return true;
	}

	if (name == "new_item_value") {
		new_item_value = p_value;
		return true;
	}

	if (name.begins_with("indices")) {
		int index = name.get_slicec('/', 1).to_int();
		Variant key = dict.get_key_at_index(index);
		dict[key] = p_value;
		return true;
	}

	return false;
}

bool EditorPropertyDictionaryObject::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (name == "new_item_key") {
		r_ret = new_item_key;
		return true;
	}

	if (name == "new_item_value") {
		r_ret = new_item_value;
		return true;
	}

	if (name.begins_with("indices")) {
		int index = name.get_slicec('/', 1).to_int();
		Variant key = dict.get_key_at_index(index);
		r_ret = dict[key];
		if (r_ret.get_type() == Variant::OBJECT && Object::cast_to<EncodedObjectAsID>(r_ret)) {
			r_ret = Object::cast_to<EncodedObjectAsID>(r_ret)->get_object_id();
		}

		return true;
	}

	return false;
}

void EditorPropertyDictionaryObject::set_dict(const Dictionary &p_dict) {
	dict = p_dict;
}

Dictionary EditorPropertyDictionaryObject::get_dict() {
	return dict;
}

void EditorPropertyDictionaryObject::set_new_item_key(const Variant &p_new_item) {
	new_item_key = p_new_item;
}

Variant EditorPropertyDictionaryObject::get_new_item_key() {
	return new_item_key;
}

void EditorPropertyDictionaryObject::set_new_item_value(const Variant &p_new_item) {
	new_item_value = p_new_item;
}

Variant EditorPropertyDictionaryObject::get_new_item_value() {
	return new_item_value;
}

EditorPropertyDictionaryObject::EditorPropertyDictionaryObject() {
}

///////////////////// ARRAY ///////////////////////////

void EditorPropertyArray::_property_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	if (p_property.begins_with("indices")) {
		int index = p_property.get_slice("/", 1).to_int();
		Variant array = object->get_array();
		array.set(index, p_value);
		emit_changed(get_edited_property(), array, "", true);

		if (array.get_type() == Variant::ARRAY) {
			array = array.call("duplicate"); // Duplicate, so undo/redo works better.
		}
		object->set_array(array);
	}
}

void EditorPropertyArray::_change_type(Object *p_button, int p_index) {
	Button *button = Object::cast_to<Button>(p_button);
	changing_type_index = p_index;
	Rect2 rect = button->get_global_rect();
	change_type->set_as_minsize();
	change_type->set_global_position(rect.position + rect.size - Vector2(change_type->get_combined_minimum_size().x, 0));
	change_type->popup();
}

void EditorPropertyArray::_change_type_menu(int p_index) {
	if (p_index == Variant::VARIANT_MAX) {
		_remove_pressed(changing_type_index);
		return;
	}

	Variant value;
	Variant::CallError ce;
	value = Variant::construct(Variant::Type(p_index), nullptr, 0, ce);
	Variant array = object->get_array();
	array.set(changing_type_index, value);

	emit_changed(get_edited_property(), array, "", true);

	if (array.get_type() == Variant::ARRAY) {
		array = array.call("duplicate"); // Duplicate, so undo/redo works better.
	}

	object->set_array(array);
	update_property();
}

void EditorPropertyArray::_object_id_selected(const String &p_property, ObjectID p_id) {
	emit_signal("object_id_selected", p_property, p_id);
}

void EditorPropertyArray::update_property() {
	Variant array = get_edited_object()->get(get_edited_property());

	String arrtype = "";
	switch (array_type) {
		case Variant::ARRAY: {
			arrtype = "Array";

		} break;

		// Arrays.
		case Variant::POOL_BYTE_ARRAY: {
			arrtype = "PoolByteArray";

		} break;
		case Variant::POOL_INT_ARRAY: {
			arrtype = "PoolIntArray";

		} break;
		case Variant::POOL_REAL_ARRAY: {
			arrtype = "PoolFloatArray";
		} break;
		case Variant::POOL_STRING_ARRAY: {
			arrtype = "PoolStringArray";
		} break;
		case Variant::POOL_VECTOR2_ARRAY: {
			arrtype = "PoolVector2Array";
		} break;
		case Variant::POOL_VECTOR3_ARRAY: {
			arrtype = "PoolVector3Array";

		} break;
		case Variant::POOL_COLOR_ARRAY: {
			arrtype = "PoolColorArray";
		} break;
		default: {
		}
	}

	if (array.get_type() == Variant::NIL) {
		edit->set_text(String("(Nil) ") + arrtype);
		edit->set_pressed(false);
		if (vbox) {
			set_bottom_editor(nullptr);
			memdelete(vbox);
			vbox = nullptr;
		}
		return;
	}

	int size = array.call("size");
	int pages = MAX(0, size - 1) / page_length + 1;
	page_index = MIN(page_index, pages - 1);
	int offset = page_index * page_length;

	edit->set_text(arrtype + " (size " + itos(size) + ")");

	bool unfolded = get_edited_object()->editor_is_section_unfolded(get_edited_property());
	if (edit->is_pressed() != unfolded) {
		edit->set_pressed(unfolded);
	}

	if (unfolded) {
		updating = true;

		if (!vbox) {
			vbox = memnew(VBoxContainer);
			add_child(vbox);
			set_bottom_editor(vbox);

			HBoxContainer *hbox = memnew(HBoxContainer);
			vbox->add_child(hbox);

			Label *label = memnew(Label(TTR("Size:")));
			label->set_h_size_flags(SIZE_EXPAND_FILL);
			hbox->add_child(label);

			size_slider = memnew(EditorSpinSlider);
			size_slider->set_step(1);
			size_slider->set_max(1000000);
			size_slider->set_h_size_flags(SIZE_EXPAND_FILL);
			size_slider->connect("value_changed", this, "_length_changed");
			hbox->add_child(size_slider);

			page_hbox = memnew(HBoxContainer);
			vbox->add_child(page_hbox);

			label = memnew(Label(TTR("Page:")));
			label->set_h_size_flags(SIZE_EXPAND_FILL);
			page_hbox->add_child(label);

			page_slider = memnew(EditorSpinSlider);
			page_slider->set_step(1);
			page_slider->set_h_size_flags(SIZE_EXPAND_FILL);
			page_slider->connect("value_changed", this, "_page_changed");
			page_hbox->add_child(page_slider);
		} else {
			// Bye bye children of the box.
			for (int i = vbox->get_child_count() - 1; i >= 2; i--) {
				Node *child = vbox->get_child(i);
				if (child == reorder_selected_element_hbox) {
					continue; // Don't remove the property that the user is moving.
				}

				child->queue_delete(); // Button still needed after pressed is called.
				vbox->remove_child(child);
			}
		}

		size_slider->set_value(size);
		page_slider->set_max(pages);
		page_slider->set_value(page_index);
		page_hbox->set_visible(pages > 1);

		if (array.get_type() == Variant::ARRAY) {
			array = array.call("duplicate");
		}

		object->set_array(array);

		int amount = MIN(size - offset, page_length);
		for (int i = 0; i < amount; i++) {
			bool reorder_is_from_current_page = reorder_from_index / page_length == page_index;
			if (reorder_is_from_current_page && i == reorder_from_index % page_length) {
				// Don't duplicate the property that the user is moving.
				continue;
			}
			if (!reorder_is_from_current_page && i == reorder_to_index % page_length) {
				// Don't create the property the moving property will take the place of,
				// e.g. (if page_length == 20) don't create element 20 if dragging an item from
				// the first page to the second page because element 20 would become element 19.
				continue;
			}

			HBoxContainer *hbox = memnew(HBoxContainer);
			vbox->add_child(hbox);

			Button *reorder_button = memnew(Button);
			reorder_button->set_icon(get_icon("TripleBar", "EditorIcons"));
			reorder_button->set_default_cursor_shape(Control::CURSOR_MOVE);
			reorder_button->connect("gui_input", this, "_reorder_button_gui_input");
			reorder_button->connect("button_down", this, "_reorder_button_down", varray(i + offset));
			reorder_button->connect("button_up", this, "_reorder_button_up");
			hbox->add_child(reorder_button);

			String prop_name = "indices/" + itos(i + offset);

			EditorProperty *prop = nullptr;
			Variant value = array.get(i + offset);
			Variant::Type value_type = value.get_type();

			if (value_type == Variant::NIL && subtype != Variant::NIL) {
				value_type = subtype;
			}

			if (value_type == Variant::OBJECT && Object::cast_to<EncodedObjectAsID>(value)) {
				EditorPropertyObjectID *editor = memnew(EditorPropertyObjectID);
				editor->setup("Object");
				prop = editor;
			} else {
				prop = EditorInspector::instantiate_property_editor(nullptr, value_type, "", subtype_hint, subtype_hint_string, 0);
			}

			prop->set_object_and_property(object.ptr(), prop_name);
			prop->set_label(itos(i + offset));
			prop->set_selectable(false);
			prop->set_use_folding(is_using_folding());
			prop->connect("property_changed", this, "_property_changed");
			prop->connect("object_id_selected", this, "_object_id_selected");
			prop->set_h_size_flags(SIZE_EXPAND_FILL);
			hbox->add_child(prop);

			bool is_untyped_array = array.get_type() == Variant::ARRAY && subtype == Variant::NIL;

			if (is_untyped_array) {
				Button *edit = memnew(Button);
				edit->set_icon(get_icon("Edit", "EditorIcons"));
				hbox->add_child(edit);
				edit->connect("pressed", this, "_change_type", varray(edit, i + offset));
			} else {
				Button *remove = memnew(Button);
				remove->set_icon(get_icon("Remove", "EditorIcons"));
				remove->connect("pressed", this, "_remove_pressed", varray(i + offset));
				hbox->add_child(remove);
			}

			prop->update_property();
		}

		if (reorder_to_index % page_length > 0) {
			vbox->move_child(vbox->get_child(2), reorder_to_index % page_length + 2);
		}

		updating = false;

	} else {
		if (vbox) {
			set_bottom_editor(nullptr);
			memdelete(vbox);
			vbox = nullptr;
		}
	}
}

void EditorPropertyArray::_remove_pressed(int p_index) {
	Variant array = object->get_array();
	array.call("remove", p_index);

	emit_changed(get_edited_property(), array, "", false);
	update_property();
}

void EditorPropertyArray::_button_draw() {
	if (dropping) {
		Color color = get_color("accent_color", "Editor");
		edit->draw_rect(Rect2(Point2(), edit->get_size()), color, false);
	}
}

bool EditorPropertyArray::_is_drop_valid(const Dictionary &p_drag_data) const {
	String allowed_type = Variant::get_type_name(subtype);

	// When the subtype is of type Object, an additional subtype may be specified in the hint string
	// (e.g. Resource, Texture2D, ShaderMaterial, etc). We want the allowed type to be that, not just "Object".
	if (subtype == Variant::OBJECT && subtype_hint_string != "") {
		allowed_type = subtype_hint_string;
	}

	Dictionary drag_data = p_drag_data;

	if (drag_data.has("type") && String(drag_data["type"]) == "files") {
		Vector<String> files = drag_data["files"];

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			String ftype = EditorFileSystem::get_singleton()->get_file_type(file);

			for (int j = 0; j < allowed_type.get_slice_count(","); j++) {
				String at = allowed_type.get_slice(",", j).strip_edges();
				// Fail if one of the files is not of allowed type
				if (!ClassDB::is_parent_class(ftype, at)) {
					return false;
				}
			}
		}

		// If no files fail, drop is valid
		return true;
	}

	return false;
}

bool EditorPropertyArray::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	return _is_drop_valid(p_data);
}

void EditorPropertyArray::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	ERR_FAIL_COND(!_is_drop_valid(p_data));

	Dictionary drag_data = p_data;

	if (drag_data.has("type") && String(drag_data["type"]) == "files") {
		Vector<String> files = drag_data["files"];

		Variant array = object->get_array();

		// Handle the case where array is not initialised yet.
		if (!array.is_array()) {
			Variant::CallError ce;
			array = Variant::construct(array_type, nullptr, 0, ce);
		}

		// Loop the file array and add to existing array.
		for (int i = 0; i < files.size(); i++) {
			String file = files[i];

			RES res = ResourceLoader::load(file);
			if (res.is_valid()) {
				array.call("push_back", res);
			}
		}

		if (array.get_type() == Variant::ARRAY) {
			array = array.call("duplicate");
		}

		emit_changed(get_edited_property(), array, "", false);
		object->set_array(array);

		update_property();
	}
}

void EditorPropertyArray::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			change_type->clear();
			for (int i = 0; i < Variant::VARIANT_MAX; i++) {
				String type = Variant::get_type_name(Variant::Type(i));
				change_type->add_icon_item(get_icon(type, "EditorIcons"), type, i);
			}
			change_type->add_separator();
			change_type->add_icon_item(get_icon("Remove", "EditorIcons"), TTR("Remove Item"), Variant::VARIANT_MAX);
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			if (is_visible_in_tree()) {
				if (_is_drop_valid(get_viewport()->gui_get_drag_data())) {
					dropping = true;
					edit->update();
				}
			}
		} break;

		case NOTIFICATION_DRAG_END: {
			if (dropping) {
				dropping = false;
				edit->update();
			}
		} break;
	}
}

void EditorPropertyArray::_edit_pressed() {
	Variant array = get_edited_object()->get(get_edited_property());
	if (!array.is_array()) {
		Variant::CallError ce;
		array = Variant::construct(array_type, nullptr, 0, ce);

		get_edited_object()->set(get_edited_property(), array);
	}

	get_edited_object()->editor_set_section_unfold(get_edited_property(), edit->is_pressed());
	update_property();
}

void EditorPropertyArray::_page_changed(double p_page) {
	if (updating) {
		return;
	}
	page_index = p_page;
	update_property();
}

void EditorPropertyArray::_length_changed(double p_page) {
	if (updating) {
		return;
	}

	Variant array = object->get_array();
	int previous_size = array.call("size");

	array.call("resize", int(p_page));

	if (array.get_type() == Variant::ARRAY) {
		if (subtype != Variant::NIL) {
			int size = array.call("size");
			for (int i = previous_size; i < size; i++) {
				if (array.get(i).get_type() == Variant::NIL) {
					Variant::CallError ce;
					array.set(i, Variant::construct(subtype, nullptr, 0, ce));
				}
			}
		}
		array = array.call("duplicate"); // Duplicate, so undo/redo works better.
	} else {
		int size = array.call("size");
		// Pool*Array don't initialize their elements, have to do it manually.
		for (int i = previous_size; i < size; i++) {
			Variant::CallError ce;
			array.set(i, Variant::construct(array.get(i).get_type(), nullptr, 0, ce));
		}
	}

	emit_changed(get_edited_property(), array, "", false);
	object->set_array(array);
	update_property();
}

void EditorPropertyArray::setup(Variant::Type p_array_type, const String &p_hint_string) {
	array_type = p_array_type;

	// The format of p_hint_string is:
	// subType/subTypeHint:nextSubtype ... etc.
	if (array_type == Variant::ARRAY && !p_hint_string.empty()) {
		int hint_subtype_separator = p_hint_string.find(":");
		if (hint_subtype_separator >= 0) {
			String subtype_string = p_hint_string.substr(0, hint_subtype_separator);
			int slash_pos = subtype_string.find("/");
			if (slash_pos >= 0) {
				subtype_hint = PropertyHint(subtype_string.substr(slash_pos + 1, subtype_string.size() - slash_pos - 1).to_int());
				subtype_string = subtype_string.substr(0, slash_pos);
			}

			subtype_hint_string = p_hint_string.substr(hint_subtype_separator + 1, p_hint_string.size() - hint_subtype_separator - 1);
			subtype = Variant::Type(subtype_string.to_int());
		}
	}
}

void EditorPropertyArray::_reorder_button_gui_input(const Ref<InputEvent> &p_event) {
	if (reorder_from_index < 0) {
		return;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		Variant array = object->get_array();
		int size = array.call("size");

		// Cumulate the mouse delta, many small changes (dragging slowly) should result in reordering at some point.
		reorder_mouse_y_delta += mm->get_relative().y;

		// Reordering is done by moving the dragged element by +1/-1 index at a time based on the cumulated mouse delta so if
		// already at the array bounds make sure to ignore the remaining out of bounds drag (by resetting the cumulated delta).
		if ((reorder_to_index == 0 && reorder_mouse_y_delta < 0.0f) || (reorder_to_index == size - 1 && reorder_mouse_y_delta > 0.0f)) {
			reorder_mouse_y_delta = 0.0f;
			return;
		}

		float required_y_distance = 20.0f * EDSCALE;
		if (ABS(reorder_mouse_y_delta) > required_y_distance) {
			int direction = reorder_mouse_y_delta > 0.0f ? 1 : -1;
			reorder_mouse_y_delta -= required_y_distance * direction;

			reorder_to_index += direction;
			if ((direction < 0 && reorder_to_index % page_length == page_length - 1) || (direction > 0 && reorder_to_index % page_length == 0)) {
				// Automatically move to the next/previous page.
				page_slider->set_value(page_index + direction);
			}
			vbox->move_child(reorder_selected_element_hbox, reorder_to_index % page_length + 2);
			// Ensure the moving element is visible.
			EditorNode::get_singleton()->get_inspector()->ensure_control_visible(reorder_selected_element_hbox);
		}
	}
}

void EditorPropertyArray::_reorder_button_down(int p_index) {
	reorder_from_index = p_index;
	reorder_to_index = p_index;
	reorder_selected_element_hbox = Object::cast_to<HBoxContainer>(vbox->get_child(p_index % page_length + 2));
	reorder_selected_button = Object::cast_to<Button>(reorder_selected_element_hbox->get_child(0));
	// Ideally it'd to be able to show the mouse but I had issues with
	// Control's `mouse_exit()`/`mouse_entered()` signals not getting called.
	Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
}

void EditorPropertyArray::_reorder_button_up() {
	if (reorder_from_index != reorder_to_index) {
		// Move the element.
		Variant array = object->get_array();

		Variant value_to_move = array.get(reorder_from_index);
		array.call("remove", reorder_from_index);
		array.call("insert", reorder_to_index, value_to_move);

		emit_changed(get_edited_property(), array, "", false);
		object->set_array(array);
		update_property();
	}

	reorder_from_index = -1;
	reorder_to_index = -1;
	reorder_mouse_y_delta = 0.0f;

	Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);
	reorder_selected_button->warp_mouse(reorder_selected_button->get_size() / 2.0f);

	reorder_selected_element_hbox = nullptr;
	reorder_selected_button = nullptr;
}

void EditorPropertyArray::_bind_methods() {
	ClassDB::bind_method("_edit_pressed", &EditorPropertyArray::_edit_pressed);
	ClassDB::bind_method("_page_changed", &EditorPropertyArray::_page_changed);
	ClassDB::bind_method("_length_changed", &EditorPropertyArray::_length_changed);
	ClassDB::bind_method("_property_changed", &EditorPropertyArray::_property_changed, DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method("_change_type", &EditorPropertyArray::_change_type);
	ClassDB::bind_method("_change_type_menu", &EditorPropertyArray::_change_type_menu);
	ClassDB::bind_method("_object_id_selected", &EditorPropertyArray::_object_id_selected);
	ClassDB::bind_method("_remove_pressed", &EditorPropertyArray::_remove_pressed);
	ClassDB::bind_method("_reorder_button_gui_input", &EditorPropertyArray::_reorder_button_gui_input);
	ClassDB::bind_method("_reorder_button_down", &EditorPropertyArray::_reorder_button_down);
	ClassDB::bind_method("_reorder_button_up", &EditorPropertyArray::_reorder_button_up);
	ClassDB::bind_method("_button_draw", &EditorPropertyArray::_button_draw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &EditorPropertyArray::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &EditorPropertyArray::drop_data_fw);
}

EditorPropertyArray::EditorPropertyArray() {
	object.instance();
	page_length = int(EDITOR_GET("interface/inspector/max_array_dictionary_items_per_page"));
	edit = memnew(Button);
	edit->set_flat(true);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);
	edit->set_clip_text(true);
	edit->connect("pressed", this, "_edit_pressed");
	edit->set_toggle_mode(true);
	edit->set_drag_forwarding(this);
	edit->connect("draw", this, "_button_draw");
	add_child(edit);
	add_focusable(edit);
	vbox = nullptr;
	page_slider = nullptr;
	size_slider = nullptr;
	updating = false;
	change_type = memnew(PopupMenu);
	add_child(change_type);
	change_type->connect("id_pressed", this, "_change_type_menu");

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		String type = Variant::get_type_name(Variant::Type(i));
		change_type->add_item(type, i);
	}
	change_type->add_separator();
	change_type->add_item(TTR("Remove Item"), Variant::VARIANT_MAX);
	changing_type_index = -1;

	subtype = Variant::NIL;
	subtype_hint = PROPERTY_HINT_NONE;
	subtype_hint_string = "";

	dropping = false;
}

///////////////////// DICTIONARY ///////////////////////////

void EditorPropertyDictionary::_property_changed(const String &p_property, Variant p_value, const String &p_name, bool changing) {
	if (p_property == "new_item_key") {
		object->set_new_item_key(p_value);
	} else if (p_property == "new_item_value") {
		object->set_new_item_value(p_value);
	} else if (p_property.begins_with("indices")) {
		int index = p_property.get_slice("/", 1).to_int();
		Dictionary dict = object->get_dict();
		Variant key = dict.get_key_at_index(index);
		dict[key] = p_value;

		emit_changed(get_edited_property(), dict, "", true);

		dict = dict.duplicate(); // Duplicate, so undo/redo works better\.
		object->set_dict(dict);
	}
}

void EditorPropertyDictionary::_change_type(Object *p_button, int p_index) {
	Button *button = Object::cast_to<Button>(p_button);

	Rect2 rect = button->get_global_rect();
	change_type->set_as_minsize();
	change_type->set_global_position(rect.position + rect.size - Vector2(change_type->get_combined_minimum_size().x, 0));
	change_type->popup();
	changing_type_index = p_index;
}

void EditorPropertyDictionary::_add_key_value() {
	// Do not allow nil as valid key. I experienced errors with this
	if (object->get_new_item_key().get_type() == Variant::NIL) {
		return;
	}

	Dictionary dict = object->get_dict();

	dict[object->get_new_item_key()] = object->get_new_item_value();
	object->set_new_item_key(Variant());
	object->set_new_item_value(Variant());

	emit_changed(get_edited_property(), dict, "", false);

	dict = dict.duplicate(); // Duplicate, so undo/redo works better.
	object->set_dict(dict);
	update_property();
}

void EditorPropertyDictionary::_change_type_menu(int p_index) {
	if (changing_type_index < 0) {
		Variant value;
		Variant::CallError ce;
		value = Variant::construct(Variant::Type(p_index), nullptr, 0, ce);
		if (changing_type_index == -1) {
			object->set_new_item_key(value);
		} else {
			object->set_new_item_value(value);
		}
		update_property();
		return;
	}

	Dictionary dict = object->get_dict();

	if (p_index < Variant::VARIANT_MAX) {
		Variant value;
		Variant::CallError ce;
		value = Variant::construct(Variant::Type(p_index), nullptr, 0, ce);
		Variant key = dict.get_key_at_index(changing_type_index);
		dict[key] = value;
	} else {
		Variant key = dict.get_key_at_index(changing_type_index);
		dict.erase(key);
	}

	emit_changed(get_edited_property(), dict, "", false);

	dict = dict.duplicate(); // Duplicate, so undo/redo works better.
	object->set_dict(dict);
	update_property();
}

void EditorPropertyDictionary::update_property() {
	Variant updated_val = get_edited_object()->get(get_edited_property());

	if (updated_val.get_type() == Variant::NIL) {
		edit->set_text("Dictionary (Nil)"); // This provides symmetry with the array property.
		edit->set_pressed(false);
		if (vbox) {
			set_bottom_editor(nullptr);
			memdelete(vbox);
			button_add_item = nullptr;
			vbox = nullptr;
		}
		return;
	}

	Dictionary dict = updated_val;

	edit->set_text("Dictionary (size " + itos(dict.size()) + ")");

	bool unfolded = get_edited_object()->editor_is_section_unfolded(get_edited_property());
	if (edit->is_pressed() != unfolded) {
		edit->set_pressed(unfolded);
	}

	if (unfolded) {
		updating = true;

		if (!vbox) {
			vbox = memnew(VBoxContainer);
			add_child(vbox);
			set_bottom_editor(vbox);

			page_hbox = memnew(HBoxContainer);
			vbox->add_child(page_hbox);
			Label *label = memnew(Label(TTR("Page:")));
			label->set_h_size_flags(SIZE_EXPAND_FILL);
			page_hbox->add_child(label);
			page_slider = memnew(EditorSpinSlider);
			page_slider->set_step(1);
			page_hbox->add_child(page_slider);
			page_slider->set_h_size_flags(SIZE_EXPAND_FILL);
			page_slider->connect("value_changed", this, "_page_changed");
		} else {
			// Queue children for deletion, deleting immediately might cause errors.
			for (int i = 1; i < vbox->get_child_count(); i++) {
				vbox->get_child(i)->queue_delete();
			}
		}

		int size = dict.size();

		int pages = MAX(0, size - 1) / page_length + 1;

		page_slider->set_max(pages);
		page_index = MIN(page_index, pages - 1);
		page_slider->set_value(page_index);
		page_hbox->set_visible(pages > 1);

		int offset = page_index * page_length;

		int amount = MIN(size - offset, page_length);

		dict = dict.duplicate();

		object->set_dict(dict);
		VBoxContainer *add_vbox = nullptr;

		for (int i = 0; i < amount + 2; i++) {
			String prop_name;
			Variant key;
			Variant value;

			if (i < amount) {
				prop_name = "indices/" + itos(i + offset);
				key = dict.get_key_at_index(i + offset);
				value = dict.get_value_at_index(i + offset);
			} else if (i == amount) {
				prop_name = "new_item_key";
				value = object->get_new_item_key();
			} else if (i == amount + 1) {
				prop_name = "new_item_value";
				value = object->get_new_item_value();
			}

			EditorProperty *prop = nullptr;

			switch (value.get_type()) {
				case Variant::NIL: {
					prop = memnew(EditorPropertyNil);

				} break;

				// Atomic types.
				case Variant::BOOL: {
					prop = memnew(EditorPropertyCheck);

				} break;
				case Variant::INT: {
					EditorPropertyInteger *editor = memnew(EditorPropertyInteger);
					editor->setup(-100000, 100000, 1, true, true);
					prop = editor;

				} break;
				case Variant::REAL: {
					EditorPropertyFloat *editor = memnew(EditorPropertyFloat);
					editor->setup(-100000, 100000, 0.001, true, false, true, true);
					prop = editor;
				} break;
				case Variant::STRING: {
					prop = memnew(EditorPropertyText);

				} break;

				// Math types.
				case Variant::VECTOR2: {
					EditorPropertyVector2 *editor = memnew(EditorPropertyVector2);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::RECT2: {
					EditorPropertyRect2 *editor = memnew(EditorPropertyRect2);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::VECTOR3: {
					EditorPropertyVector3 *editor = memnew(EditorPropertyVector3);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::TRANSFORM2D: {
					EditorPropertyTransform2D *editor = memnew(EditorPropertyTransform2D);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::PLANE: {
					EditorPropertyPlane *editor = memnew(EditorPropertyPlane);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::QUAT: {
					EditorPropertyQuat *editor = memnew(EditorPropertyQuat);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::AABB: {
					EditorPropertyAABB *editor = memnew(EditorPropertyAABB);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::BASIS: {
					EditorPropertyBasis *editor = memnew(EditorPropertyBasis);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;
				case Variant::TRANSFORM: {
					EditorPropertyTransform *editor = memnew(EditorPropertyTransform);
					editor->setup(-100000, 100000, 0.001, true);
					prop = editor;

				} break;

				// Miscellaneous types.
				case Variant::COLOR: {
					prop = memnew(EditorPropertyColor);

				} break;
				case Variant::NODE_PATH: {
					prop = memnew(EditorPropertyNodePath);

				} break;
				case Variant::_RID: {
					prop = memnew(EditorPropertyRID);

				} break;
				case Variant::OBJECT: {
					if (Object::cast_to<EncodedObjectAsID>(value)) {
						EditorPropertyObjectID *editor = memnew(EditorPropertyObjectID);
						editor->setup("Object");
						prop = editor;

					} else {
						EditorPropertyResource *editor = memnew(EditorPropertyResource);
						editor->setup(object.ptr(), prop_name, "Resource");
						editor->set_use_folding(is_using_folding());
						prop = editor;
					}

				} break;
				case Variant::DICTIONARY: {
					prop = memnew(EditorPropertyDictionary);

				} break;
				case Variant::ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::ARRAY);
					prop = editor;
				} break;

				// Arrays.
				case Variant::POOL_BYTE_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_BYTE_ARRAY);
					prop = editor;
				} break;
				case Variant::POOL_INT_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_INT_ARRAY);
					prop = editor;
				} break;
				case Variant::POOL_REAL_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_REAL_ARRAY);
					prop = editor;
				} break;
				case Variant::POOL_STRING_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_STRING_ARRAY);
					prop = editor;
				} break;
				case Variant::POOL_VECTOR2_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_VECTOR2_ARRAY);
					prop = editor;
				} break;
				case Variant::POOL_VECTOR3_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_VECTOR3_ARRAY);
					prop = editor;
				} break;
				case Variant::POOL_COLOR_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::POOL_COLOR_ARRAY);
					prop = editor;
				} break;
				default: {
				}
			}

			if (i == amount) {
				PanelContainer *pc = memnew(PanelContainer);
				vbox->add_child(pc);
				Ref<StyleBoxFlat> flat;
				flat.instance();
				for (int j = 0; j < 4; j++) {
					flat->set_default_margin(Margin(j), 2 * EDSCALE);
				}
				flat->set_bg_color(get_color("prop_subsection", "Editor"));

				pc->add_style_override("panel", flat);
				add_vbox = memnew(VBoxContainer);
				pc->add_child(add_vbox);
			}
			prop->set_object_and_property(object.ptr(), prop_name);
			int change_index = 0;

			if (i < amount) {
				String cs = key.get_construct_string();
				prop->set_label(key.get_construct_string());
				prop->set_tooltip(cs);
				change_index = i + offset;
			} else if (i == amount) {
				prop->set_label(TTR("New Key:"));
				change_index = -1;
			} else if (i == amount + 1) {
				prop->set_label(TTR("New Value:"));
				change_index = -2;
			}

			prop->set_selectable(false);
			prop->connect("property_changed", this, "_property_changed");
			prop->connect("object_id_selected", this, "_object_id_selected");

			HBoxContainer *hbox = memnew(HBoxContainer);
			if (add_vbox) {
				add_vbox->add_child(hbox);
			} else {
				vbox->add_child(hbox);
			}
			hbox->add_child(prop);
			prop->set_h_size_flags(SIZE_EXPAND_FILL);
			Button *edit = memnew(Button);
			edit->set_icon(get_icon("Edit", "EditorIcons"));
			hbox->add_child(edit);
			edit->connect("pressed", this, "_change_type", varray(edit, change_index));

			prop->update_property();

			if (i == amount + 1) {
				button_add_item = memnew(Button);
				button_add_item->set_text(TTR("Add Key/Value Pair"));
				button_add_item->set_icon(get_icon("Add", "EditorIcons"));
				button_add_item->connect("pressed", this, "_add_key_value");
				add_vbox->add_child(button_add_item);
			}
		}

		updating = false;

	} else {
		if (vbox) {
			set_bottom_editor(nullptr);
			memdelete(vbox);
			button_add_item = nullptr;
			vbox = nullptr;
		}
	}
}

void EditorPropertyDictionary::_object_id_selected(const String &p_property, ObjectID p_id) {
	emit_signal("object_id_selected", p_property, p_id);
}

void EditorPropertyDictionary::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			change_type->clear();
			for (int i = 0; i < Variant::VARIANT_MAX; i++) {
				String type = Variant::get_type_name(Variant::Type(i));
				change_type->add_icon_item(get_icon(type, "EditorIcons"), type, i);
			}
			change_type->add_separator();
			change_type->add_icon_item(get_icon("Remove", "EditorIcons"), TTR("Remove Item"), Variant::VARIANT_MAX);

			if (Object::cast_to<Button>(button_add_item)) {
				button_add_item->set_icon(get_icon("Add", "EditorIcons"));
			}
		} break;
	}
}

void EditorPropertyDictionary::_edit_pressed() {
	Variant prop_val = get_edited_object()->get(get_edited_property());
	if (prop_val.get_type() == Variant::NIL) {
		Variant::CallError ce;
		prop_val = Variant::construct(Variant::DICTIONARY, nullptr, 0, ce);
		get_edited_object()->set(get_edited_property(), prop_val);
	}

	get_edited_object()->editor_set_section_unfold(get_edited_property(), edit->is_pressed());
	update_property();
}

void EditorPropertyDictionary::_page_changed(double p_page) {
	if (updating) {
		return;
	}
	page_index = p_page;
	update_property();
}

void EditorPropertyDictionary::_bind_methods() {
	ClassDB::bind_method("_edit_pressed", &EditorPropertyDictionary::_edit_pressed);
	ClassDB::bind_method("_page_changed", &EditorPropertyDictionary::_page_changed);
	ClassDB::bind_method("_property_changed", &EditorPropertyDictionary::_property_changed, DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method("_change_type", &EditorPropertyDictionary::_change_type);
	ClassDB::bind_method("_change_type_menu", &EditorPropertyDictionary::_change_type_menu);
	ClassDB::bind_method("_add_key_value", &EditorPropertyDictionary::_add_key_value);
	ClassDB::bind_method("_object_id_selected", &EditorPropertyDictionary::_object_id_selected);
}

EditorPropertyDictionary::EditorPropertyDictionary() {
	object.instance();
	page_length = int(EDITOR_GET("interface/inspector/max_array_dictionary_items_per_page"));
	edit = memnew(Button);
	edit->set_flat(true);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);
	edit->set_clip_text(true);
	edit->connect("pressed", this, "_edit_pressed");
	edit->set_toggle_mode(true);
	add_child(edit);
	add_focusable(edit);
	vbox = nullptr;
	page_slider = nullptr;
	button_add_item = nullptr;
	updating = false;
	change_type = memnew(PopupMenu);
	add_child(change_type);
	change_type->connect("id_pressed", this, "_change_type_menu");
	changing_type_index = -1;
}
