/**************************************************************************/
/*  editor_properties_array_dict.cpp                                      */
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

#include "editor_properties_array_dict.h"

#include "core/input/input.h"
#include "core/io/marshalls.h"
#include "editor/editor_properties.h"
#include "editor/editor_properties_vector.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_spin_slider.h"
#include "editor/inspector_dock.h"
#include "scene/gui/button.h"
#include "scene/resources/packed_scene.h"

bool EditorPropertyArrayObject::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	if (!name.begins_with("indices")) {
		return false;
	}

	int index;
	if (name.begins_with("metadata/")) {
		index = name.get_slice("/", 2).to_int();
	} else {
		index = name.get_slice("/", 1).to_int();
	}

	array.set(index, p_value);
	return true;
}

bool EditorPropertyArrayObject::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	if (!name.begins_with("indices")) {
		return false;
	}

	int index;
	if (name.begins_with("metadata/")) {
		index = name.get_slice("/", 2).to_int();
	} else {
		index = name.get_slice("/", 1).to_int();
	}

	bool valid;
	r_ret = array.get(index, &valid);

	if (r_ret.get_type() == Variant::OBJECT && Object::cast_to<EncodedObjectAsID>(r_ret)) {
		r_ret = Object::cast_to<EncodedObjectAsID>(r_ret)->get_object_id();
	}

	return valid;
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

void EditorPropertyArray::initialize_array(Variant &p_array) {
	if (array_type == Variant::ARRAY && subtype != Variant::NIL) {
		Array array;
		StringName subtype_class;
		Ref<Script> subtype_script;
		if (subtype == Variant::OBJECT && !subtype_hint_string.is_empty()) {
			if (ClassDB::class_exists(subtype_hint_string)) {
				subtype_class = subtype_hint_string;
			}
		}
		array.set_typed(subtype, subtype_class, subtype_script);
		p_array = array;
	} else {
		VariantInternal::initialize(&p_array, array_type);
	}
}

void EditorPropertyArray::_property_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	if (!p_property.begins_with("indices")) {
		return;
	}

	if (p_value.get_type() == Variant::OBJECT && p_value.is_null()) {
		p_value = Variant(); // `EditorResourcePicker` resets to `Ref<Resource>()`. See GH-82716.
	}

	int index;
	if (p_property.begins_with("metadata/")) {
		index = p_property.get_slice("/", 2).to_int();
	} else {
		index = p_property.get_slice("/", 1).to_int();
	}

	Variant array = object->get_array().duplicate();
	array.set(index, p_value);
	object->set_array(array);
	emit_changed(get_edited_property(), array, "", true);
}

void EditorPropertyArray::_change_type(Object *p_button, int p_index) {
	Button *button = Object::cast_to<Button>(p_button);
	changing_type_index = p_index;
	Rect2 rect = button->get_screen_rect();
	change_type->reset_size();
	change_type->set_position(rect.get_end() - Vector2(change_type->get_contents_minimum_size().x, 0));
	change_type->popup();
}

void EditorPropertyArray::_change_type_menu(int p_index) {
	if (p_index == Variant::VARIANT_MAX) {
		_remove_pressed(changing_type_index);
		return;
	}

	Variant value;
	VariantInternal::initialize(&value, Variant::Type(p_index));

	Variant array = object->get_array().duplicate();
	array.set(changing_type_index, value);

	emit_changed(get_edited_property(), array, "", true);
	update_property();
}

void EditorPropertyArray::_object_id_selected(const StringName &p_property, ObjectID p_id) {
	emit_signal(SNAME("object_id_selected"), p_property, p_id);
}

void EditorPropertyArray::update_property() {
	Variant array = get_edited_property_value();

	String array_type_name = Variant::get_type_name(array_type);
	if (array_type == Variant::ARRAY && subtype != Variant::NIL) {
		String type_name;
		if (subtype == Variant::OBJECT && (subtype_hint == PROPERTY_HINT_RESOURCE_TYPE || subtype_hint == PROPERTY_HINT_NODE_TYPE)) {
			type_name = subtype_hint_string;
		} else {
			type_name = Variant::get_type_name(subtype);
		}

		array_type_name = vformat("%s[%s]", array_type_name, type_name);
	}

	if (!array.is_array()) {
		edit->set_text(vformat(TTR("(Nil) %s"), array_type_name));
		edit->set_pressed(false);
		if (container) {
			set_bottom_editor(nullptr);
			memdelete(container);
			button_add_item = nullptr;
			container = nullptr;
		}
		return;
	}

	object->set_array(array);

	int size = array.call("size");
	int max_page = MAX(0, size - 1) / page_length;
	page_index = MIN(page_index, max_page);
	int offset = page_index * page_length;

	edit->set_text(vformat(TTR("%s (size %s)"), array_type_name, itos(size)));

	bool unfolded = get_edited_object()->editor_is_section_unfolded(get_edited_property());
	if (edit->is_pressed() != unfolded) {
		edit->set_pressed(unfolded);
	}

	if (unfolded) {
		updating = true;

		if (!container) {
			container = memnew(MarginContainer);
			container->set_theme_type_variation("MarginContainer4px");
			container->set_mouse_filter(MOUSE_FILTER_STOP);
			add_child(container);
			set_bottom_editor(container);

			VBoxContainer *vbox = memnew(VBoxContainer);
			container->add_child(vbox);

			HBoxContainer *hbox = memnew(HBoxContainer);
			vbox->add_child(hbox);

			Label *size_label = memnew(Label(TTR("Size:")));
			size_label->set_h_size_flags(SIZE_EXPAND_FILL);
			hbox->add_child(size_label);

			size_slider = memnew(EditorSpinSlider);
			size_slider->set_step(1);
			size_slider->set_max(INT32_MAX);
			size_slider->set_h_size_flags(SIZE_EXPAND_FILL);
			size_slider->set_read_only(is_read_only());
			size_slider->connect("value_changed", callable_mp(this, &EditorPropertyArray::_length_changed));
			hbox->add_child(size_slider);

			property_vbox = memnew(VBoxContainer);
			property_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
			vbox->add_child(property_vbox);

			button_add_item = EditorInspector::create_inspector_action_button(TTR("Add Element"));
			button_add_item->set_icon(get_editor_theme_icon(SNAME("Add")));
			button_add_item->connect(SNAME("pressed"), callable_mp(this, &EditorPropertyArray::_add_element));
			button_add_item->set_disabled(is_read_only());
			vbox->add_child(button_add_item);

			paginator = memnew(EditorPaginator);
			paginator->connect("page_changed", callable_mp(this, &EditorPropertyArray::_page_changed));
			vbox->add_child(paginator);
		} else {
			// Bye bye children of the box.
			for (int i = property_vbox->get_child_count() - 1; i >= 0; i--) {
				Node *child = property_vbox->get_child(i);
				if (child == reorder_selected_element_hbox) {
					continue; // Don't remove the property that the user is moving.
				}

				child->queue_free(); // Button still needed after pressed is called.
				property_vbox->remove_child(child);
			}
		}

		size_slider->set_value(size);
		property_vbox->set_visible(size > 0);
		button_add_item->set_visible(page_index == max_page);
		paginator->update(page_index, max_page);
		paginator->set_visible(max_page > 0);

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
			property_vbox->add_child(hbox);

			Button *reorder_button = memnew(Button);
			reorder_button->set_icon(get_editor_theme_icon(SNAME("TripleBar")));
			reorder_button->set_default_cursor_shape(Control::CURSOR_MOVE);
			reorder_button->set_disabled(is_read_only());
			reorder_button->connect("gui_input", callable_mp(this, &EditorPropertyArray::_reorder_button_gui_input));
			reorder_button->connect("button_down", callable_mp(this, &EditorPropertyArray::_reorder_button_down).bind(i + offset));
			reorder_button->connect("button_up", callable_mp(this, &EditorPropertyArray::_reorder_button_up));
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
				prop = EditorInspector::instantiate_property_editor(nullptr, value_type, "", subtype_hint, subtype_hint_string, PROPERTY_USAGE_NONE);
			}

			prop->set_object_and_property(object.ptr(), prop_name);
			prop->set_label(itos(i + offset));
			prop->set_selectable(false);
			prop->set_use_folding(is_using_folding());
			prop->connect("property_changed", callable_mp(this, &EditorPropertyArray::_property_changed));
			prop->connect("object_id_selected", callable_mp(this, &EditorPropertyArray::_object_id_selected));
			prop->set_h_size_flags(SIZE_EXPAND_FILL);
			prop->set_read_only(is_read_only());
			hbox->add_child(prop);

			bool is_untyped_array = array.get_type() == Variant::ARRAY && subtype == Variant::NIL;

			if (is_untyped_array) {
				Button *edit_btn = memnew(Button);
				edit_btn->set_icon(get_editor_theme_icon(SNAME("Edit")));
				hbox->add_child(edit_btn);
				edit_btn->set_disabled(is_read_only());
				edit_btn->connect("pressed", callable_mp(this, &EditorPropertyArray::_change_type).bind(edit_btn, i + offset));
			} else {
				Button *remove_btn = memnew(Button);
				remove_btn->set_icon(get_editor_theme_icon(SNAME("Remove")));
				remove_btn->set_disabled(is_read_only());
				remove_btn->connect("pressed", callable_mp(this, &EditorPropertyArray::_remove_pressed).bind(i + offset));
				hbox->add_child(remove_btn);
			}

			prop->update_property();
		}

		if (reorder_to_index % page_length > 0) {
			property_vbox->move_child(property_vbox->get_child(0), reorder_to_index % page_length);
		}

		updating = false;

	} else {
		if (container) {
			set_bottom_editor(nullptr);
			memdelete(container);
			button_add_item = nullptr;
			container = nullptr;
		}
	}
}

void EditorPropertyArray::_remove_pressed(int p_index) {
	Variant array = object->get_array().duplicate();
	array.call("remove_at", p_index);

	emit_changed(get_edited_property(), array, "", false);
	update_property();
}

void EditorPropertyArray::_button_draw() {
	if (dropping) {
		Color color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		edit->draw_rect(Rect2(Point2(), edit->get_size()), color, false);
	}
}

bool EditorPropertyArray::_is_drop_valid(const Dictionary &p_drag_data) const {
	if (is_read_only()) {
		return false;
	}

	String allowed_type = Variant::get_type_name(subtype);

	// When the subtype is of type Object, an additional subtype may be specified in the hint string
	// (e.g. Resource, Texture2D, ShaderMaterial, etc). We want the allowed type to be that, not just "Object".
	if (subtype == Variant::OBJECT && !subtype_hint_string.is_empty()) {
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
				// Fail if one of the files is not of allowed type.
				if (!ClassDB::is_parent_class(ftype, at)) {
					return false;
				}
			}
		}

		// If no files fail, drop is valid.
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

		// Handle the case where array is not initialized yet.
		if (!array.is_array()) {
			initialize_array(array);
		} else {
			array = array.duplicate();
		}

		// Loop the file array and add to existing array.
		for (int i = 0; i < files.size(); i++) {
			String file = files[i];

			Ref<Resource> res = ResourceLoader::load(file);
			if (res.is_valid()) {
				array.call("push_back", res);
			}
		}

		emit_changed(get_edited_property(), array, "", false);
		update_property();
	}
}

void EditorPropertyArray::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			change_type->clear();
			for (int i = 0; i < Variant::VARIANT_MAX; i++) {
				if (i == Variant::CALLABLE || i == Variant::SIGNAL || i == Variant::RID) {
					// These types can't be constructed or serialized properly, so skip them.
					continue;
				}

				String type = Variant::get_type_name(Variant::Type(i));
				change_type->add_icon_item(get_editor_theme_icon(type), type, i);
			}
			change_type->add_separator();
			change_type->add_icon_item(get_editor_theme_icon(SNAME("Remove")), TTR("Remove Item"), Variant::VARIANT_MAX);

			if (button_add_item) {
				button_add_item->set_icon(get_editor_theme_icon(SNAME("Add")));
			}
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			if (is_visible_in_tree()) {
				if (_is_drop_valid(get_viewport()->gui_get_drag_data())) {
					dropping = true;
					edit->queue_redraw();
				}
			}
		} break;

		case NOTIFICATION_DRAG_END: {
			if (dropping) {
				dropping = false;
				edit->queue_redraw();
			}
		} break;
	}
}

void EditorPropertyArray::_edit_pressed() {
	Variant array = get_edited_property_value();
	if (!array.is_array() && edit->is_pressed()) {
		initialize_array(array);
		get_edited_object()->set(get_edited_property(), array);
	}

	get_edited_object()->editor_set_section_unfold(get_edited_property(), edit->is_pressed());
	update_property();
}

void EditorPropertyArray::_page_changed(int p_page) {
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

	Variant array = object->get_array().duplicate();
	array.call("resize", int(p_page));

	emit_changed(get_edited_property(), array, "", false);
	update_property();
}

void EditorPropertyArray::_add_element() {
	_length_changed(double(object->get_array().call("size")) + 1.0);
}

void EditorPropertyArray::setup(Variant::Type p_array_type, const String &p_hint_string) {
	array_type = p_array_type;

	// The format of p_hint_string is:
	// subType/subTypeHint:nextSubtype ... etc.
	if (!p_hint_string.is_empty()) {
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
	if (reorder_from_index < 0 || is_read_only()) {
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
				_page_changed(page_index + direction);
			}
			property_vbox->move_child(reorder_selected_element_hbox, reorder_to_index % page_length);
			// Ensure the moving element is visible.
			InspectorDock::get_inspector_singleton()->ensure_control_visible(reorder_selected_element_hbox);
		}
	}
}

void EditorPropertyArray::_reorder_button_down(int p_index) {
	if (is_read_only()) {
		return;
	}

	reorder_from_index = p_index;
	reorder_to_index = p_index;
	reorder_selected_element_hbox = Object::cast_to<HBoxContainer>(property_vbox->get_child(p_index % page_length));
	reorder_selected_button = Object::cast_to<Button>(reorder_selected_element_hbox->get_child(0));
	// Ideally it'd to be able to show the mouse but I had issues with
	// Control's `mouse_exit()`/`mouse_entered()` signals not getting called.
	Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);
}

void EditorPropertyArray::_reorder_button_up() {
	if (is_read_only()) {
		return;
	}

	if (reorder_from_index != reorder_to_index) {
		// Move the element.
		Variant array = object->get_array().duplicate();

		Variant value_to_move = array.get(reorder_from_index);
		array.call("remove_at", reorder_from_index);
		array.call("insert", reorder_to_index, value_to_move);

		emit_changed(get_edited_property(), array, "", false);
		update_property();
	}

	reorder_from_index = -1;
	reorder_to_index = -1;
	reorder_mouse_y_delta = 0.0f;

	Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);

	ERR_FAIL_NULL(reorder_selected_button);
	reorder_selected_button->warp_mouse(reorder_selected_button->get_size() / 2.0f);

	reorder_selected_element_hbox = nullptr;
	reorder_selected_button = nullptr;
}

void EditorPropertyArray::_bind_methods() {
}

EditorPropertyArray::EditorPropertyArray() {
	object.instantiate();
	page_length = int(EDITOR_GET("interface/inspector/max_array_dictionary_items_per_page"));

	edit = memnew(Button);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);
	edit->set_clip_text(true);
	edit->connect("pressed", callable_mp(this, &EditorPropertyArray::_edit_pressed));
	edit->set_toggle_mode(true);
	SET_DRAG_FORWARDING_CD(edit, EditorPropertyArray);
	edit->connect("draw", callable_mp(this, &EditorPropertyArray::_button_draw));
	add_child(edit);
	add_focusable(edit);

	change_type = memnew(PopupMenu);
	add_child(change_type);
	change_type->connect("id_pressed", callable_mp(this, &EditorPropertyArray::_change_type_menu));
	changing_type_index = -1;

	subtype = Variant::NIL;
	subtype_hint = PROPERTY_HINT_NONE;
	subtype_hint_string = "";
}

///////////////////// DICTIONARY ///////////////////////////

void EditorPropertyDictionary::_property_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	if (p_value.get_type() == Variant::OBJECT && p_value.is_null()) {
		p_value = Variant(); // `EditorResourcePicker` resets to `Ref<Resource>()`. See GH-82716.
	}

	if (p_property == "new_item_key") {
		object->set_new_item_key(p_value);
	} else if (p_property == "new_item_value") {
		object->set_new_item_value(p_value);
	} else if (p_property.begins_with("indices")) {
		int index = p_property.get_slice("/", 1).to_int();

		Dictionary dict = object->get_dict().duplicate();
		Variant key = dict.get_key_at_index(index);
		dict[key] = p_value;

		object->set_dict(dict);
		emit_changed(get_edited_property(), dict, "", true);
	}
}

void EditorPropertyDictionary::_change_type(Object *p_button, int p_index) {
	Button *button = Object::cast_to<Button>(p_button);

	Rect2 rect = button->get_screen_rect();
	change_type->set_item_disabled(change_type->get_item_index(Variant::VARIANT_MAX), p_index < 0);
	change_type->reset_size();
	change_type->set_position(rect.get_end() - Vector2(change_type->get_contents_minimum_size().x, 0));
	change_type->popup();
	changing_type_index = p_index;
}

void EditorPropertyDictionary::_add_key_value() {
	// Do not allow nil as valid key. I experienced errors with this
	if (object->get_new_item_key().get_type() == Variant::NIL) {
		return;
	}

	Dictionary dict = object->get_dict().duplicate();
	dict[object->get_new_item_key()] = object->get_new_item_value();
	object->set_new_item_key(Variant());
	object->set_new_item_value(Variant());

	emit_changed(get_edited_property(), dict, "", false);
	update_property();
}

void EditorPropertyDictionary::_change_type_menu(int p_index) {
	if (changing_type_index < 0) {
		Variant value;
		VariantInternal::initialize(&value, Variant::Type(p_index));
		if (changing_type_index == -1) {
			object->set_new_item_key(value);
		} else {
			object->set_new_item_value(value);
		}
		update_property();
		return;
	}

	Dictionary dict = object->get_dict().duplicate();
	if (p_index < Variant::VARIANT_MAX) {
		Variant value;
		VariantInternal::initialize(&value, Variant::Type(p_index));
		Variant key = dict.get_key_at_index(changing_type_index);
		dict[key] = value;
	} else {
		Variant key = dict.get_key_at_index(changing_type_index);
		dict.erase(key);
	}

	emit_changed(get_edited_property(), dict, "", false);
	update_property();
}

void EditorPropertyDictionary::setup(PropertyHint p_hint) {
	property_hint = p_hint;
}

void EditorPropertyDictionary::update_property() {
	Variant updated_val = get_edited_property_value();

	if (updated_val.get_type() == Variant::NIL) {
		edit->set_text(TTR("Dictionary (Nil)")); // This provides symmetry with the array property.
		edit->set_pressed(false);
		if (container) {
			set_bottom_editor(nullptr);
			memdelete(container);
			button_add_item = nullptr;
			container = nullptr;
		}
		return;
	}

	Dictionary dict = updated_val;
	object->set_dict(updated_val);

	edit->set_text(vformat(TTR("Dictionary (size %d)"), dict.size()));

	bool unfolded = get_edited_object()->editor_is_section_unfolded(get_edited_property());
	if (edit->is_pressed() != unfolded) {
		edit->set_pressed(unfolded);
	}

	if (unfolded) {
		updating = true;

		if (!container) {
			container = memnew(MarginContainer);
			container->set_theme_type_variation("MarginContainer4px");
			add_child(container);
			set_bottom_editor(container);

			VBoxContainer *vbox = memnew(VBoxContainer);
			container->add_child(vbox);

			property_vbox = memnew(VBoxContainer);
			property_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
			vbox->add_child(property_vbox);

			paginator = memnew(EditorPaginator);
			paginator->connect("page_changed", callable_mp(this, &EditorPropertyDictionary::_page_changed));
			vbox->add_child(paginator);
		} else {
			// Queue children for deletion, deleting immediately might cause errors.
			for (int i = property_vbox->get_child_count() - 1; i >= 0; i--) {
				property_vbox->get_child(i)->queue_free();
			}
		}

		int size = dict.size();

		int max_page = MAX(0, size - 1) / page_length;
		page_index = MIN(page_index, max_page);

		paginator->update(page_index, max_page);
		paginator->set_visible(max_page > 0);

		int offset = page_index * page_length;

		int amount = MIN(size - offset, page_length);
		int total_amount = page_index == max_page ? amount + 2 : amount; // For the "Add Key/Value Pair" box on last page.

		VBoxContainer *add_vbox = nullptr;
		double default_float_step = EDITOR_GET("interface/inspector/default_float_step");

		for (int i = 0; i < total_amount; i++) {
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
					editor->setup(-100000, 100000, 1, false, true, true);
					prop = editor;

				} break;
				case Variant::FLOAT: {
					EditorPropertyFloat *editor = memnew(EditorPropertyFloat);
					editor->setup(-100000, 100000, default_float_step, true, false, true, true);
					prop = editor;
				} break;
				case Variant::STRING: {
					if (i != amount && property_hint == PROPERTY_HINT_MULTILINE_TEXT) {
						// If this is NOT the new key field and there's a multiline hint,
						// show the field as multiline
						prop = memnew(EditorPropertyMultilineText);
					} else {
						prop = memnew(EditorPropertyText);
					}

				} break;

				// Math types.
				case Variant::VECTOR2: {
					EditorPropertyVector2 *editor = memnew(EditorPropertyVector2);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::VECTOR2I: {
					EditorPropertyVector2i *editor = memnew(EditorPropertyVector2i);
					editor->setup(-100000, 100000);
					prop = editor;

				} break;
				case Variant::RECT2: {
					EditorPropertyRect2 *editor = memnew(EditorPropertyRect2);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::RECT2I: {
					EditorPropertyRect2i *editor = memnew(EditorPropertyRect2i);
					editor->setup(-100000, 100000);
					prop = editor;

				} break;
				case Variant::VECTOR3: {
					EditorPropertyVector3 *editor = memnew(EditorPropertyVector3);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::VECTOR3I: {
					EditorPropertyVector3i *editor = memnew(EditorPropertyVector3i);
					editor->setup(-100000, 100000);
					prop = editor;

				} break;
				case Variant::VECTOR4: {
					EditorPropertyVector4 *editor = memnew(EditorPropertyVector4);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::VECTOR4I: {
					EditorPropertyVector4i *editor = memnew(EditorPropertyVector4i);
					editor->setup(-100000, 100000);
					prop = editor;

				} break;
				case Variant::TRANSFORM2D: {
					EditorPropertyTransform2D *editor = memnew(EditorPropertyTransform2D);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::PLANE: {
					EditorPropertyPlane *editor = memnew(EditorPropertyPlane);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::QUATERNION: {
					EditorPropertyQuaternion *editor = memnew(EditorPropertyQuaternion);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::AABB: {
					EditorPropertyAABB *editor = memnew(EditorPropertyAABB);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::BASIS: {
					EditorPropertyBasis *editor = memnew(EditorPropertyBasis);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::TRANSFORM3D: {
					EditorPropertyTransform3D *editor = memnew(EditorPropertyTransform3D);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;
				case Variant::PROJECTION: {
					EditorPropertyProjection *editor = memnew(EditorPropertyProjection);
					editor->setup(-100000, 100000, default_float_step, true);
					prop = editor;

				} break;

				// Miscellaneous types.
				case Variant::COLOR: {
					prop = memnew(EditorPropertyColor);

				} break;
				case Variant::STRING_NAME: {
					EditorPropertyText *ept = memnew(EditorPropertyText);
					ept->set_string_name(true);
					prop = ept;

				} break;
				case Variant::NODE_PATH: {
					prop = memnew(EditorPropertyNodePath);

				} break;
				case Variant::RID: {
					prop = memnew(EditorPropertyRID);

				} break;
				case Variant::SIGNAL: {
					prop = memnew(EditorPropertySignal);

				} break;
				case Variant::CALLABLE: {
					prop = memnew(EditorPropertyCallable);

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
				case Variant::PACKED_BYTE_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_BYTE_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_INT32_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_INT32_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_FLOAT32_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_FLOAT32_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_INT64_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_INT64_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_FLOAT64_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_FLOAT64_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_STRING_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_STRING_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_VECTOR2_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_VECTOR2_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_VECTOR3_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_VECTOR3_ARRAY);
					prop = editor;
				} break;
				case Variant::PACKED_COLOR_ARRAY: {
					EditorPropertyArray *editor = memnew(EditorPropertyArray);
					editor->setup(Variant::PACKED_COLOR_ARRAY);
					prop = editor;
				} break;
				default: {
				}
			}

			ERR_FAIL_NULL(prop);

			prop->set_read_only(is_read_only());

			if (i == amount) {
				PanelContainer *pc = memnew(PanelContainer);
				property_vbox->add_child(pc);
				pc->add_theme_style_override(SNAME("panel"), get_theme_stylebox(SNAME("DictionaryAddItem"), EditorStringName(EditorStyles)));

				add_vbox = memnew(VBoxContainer);
				pc->add_child(add_vbox);
			}
			prop->set_object_and_property(object.ptr(), prop_name);
			int change_index = 0;

			if (i < amount) {
				String cs = key.get_construct_string();
				prop->set_label(key.get_construct_string());
				prop->set_tooltip_text(cs);
				change_index = i + offset;
			} else if (i == amount) {
				prop->set_label(TTR("New Key:"));
				change_index = -1;
			} else if (i == amount + 1) {
				prop->set_label(TTR("New Value:"));
				change_index = -2;
			}

			prop->set_selectable(false);
			prop->connect("property_changed", callable_mp(this, &EditorPropertyDictionary::_property_changed));
			prop->connect("object_id_selected", callable_mp(this, &EditorPropertyDictionary::_object_id_selected));

			HBoxContainer *hbox = memnew(HBoxContainer);
			if (add_vbox) {
				add_vbox->add_child(hbox);
			} else {
				property_vbox->add_child(hbox);
			}
			hbox->add_child(prop);
			prop->set_h_size_flags(SIZE_EXPAND_FILL);
			Button *edit_btn = memnew(Button);
			edit_btn->set_icon(get_editor_theme_icon(SNAME("Edit")));
			edit_btn->set_disabled(is_read_only());
			hbox->add_child(edit_btn);
			edit_btn->connect("pressed", callable_mp(this, &EditorPropertyDictionary::_change_type).bind(edit_btn, change_index));

			prop->update_property();

			if (i == amount + 1) {
				button_add_item = EditorInspector::create_inspector_action_button(TTR("Add Key/Value Pair"));
				button_add_item->set_icon(get_editor_theme_icon(SNAME("Add")));
				button_add_item->set_disabled(is_read_only());
				button_add_item->connect("pressed", callable_mp(this, &EditorPropertyDictionary::_add_key_value));
				add_vbox->add_child(button_add_item);
			}
		}

		updating = false;

	} else {
		if (container) {
			set_bottom_editor(nullptr);
			memdelete(container);
			button_add_item = nullptr;
			container = nullptr;
		}
	}
}

void EditorPropertyDictionary::_object_id_selected(const StringName &p_property, ObjectID p_id) {
	emit_signal(SNAME("object_id_selected"), p_property, p_id);
}

void EditorPropertyDictionary::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			change_type->clear();
			for (int i = 0; i < Variant::VARIANT_MAX; i++) {
				if (i == Variant::CALLABLE || i == Variant::SIGNAL || i == Variant::RID) {
					// These types can't be constructed or serialized properly, so skip them.
					continue;
				}

				String type = Variant::get_type_name(Variant::Type(i));
				change_type->add_icon_item(get_editor_theme_icon(type), type, i);
			}
			change_type->add_separator();
			change_type->add_icon_item(get_editor_theme_icon(SNAME("Remove")), TTR("Remove Item"), Variant::VARIANT_MAX);

			if (button_add_item) {
				button_add_item->set_icon(get_editor_theme_icon(SNAME("Add")));
			}
		} break;
	}
}

void EditorPropertyDictionary::_edit_pressed() {
	Variant prop_val = get_edited_property_value();
	if (prop_val.get_type() == Variant::NIL && edit->is_pressed()) {
		VariantInternal::initialize(&prop_val, Variant::DICTIONARY);
		get_edited_object()->set(get_edited_property(), prop_val);
	}

	get_edited_object()->editor_set_section_unfold(get_edited_property(), edit->is_pressed());
	update_property();
}

void EditorPropertyDictionary::_page_changed(int p_page) {
	if (updating) {
		return;
	}
	page_index = p_page;
	update_property();
}

void EditorPropertyDictionary::_bind_methods() {
}

EditorPropertyDictionary::EditorPropertyDictionary() {
	object.instantiate();
	page_length = int(EDITOR_GET("interface/inspector/max_array_dictionary_items_per_page"));

	edit = memnew(Button);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);
	edit->set_clip_text(true);
	edit->connect("pressed", callable_mp(this, &EditorPropertyDictionary::_edit_pressed));
	edit->set_toggle_mode(true);
	add_child(edit);
	add_focusable(edit);

	container = nullptr;
	button_add_item = nullptr;
	paginator = nullptr;
	change_type = memnew(PopupMenu);
	add_child(change_type);
	change_type->connect("id_pressed", callable_mp(this, &EditorPropertyDictionary::_change_type_menu));
	changing_type_index = -1;
}

///////////////////// LOCALIZABLE STRING ///////////////////////////

void EditorPropertyLocalizableString::_property_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	if (p_property.begins_with("indices")) {
		int index = p_property.get_slice("/", 1).to_int();

		Dictionary dict = object->get_dict().duplicate();
		Variant key = dict.get_key_at_index(index);
		dict[key] = p_value;

		object->set_dict(dict);
		emit_changed(get_edited_property(), dict, "", true);
	}
}

void EditorPropertyLocalizableString::_add_locale_popup() {
	locale_select->popup_locale_dialog();
}

void EditorPropertyLocalizableString::_add_locale(const String &p_locale) {
	Dictionary dict = object->get_dict().duplicate();
	object->set_new_item_key(p_locale);
	object->set_new_item_value(String());
	dict[object->get_new_item_key()] = object->get_new_item_value();

	emit_changed(get_edited_property(), dict, "", false);
	update_property();
}

void EditorPropertyLocalizableString::_remove_item(Object *p_button, int p_index) {
	Dictionary dict = object->get_dict().duplicate();

	Variant key = dict.get_key_at_index(p_index);
	dict.erase(key);

	emit_changed(get_edited_property(), dict, "", false);
	update_property();
}

void EditorPropertyLocalizableString::update_property() {
	Variant updated_val = get_edited_property_value();

	if (updated_val.get_type() == Variant::NIL) {
		edit->set_text(TTR("Localizable String (Nil)")); // This provides symmetry with the array property.
		edit->set_pressed(false);
		if (container) {
			set_bottom_editor(nullptr);
			memdelete(container);
			button_add_item = nullptr;
			container = nullptr;
		}
		return;
	}

	Dictionary dict = updated_val;
	object->set_dict(dict);

	edit->set_text(vformat(TTR("Localizable String (size %d)"), dict.size()));

	bool unfolded = get_edited_object()->editor_is_section_unfolded(get_edited_property());
	if (edit->is_pressed() != unfolded) {
		edit->set_pressed(unfolded);
	}

	if (unfolded) {
		updating = true;

		if (!container) {
			container = memnew(MarginContainer);
			container->set_theme_type_variation("MarginContainer4px");
			add_child(container);
			set_bottom_editor(container);

			VBoxContainer *vbox = memnew(VBoxContainer);
			container->add_child(vbox);

			property_vbox = memnew(VBoxContainer);
			property_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
			vbox->add_child(property_vbox);

			paginator = memnew(EditorPaginator);
			paginator->connect("page_changed", callable_mp(this, &EditorPropertyLocalizableString::_page_changed));
			vbox->add_child(paginator);
		} else {
			// Queue children for deletion, deleting immediately might cause errors.
			for (int i = property_vbox->get_child_count() - 1; i >= 0; i--) {
				property_vbox->get_child(i)->queue_free();
			}
		}

		int size = dict.size();

		int max_page = MAX(0, size - 1) / page_length;
		page_index = MIN(page_index, max_page);

		paginator->update(page_index, max_page);
		paginator->set_visible(max_page > 0);

		int offset = page_index * page_length;

		int amount = MIN(size - offset, page_length);

		for (int i = 0; i < amount; i++) {
			String prop_name;
			Variant key;
			Variant value;

			prop_name = "indices/" + itos(i + offset);
			key = dict.get_key_at_index(i + offset);
			value = dict.get_value_at_index(i + offset);

			EditorProperty *prop = memnew(EditorPropertyText);

			prop->set_object_and_property(object.ptr(), prop_name);
			int remove_index = 0;

			String cs = key.get_construct_string();
			prop->set_label(cs);
			prop->set_tooltip_text(cs);
			remove_index = i + offset;

			prop->set_selectable(false);
			prop->connect("property_changed", callable_mp(this, &EditorPropertyLocalizableString::_property_changed));
			prop->connect("object_id_selected", callable_mp(this, &EditorPropertyLocalizableString::_object_id_selected));

			HBoxContainer *hbox = memnew(HBoxContainer);
			property_vbox->add_child(hbox);
			hbox->add_child(prop);
			prop->set_h_size_flags(SIZE_EXPAND_FILL);
			Button *edit_btn = memnew(Button);
			edit_btn->set_icon(get_editor_theme_icon(SNAME("Remove")));
			hbox->add_child(edit_btn);
			edit_btn->connect("pressed", callable_mp(this, &EditorPropertyLocalizableString::_remove_item).bind(edit_btn, remove_index));

			prop->update_property();
		}

		if (page_index == max_page) {
			button_add_item = EditorInspector::create_inspector_action_button(TTR("Add Translation"));
			button_add_item->set_icon(get_editor_theme_icon(SNAME("Add")));
			button_add_item->connect("pressed", callable_mp(this, &EditorPropertyLocalizableString::_add_locale_popup));
			property_vbox->add_child(button_add_item);
		}

		updating = false;

	} else {
		if (container) {
			set_bottom_editor(nullptr);
			memdelete(container);
			button_add_item = nullptr;
			container = nullptr;
		}
	}
}

void EditorPropertyLocalizableString::_object_id_selected(const StringName &p_property, ObjectID p_id) {
	emit_signal(SNAME("object_id_selected"), p_property, p_id);
}

void EditorPropertyLocalizableString::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			if (button_add_item) {
				button_add_item->set_icon(get_editor_theme_icon(SNAME("Add")));
			}
		} break;
	}
}

void EditorPropertyLocalizableString::_edit_pressed() {
	Variant prop_val = get_edited_property_value();
	if (prop_val.get_type() == Variant::NIL && edit->is_pressed()) {
		VariantInternal::initialize(&prop_val, Variant::DICTIONARY);
		get_edited_object()->set(get_edited_property(), prop_val);
	}

	get_edited_object()->editor_set_section_unfold(get_edited_property(), edit->is_pressed());
	update_property();
}

void EditorPropertyLocalizableString::_page_changed(int p_page) {
	if (updating) {
		return;
	}
	page_index = p_page;
	update_property();
}

void EditorPropertyLocalizableString::_bind_methods() {
}

EditorPropertyLocalizableString::EditorPropertyLocalizableString() {
	object.instantiate();
	page_length = int(EDITOR_GET("interface/inspector/max_array_dictionary_items_per_page"));

	edit = memnew(Button);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);
	edit->set_clip_text(true);
	edit->connect("pressed", callable_mp(this, &EditorPropertyLocalizableString::_edit_pressed));
	edit->set_toggle_mode(true);
	add_child(edit);
	add_focusable(edit);

	container = nullptr;
	button_add_item = nullptr;
	paginator = nullptr;
	updating = false;

	locale_select = memnew(EditorLocaleDialog);
	locale_select->connect("locale_selected", callable_mp(this, &EditorPropertyLocalizableString::_add_locale));
	add_child(locale_select);
}
