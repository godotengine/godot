#include "editor_properties_array_dict.h"
#include "editor_properties.h"
#define ITEMS_PER_PAGE 100

bool EditorPropertyArrayObject::_set(const StringName &p_name, const Variant &p_value) {

	String pn = p_name;

	if (pn.begins_with("indices")) {
		int idx = pn.get_slicec('/', 1).to_int();
		array.set(idx, p_value);
		return true;
	}

	return false;
}

bool EditorPropertyArrayObject::_get(const StringName &p_name, Variant &r_ret) const {

	String pn = p_name;

	if (pn.begins_with("indices")) {

		int idx = pn.get_slicec('/', 1).to_int();
		bool valid;
		r_ret = array.get(idx, &valid);
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

/////////////////////

void EditorPropertyArray::_property_changed(const String &p_prop, Variant p_value) {

	if (p_prop.begins_with("indices")) {
		int idx = p_prop.get_slice("/", 1).to_int();
		Variant array = object->get_array();
		array.set(idx, p_value);
		emit_signal("property_changed", get_edited_property(), array);

		if (array.get_type() == Variant::ARRAY) {
			array = array.call("duplicate"); //dupe, so undo/redo works better
		}
		object->set_array(array);
	}
}

void EditorPropertyArray::_change_type(Object *p_button, int p_index) {

	Button *button = Object::cast_to<Button>(p_button);

	Rect2 rect = button->get_global_rect();
	change_type->set_as_minsize();
	change_type->set_global_position(rect.position + rect.size - Vector2(change_type->get_combined_minimum_size().x, 0));
	change_type->popup();
	changing_type_idx = p_index;
}

void EditorPropertyArray::_change_type_menu(int p_index) {

	Variant value;
	Variant::CallError ce;
	value = Variant::construct(Variant::Type(p_index), NULL, 0, ce);
	Variant array = object->get_array();
	array.set(changing_type_idx, value);

	emit_signal("property_changed", get_edited_property(), array);

	if (array.get_type() == Variant::ARRAY) {
		array = array.call("duplicate"); //dupe, so undo/redo works better
	}
	object->set_array(array);
	update_property();
}

void EditorPropertyArray::update_property() {

	Variant array = get_edited_object()->get(get_edited_property());

	if ((!array.is_array()) != edit->is_disabled()) {

		if (array.is_array()) {
			edit->set_disabled(false);
			edit->set_pressed(false);

		} else {
			edit->set_disabled(true);
			if (vbox) {
				memdelete(vbox);
			}
		}
	}

	if (!array.is_array()) {
		return;
	}

	String arrtype;
	switch (array.get_type()) {
		case Variant::ARRAY: {

			arrtype = "Array";

		} break;

		// arrays
		case Variant::POOL_BYTE_ARRAY: {
			arrtype = "ByteArray";

		} break;
		case Variant::POOL_INT_ARRAY: {
			arrtype = "IntArray";

		} break;
		case Variant::POOL_REAL_ARRAY: {

			arrtype = "FltArray";
		} break;
		case Variant::POOL_STRING_ARRAY: {

			arrtype = "StrArray";
		} break;
		case Variant::POOL_VECTOR2_ARRAY: {

			arrtype = "Vec2Array";
		} break;
		case Variant::POOL_VECTOR3_ARRAY: {
			arrtype = "Vec3Array";

		} break;
		case Variant::POOL_COLOR_ARRAY: {
			arrtype = "ColArray";
		} break;
	}

	edit->set_text(arrtype + "[" + itos(array.call("size")) + "]");

#ifdef TOOLS_ENABLED

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
			HBoxContainer *hbc = memnew(HBoxContainer);
			vbox->add_child(hbc);
			Label *label = memnew(Label(TTR("Size: ")));
			label->set_h_size_flags(SIZE_EXPAND_FILL);
			hbc->add_child(label);
			length = memnew(EditorSpinSlider);
			length->set_step(1);
			length->set_max(1000000);
			length->set_h_size_flags(SIZE_EXPAND_FILL);
			hbc->add_child(length);
			length->connect("value_changed", this, "_length_changed");

			page_hb = memnew(HBoxContainer);
			vbox->add_child(page_hb);
			label = memnew(Label(TTR("Page: ")));
			label->set_h_size_flags(SIZE_EXPAND_FILL);
			page_hb->add_child(label);
			page = memnew(EditorSpinSlider);
			page->set_step(1);
			page_hb->add_child(page);
			page->set_h_size_flags(SIZE_EXPAND_FILL);
			page->connect("value_changed", this, "_page_changed");
		} else {
			//bye bye children of the box
			while (vbox->get_child_count() > 2) {
				memdelete(vbox->get_child(2));
			}
		}

		int len = array.call("size");

		length->set_value(len);

		int pages = MAX(0, len - 1) / page_len + 1;

		page->set_max(pages);
		page_idx = MIN(page_idx, pages - 1);
		page->set_value(page_idx);
		page_hb->set_visible(pages > 1);

		int offset = page_idx * page_len;

		int amount = MIN(len - offset, page_len);

		if (array.get_type() == Variant::ARRAY) {
			array = array.call("duplicate");
		}

		object->set_array(array);

		for (int i = 0; i < amount; i++) {
			String prop_name = "indices/" + itos(i + offset);

			EditorProperty *prop = NULL;
			Variant value = array.get(i + offset);

			switch (value.get_type()) {
				case Variant::NIL: {
					prop = memnew(EditorPropertyNil);

				} break;

				// atomic types
				case Variant::BOOL: {

					prop = memnew(EditorPropertyCheck);

				} break;
				case Variant::INT: {
					EditorPropertyInteger *ed = memnew(EditorPropertyInteger);
					ed->setup(-100000, 100000, true, true);
					prop = ed;

				} break;
				case Variant::REAL: {

					EditorPropertyFloat *ed = memnew(EditorPropertyFloat);
					ed->setup(-100000, 100000, 0.001, true, false, true, true);
					prop = ed;
				} break;
				case Variant::STRING: {

					prop = memnew(EditorPropertyText);

				} break;

					// math types

				case Variant::VECTOR2: {

					EditorPropertyVector2 *ed = memnew(EditorPropertyVector2);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::RECT2: {

					EditorPropertyRect2 *ed = memnew(EditorPropertyRect2);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::VECTOR3: {

					EditorPropertyVector3 *ed = memnew(EditorPropertyVector3);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::TRANSFORM2D: {

					EditorPropertyTransform2D *ed = memnew(EditorPropertyTransform2D);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::PLANE: {

					EditorPropertyPlane *ed = memnew(EditorPropertyPlane);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::QUAT: {

					EditorPropertyQuat *ed = memnew(EditorPropertyQuat);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::AABB: {

					EditorPropertyAABB *ed = memnew(EditorPropertyAABB);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::BASIS: {
					EditorPropertyBasis *ed = memnew(EditorPropertyBasis);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;
				case Variant::TRANSFORM: {
					EditorPropertyTransform *ed = memnew(EditorPropertyTransform);
					ed->setup(-100000, 100000, 0.001, true);
					prop = ed;

				} break;

				// misc types
				case Variant::COLOR: {
					prop = memnew(EditorPropertyColor);

				} break;
				case Variant::NODE_PATH: {
					prop = memnew(EditorPropertyNodePath);

				} break;
				case Variant::_RID: {
					prop = memnew(EditorPropertyNil);

				} break;
				case Variant::OBJECT: {

					prop = memnew(EditorPropertyResource);

				} break;
				case Variant::DICTIONARY: {
					prop = memnew(EditorPropertyNil);

				} break;
				case Variant::ARRAY: {

					prop = memnew(EditorPropertyArray);

				} break;

				// arrays
				case Variant::POOL_BYTE_ARRAY: {
					prop = memnew(EditorPropertyArray);

				} break;
				case Variant::POOL_INT_ARRAY: {
					prop = memnew(EditorPropertyArray);

				} break;
				case Variant::POOL_REAL_ARRAY: {

					prop = memnew(EditorPropertyArray);
				} break;
				case Variant::POOL_STRING_ARRAY: {

					prop = memnew(EditorPropertyArray);
				} break;
				case Variant::POOL_VECTOR2_ARRAY: {

					prop = memnew(EditorPropertyArray);
				} break;
				case Variant::POOL_VECTOR3_ARRAY: {
					prop = memnew(EditorPropertyArray);

				} break;
				case Variant::POOL_COLOR_ARRAY: {
					prop = memnew(EditorPropertyArray);

				} break;
				default: {}
			}

			prop->set_object_and_property(object.ptr(), prop_name);
			prop->set_label(itos(i + offset));
			prop->set_selectable(false);
			prop->connect("property_changed", this, "_property_changed");
			if (array.get_type() == Variant::ARRAY) {
				HBoxContainer *hb = memnew(HBoxContainer);
				vbox->add_child(hb);
				hb->add_child(prop);
				prop->set_h_size_flags(SIZE_EXPAND_FILL);
				Button *edit = memnew(Button);
				edit->set_icon(get_icon("Edit", "EditorIcons"));
				hb->add_child(edit);
				edit->connect("pressed", this, "_change_type", varray(edit, i + offset));
			} else {
				vbox->add_child(prop);
			}

			prop->update_property();
		}

		updating = false;

	} else {
		if (vbox) {
			set_bottom_editor(NULL);
			memdelete(vbox);
			vbox = NULL;
		}
	}
#endif
}

void EditorPropertyArray::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
	}
}
void EditorPropertyArray::_edit_pressed() {

	get_edited_object()->editor_set_section_unfold(get_edited_property(), edit->is_pressed());
	update_property();
}

void EditorPropertyArray::_page_changed(double p_page) {
	if (updating)
		return;
	page_idx = p_page;
	update_property();
}

void EditorPropertyArray::_length_changed(double p_page) {
	if (updating)
		return;

	Variant array = object->get_array();
	array.call("resize", int(p_page));
	emit_signal("property_changed", get_edited_property(), array);

	if (array.get_type() == Variant::ARRAY) {
		array = array.call("duplicate"); //dupe, so undo/redo works better
	}
	object->set_array(array);
	update_property();
}

void EditorPropertyArray::_bind_methods() {
	ClassDB::bind_method("_edit_pressed", &EditorPropertyArray::_edit_pressed);
	ClassDB::bind_method("_page_changed", &EditorPropertyArray::_page_changed);
	ClassDB::bind_method("_length_changed", &EditorPropertyArray::_length_changed);
	ClassDB::bind_method("_property_changed", &EditorPropertyArray::_property_changed);
	ClassDB::bind_method("_change_type", &EditorPropertyArray::_change_type);
	ClassDB::bind_method("_change_type_menu", &EditorPropertyArray::_change_type_menu);
}

EditorPropertyArray::EditorPropertyArray() {

	object.instance();
	page_idx = 0;
	page_len = 10;
	edit = memnew(Button);
	edit->set_flat(true);
	edit->set_h_size_flags(SIZE_EXPAND_FILL);
	edit->set_clip_text(true);
	edit->connect("pressed", this, "_edit_pressed");
	edit->set_toggle_mode(true);
	add_child(edit);
	add_focusable(edit);
	vbox = NULL;
	page = NULL;
	length = NULL;
	updating = false;
	change_type = memnew(PopupMenu);
	add_child(change_type);
	change_type->connect("id_pressed", this, "_change_type_menu");
	changing_type_idx = -1;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		String type = Variant::get_type_name(Variant::Type(i));
		change_type->add_item(type, i);
	}
	changing_type_idx = -1;
}
