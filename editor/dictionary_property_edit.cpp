/*************************************************************************/
/*  dictionary_property_edit.cpp                                              */
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
#include "dictionary_property_edit.h"

#include "editor_node.h"

#define ITEMS_PER_PAGE 100

void DictionaryPropertyEdit::get_dictionary_key_values(Variant &p_dict, Variant &p_keys, Variant &p_values) const {
	Object *o = ObjectDB::get_instance(obj);
	if (!o) {
		p_dict = Dictionary();
		p_keys = Array();
		p_values = Array();
	}

	p_dict = o->get(property);
	if (p_dict.get_type() != Variant::DICTIONARY) {
		Variant::CallError ce;
		p_dict = Variant::construct(default_type, NULL, 0, ce);
	}

	p_keys = p_dict.call("keys");
	p_values = p_dict.call("values");
}

void DictionaryPropertyEdit::_notif_change() {
	_change_notify();
}
void DictionaryPropertyEdit::_notif_changev(const String &p_v) {
	_change_notify(p_v.utf8().get_data());
}

void DictionaryPropertyEdit::_set_key(int p_idx, const Variant &p_key) {

	Variant dict, keys, values;
	get_dictionary_key_values(dict, keys, values);

	// change key preserves value
	bool valid;

	Variant old_key = keys.get(p_idx, &valid);
	if (!valid)
		return;

	Variant value = values.get(p_idx, &valid);
	if (!valid)
		return;

	dict.call("erase", old_key);

	dict.set(p_key, value, &valid);
	if (!valid)
		return;

	Object *o = ObjectDB::get_instance(obj);
	if (!o)
		return;

	o->set(property, dict);
}

void DictionaryPropertyEdit::_set_value(int p_idx, const Variant &p_value) {

	Variant dict, keys, values;
	get_dictionary_key_values(dict, keys, values);

	bool valid;
	Variant key = keys.get(p_idx, &valid);
	if (!valid)
		return;

	dict.set(key, p_value, &valid);
	if (!valid)
		return;

	Object *o = ObjectDB::get_instance(obj);
	if (!o)
		return;

	o->set(property, dict);
}

bool DictionaryPropertyEdit::_set(const StringName &p_name, const Variant &p_value) {

	String pn = p_name;

	if (pn.begins_with("dictionary")) {

		if (pn == "dictionary/page") {
			page = p_value;
			_change_notify();
			return true;
		}

	} else if (pn.begins_with("items")) {

		Variant dict, keys, values;
		get_dictionary_key_values(dict, keys, values);

		if (pn.begins_with("items/key")) {

			int idx = pn.get_slicec('_', 1).get_slicec('_', 0).to_int();

			if (pn.ends_with("_type")) {

				//	type
				int type = p_value;
				Variant key = keys.get(idx);
				if (key.get_type() != type && type >= 0 && type < Variant::VARIANT_MAX) {

					Variant::CallError ce;
					Variant new_key = Variant::construct(Variant::Type(type), NULL, 0, ce);
					UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

					ur->create_action(TTR("Change Dictionary Key Type"));
					ur->add_do_method(this, "_set_key", idx, new_key);
					ur->add_undo_method(this, "_set_key", idx, key);
					ur->add_do_method(this, "_notif_change");
					ur->add_undo_method(this, "_notif_change");
					ur->commit_action();
				}

				return true;

			} else {

				Variant key = keys.get(idx);
				UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

				ur->create_action(TTR("Change Dictionary Key"));
				ur->add_do_method(this, "_set_key", idx, p_value);
				ur->add_undo_method(this, "_set_key", idx, key);
				ur->add_do_method(this, "_notif_changev", p_name);
				ur->add_undo_method(this, "_notif_changev", p_name);
				ur->commit_action();
				return true;
			}

		} else if (pn.begins_with("items/value")) {

			int idx = pn.get_slicec('_', 1).get_slicec('_', 0).to_int();

			if (pn.ends_with("_type")) {

				//	type
				int type = p_value;
				Variant value = values.get(idx);
				if (value.get_type() != type && type >= 0 && type < Variant::VARIANT_MAX) {

					Variant::CallError ce;
					Variant new_value = Variant::construct(Variant::Type(type), NULL, 0, ce);
					UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

					ur->create_action(TTR("Change Dictionary Value Type"));
					ur->add_do_method(this, "_set_value", idx, new_value);
					ur->add_undo_method(this, "_set_value", idx, value);
					ur->add_do_method(this, "_notif_change");
					ur->add_undo_method(this, "_notif_change");
					ur->commit_action();
				}

				return true;

			} else {

				Variant value = values.get(idx);
				UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

				ur->create_action(TTR("Change Dictionary Value"));
				ur->add_do_method(this, "_set_value", idx, p_value);
				ur->add_undo_method(this, "_set_value", idx, value);
				ur->add_do_method(this, "_notif_changev", p_name);
				ur->add_undo_method(this, "_notif_changev", p_name);
				ur->commit_action();
				return true;
			}
		}
	}

	return false;
}

bool DictionaryPropertyEdit::_get(const StringName &p_name, Variant &r_ret) const {

	Variant dict, keys, values;
	get_dictionary_key_values(dict, keys, values);

	String pn = p_name;
	if (pn.begins_with("dictionary")) {

		if (pn == "dictionary/size") {
			r_ret = dict.call("size");
			return true;
		}
		if (pn == "dictionary/page") {
			r_ret = page;
			return true;
		}
	} else if (pn.begins_with("items")) {

		if (pn.begins_with("items/key")) {

			int idx = pn.get_slicec('_', 1).get_slicec('_', 0).to_int();

			bool valid;
			r_ret = keys.get(idx, &valid);
			if (valid && pn.ends_with("_type"))
				r_ret = r_ret.get_type();

			return valid;

		} else if (pn.begins_with("items/value")) {

			int idx = pn.get_slicec('_', 1).get_slicec('_', 0).to_int();

			bool valid;
			r_ret = values.get(idx, &valid);
			if (valid && pn.ends_with("_type"))
				r_ret = r_ret.get_type();

			return valid;
		}
	}

	return false;
}

void DictionaryPropertyEdit::_get_property_list(List<PropertyInfo> *p_list) const {

	Variant dict, keys, values;
	get_dictionary_key_values(dict, keys, values);
	int size = dict.call("size");

	p_list->push_back(PropertyInfo(Variant::INT, "dictionary/size", PROPERTY_HINT_RANGE, "0,100000,1"));
	int pages = size / ITEMS_PER_PAGE;
	if (pages > 0)
		p_list->push_back(PropertyInfo(Variant::INT, "dictionary/page", PROPERTY_HINT_RANGE, "0," + itos(pages) + ",1"));

	int offset = page * ITEMS_PER_PAGE;

	int items = MIN(size - offset, ITEMS_PER_PAGE);

	for (int i = 0; i < items; i++) {

		Variant key = keys.get(i);
		ERR_FAIL_COND(key.get_type() == Variant::NIL);
		p_list->push_back(PropertyInfo(Variant::INT, "items/key_" + itos(i + offset) + "_type", PROPERTY_HINT_ENUM, vtypes));

		PropertyInfo key_pi = PropertyInfo(key.get_type(), "items/key_" + itos(i + offset));
		if (key.get_type() == Variant::OBJECT) {
			key_pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			key_pi.hint_string = "Resource";
		}
		p_list->push_back(key_pi);

		Variant value = values.get(i);
		p_list->push_back(PropertyInfo(Variant::INT, "items/value_" + itos(i + offset) + "_type", PROPERTY_HINT_ENUM, vtypes));
		PropertyInfo value_pi = PropertyInfo(value.get_type(), "items/value_" + itos(i + offset));
		if (value.get_type() == Variant::OBJECT) {
			value_pi.hint = PROPERTY_HINT_RESOURCE_TYPE;
			value_pi.hint_string = "Resource";
		}
		p_list->push_back(value_pi);
	}
}

void DictionaryPropertyEdit::edit(Object *p_obj, const StringName &p_prop, const String &p_hint_string, Variant::Type p_deftype) {

	page = 0;
	property = p_prop;
	obj = p_obj->get_instance_id();
	default_type = p_deftype;
}

Node *DictionaryPropertyEdit::get_node() {
	return Object::cast_to<Node>(ObjectDB::get_instance(obj));
}

void DictionaryPropertyEdit::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_set_key"), &DictionaryPropertyEdit::_set_key);
	ClassDB::bind_method(D_METHOD("_set_value"), &DictionaryPropertyEdit::_set_value);
	ClassDB::bind_method(D_METHOD("_notif_change"), &DictionaryPropertyEdit::_notif_change);
	ClassDB::bind_method(D_METHOD("_notif_changev"), &DictionaryPropertyEdit::_notif_changev);
}

DictionaryPropertyEdit::DictionaryPropertyEdit() {
	page = 0;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {

		if (i > 0)
			vtypes += ",";
		vtypes += Variant::get_type_name(Variant::Type(i));
	}
	default_type = Variant::NIL;
}
