/*************************************************************************/
/*  dictionary_property_edit.cpp                                         */
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

#include "dictionary_property_edit.h"
#include "editor_node.h"

void DictionaryPropertyEdit::_notif_change() {
	_change_notify();
}

void DictionaryPropertyEdit::_notif_changev(const String &p_v) {
	_change_notify(p_v.utf8().get_data());
}

void DictionaryPropertyEdit::_set_key(const Variant &p_old_key, const Variant &p_new_key) {
	// TODO: Set key of a dictionary is not allowed yet
}

void DictionaryPropertyEdit::_set_value(const Variant &p_key, const Variant &p_value) {
	Dictionary dict = get_dictionary();
	dict[p_key] = p_value;
	Object *o = ObjectDB::get_instance(obj);
	if (!o) {
		return;
	}

	o->set(property, dict);
}

Variant DictionaryPropertyEdit::get_dictionary() const {
	Object *o = ObjectDB::get_instance(obj);
	if (!o) {
		return Dictionary();
	}
	Variant dict = o->get(property);
	if (dict.get_type() != Variant::DICTIONARY) {
		return Dictionary();
	}
	return dict;
}

void DictionaryPropertyEdit::_get_property_list(List<PropertyInfo> *p_list) const {
	Dictionary dict = get_dictionary();

	Array keys = dict.keys();
	keys.sort();

	for (int i = 0; i < keys.size(); i++) {
		String index = itos(i);

		const Variant &key = keys[i];
		PropertyInfo pi(key.get_type(), index + ": key");
		p_list->push_back(pi);

		const Variant &value = dict[key];
		pi = PropertyInfo(value.get_type(), index + ": value");
		p_list->push_back(pi);
	}
}

void DictionaryPropertyEdit::edit(Object *p_obj, const StringName &p_prop) {
	property = p_prop;
	obj = p_obj->get_instance_id();
}

Node *DictionaryPropertyEdit::get_node() {
	Object *o = ObjectDB::get_instance(obj);
	if (!o) {
		return nullptr;
	}

	return cast_to<Node>(o);
}

bool DictionaryPropertyEdit::_dont_undo_redo() {
	return true;
}

void DictionaryPropertyEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_key"), &DictionaryPropertyEdit::_set_key);
	ClassDB::bind_method(D_METHOD("_set_value"), &DictionaryPropertyEdit::_set_value);
	ClassDB::bind_method(D_METHOD("_notif_change"), &DictionaryPropertyEdit::_notif_change);
	ClassDB::bind_method(D_METHOD("_notif_changev"), &DictionaryPropertyEdit::_notif_changev);
	ClassDB::bind_method(D_METHOD("_dont_undo_redo"), &DictionaryPropertyEdit::_dont_undo_redo);
}

bool DictionaryPropertyEdit::_set(const StringName &p_name, const Variant &p_value) {
	Dictionary dict = get_dictionary();
	Array keys = dict.keys();
	keys.sort();

	String pn = p_name;
	int slash = pn.find(": ");
	if (slash != -1 && pn.length() > slash) {
		String type = pn.substr(slash + 2, pn.length());
		int index = pn.substr(0, slash).to_int();
		if (type == "key" && index < keys.size()) {
			const Variant &key = keys[index];
			UndoRedo *ur = EditorNode::get_undo_redo();

			ur->create_action(TTR("Change Dictionary Key"));
			ur->add_do_method(this, "_set_key", key, p_value);
			ur->add_undo_method(this, "_set_key", p_value, key);
			ur->add_do_method(this, "_notif_changev", p_name);
			ur->add_undo_method(this, "_notif_changev", p_name);
			ur->commit_action();

			return true;
		} else if (type == "value" && index < keys.size()) {
			const Variant &key = keys[index];
			if (dict.has(key)) {
				Variant value = dict[key];
				UndoRedo *ur = EditorNode::get_undo_redo();

				ur->create_action(TTR("Change Dictionary Value"));
				ur->add_do_method(this, "_set_value", key, p_value);
				ur->add_undo_method(this, "_set_value", key, value);
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
	Dictionary dict = get_dictionary();
	Array keys = dict.keys();
	keys.sort();

	String pn = p_name;
	int slash = pn.find(": ");

	if (slash != -1 && pn.length() > slash) {
		String type = pn.substr(slash + 2, pn.length());
		int index = pn.substr(0, slash).to_int();

		if (type == "key" && index < keys.size()) {
			r_ret = keys[index];
			return true;
		} else if (type == "value" && index < keys.size()) {
			const Variant &key = keys[index];
			if (dict.has(key)) {
				r_ret = dict[key];
				return true;
			}
		}
	}

	return false;
}

DictionaryPropertyEdit::DictionaryPropertyEdit() {
	obj = 0;
}
