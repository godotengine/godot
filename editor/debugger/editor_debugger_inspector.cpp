/**************************************************************************/
/*  editor_debugger_inspector.cpp                                         */
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

#include "editor_debugger_inspector.h"

#include "core/debugger/debugger_marshalls.h"
#include "core/io/marshalls.h"
#include "editor/docks/inspector_dock.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/debugger/scene_debugger.h"

bool EditorDebuggerRemoteObjects::_set(const StringName &p_name, const Variant &p_value) {
	return _set_impl(p_name, p_value, "");
}

bool EditorDebuggerRemoteObjects::_set_impl(const StringName &p_name, const Variant &p_value, const String &p_field) {
	String name = p_name;
	if (!prop_values.has(name) || String(name).begins_with("Constants/")) {
		return false;
	}

	// Change it back to the real name when fetching.
	if (name == "Script") {
		name = "script";
	} else if (name.begins_with("Metadata/")) {
		name = name.replace_first("Metadata/", "metadata/");
	}

	Dictionary &values = prop_values[p_name];
	Dictionary old_values = values.duplicate();
	for (const uint64_t key : values.keys()) {
		values.set(key, p_value);
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	const int size = remote_object_ids.size();
	ur->create_action(size == 1 ? vformat(TTR("Set %s"), name) : vformat(TTR("Set %s on %d objects"), name, size), UndoRedo::MERGE_ENDS);

	ur->add_do_method(this, SNAME("emit_signal"), SNAME("values_edited"), name, values, p_field);
	ur->add_undo_method(this, SNAME("emit_signal"), SNAME("values_edited"), name, old_values, p_field);
	ur->commit_action();

	return true;
}

bool EditorDebuggerRemoteObjects::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (!prop_values.has(name)) {
		return false;
	}

	// Change it back to the real name when fetching.
	if (name == "Script") {
		name = "script";
	} else if (name.begins_with("Metadata/")) {
		name = name.replace_first("Metadata/", "metadata/");
	}

	r_ret = prop_values[p_name][remote_object_ids[0]];
	return true;
}

void EditorDebuggerRemoteObjects::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->clear(); // Sorry, don't want any categories.
	for (const PropertyInfo &prop : prop_list) {
		p_list->push_back(prop);
	}
}

void EditorDebuggerRemoteObjects::set_property_field(const StringName &p_property, const Variant &p_value, const String &p_field) {
	// Ignore the field with arrays and dictionaries, as they are passed whole when edited.
	Variant::Type type = p_value.get_type();
	if (type == Variant::ARRAY || type == Variant::DICTIONARY) {
		_set_impl(p_property, p_value, "");
	} else {
		_set_impl(p_property, p_value, p_field);
	}
}

String EditorDebuggerRemoteObjects::get_title() {
	if (!remote_object_ids.is_empty() && ObjectID(remote_object_ids[0].operator uint64_t()).is_valid()) {
		const int size = remote_object_ids.size();
		return size == 1 ? vformat(TTR("Remote %s: %d"), type_name, remote_object_ids[0]) : vformat(TTR("Remote %s (%d Selected)"), type_name, size);
	}

	return "<null>";
}

Variant EditorDebuggerRemoteObjects::get_variant(const StringName &p_name) {
	Variant var;
	_get(p_name, var);
	return var;
}

void EditorDebuggerRemoteObjects::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_title"), &EditorDebuggerRemoteObjects::get_title);
	ClassDB::bind_method("_hide_script_from_inspector", &EditorDebuggerRemoteObjects::_hide_script_from_inspector);
	ClassDB::bind_method("_hide_metadata_from_inspector", &EditorDebuggerRemoteObjects::_hide_metadata_from_inspector);

	ADD_SIGNAL(MethodInfo("values_edited", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::DICTIONARY, "values", PROPERTY_HINT_DICTIONARY_TYPE, "uint64_t:Variant"), PropertyInfo(Variant::STRING, "field")));
}

/// EditorDebuggerInspector

EditorDebuggerInspector::EditorDebuggerInspector() {
	variables = memnew(EditorDebuggerRemoteObjects);
}

EditorDebuggerInspector::~EditorDebuggerInspector() {
	clear_cache();
	memdelete(variables);
}

void EditorDebuggerInspector::_bind_methods() {
	ADD_SIGNAL(MethodInfo("object_selected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("objects_edited", PropertyInfo(Variant::ARRAY, "ids"), PropertyInfo(Variant::STRING, "property"), PropertyInfo("value"), PropertyInfo(Variant::STRING, "field")));
	ADD_SIGNAL(MethodInfo("object_property_updated", PropertyInfo(Variant::INT, "id"), PropertyInfo(Variant::STRING, "property")));
}

void EditorDebuggerInspector::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			connect("object_id_selected", callable_mp(this, &EditorDebuggerInspector::_object_selected));
		} break;

		case NOTIFICATION_ENTER_TREE: {
			variables->remote_object_ids.append(0);
			edit(variables);
		} break;
	}
}

void EditorDebuggerInspector::_objects_edited(const String &p_prop, const TypedDictionary<uint64_t, Variant> &p_values, const String &p_field) {
	emit_signal(SNAME("objects_edited"), p_prop, p_values, p_field);
}

void EditorDebuggerInspector::_object_selected(ObjectID p_object) {
	emit_signal(SNAME("object_selected"), p_object);
}

EditorDebuggerRemoteObjects *EditorDebuggerInspector::set_objects(const Array &p_arr) {
	ERR_FAIL_COND_V(p_arr.is_empty(), nullptr);

	TypedArray<uint64_t> ids;
	LocalVector<SceneDebuggerObject> objects;
	for (const Array arr : p_arr) {
		SceneDebuggerObject obj;
		obj.deserialize(arr);
		if (obj.id.is_valid()) {
			ids.push_back((uint64_t)obj.id);
			objects.push_back(obj);
		}
	}
	ERR_FAIL_COND_V(ids.is_empty(), nullptr);

	// Sorting is necessary, as selected nodes in the remote tree are ordered by index.
	ids.sort();

	EditorDebuggerRemoteObjects *remote_objects = nullptr;
	for (EditorDebuggerRemoteObjects *robjs : remote_objects_list) {
		if (robjs->remote_object_ids == ids) {
			remote_objects = robjs;
			break;
		}
	}

	if (!remote_objects) {
		remote_objects = memnew(EditorDebuggerRemoteObjects);
		remote_objects->remote_object_ids = ids;
		remote_objects->remote_object_ids.make_read_only();
		remote_objects->connect("values_edited", callable_mp(this, &EditorDebuggerInspector::_objects_edited));
		remote_objects_list.push_back(remote_objects);
	}

	StringName class_name = objects[0].class_name;
	if (class_name != SNAME("Object")) {
		// Search for the common class between all selected objects.
		bool check_type_again = true;
		while (check_type_again) {
			check_type_again = false;

			if (class_name == SNAME("Object") || class_name == StringName()) {
				// All objects inherit from Object, so no need to continue checking.
				class_name = SNAME("Object");
				break;
			}

			// Check that all objects inherit from type_name.
			for (const SceneDebuggerObject &obj : objects) {
				if (obj.class_name == class_name || ClassDB::is_parent_class(obj.class_name, class_name)) {
					continue; // class_name is the same or a parent of the object's class.
				}

				// class_name is not a parent of the node's class, so check again with the parent class.
				class_name = ClassDB::get_parent_class(class_name);
				check_type_again = true;
				break;
			}
		}
	}
	remote_objects->type_name = class_name;

	// Search for properties that are present in all selected objects.
	struct UsageData {
		int qty = 0;
		SceneDebuggerObject::SceneDebuggerProperty prop;
		TypedDictionary<uint64_t, Variant> values;
	};
	HashMap<String, UsageData> usage;
	int nc = 0;
	for (const SceneDebuggerObject &obj : objects) {
		for (const SceneDebuggerObject::SceneDebuggerProperty &prop : obj.properties) {
			PropertyInfo pinfo = prop.first;
			// Rename those variables, so they don't conflict with the ones from the resource itself.
			if (pinfo.name == "script") {
				pinfo.name = "Script";
			} else if (pinfo.name.begins_with("metadata/")) {
				pinfo.name = pinfo.name.replace_first("metadata/", "Metadata/");
			}

			if (!usage.has(pinfo.name)) {
				UsageData usage_dt;
				usage_dt.prop = prop;
				usage_dt.prop.first.name = pinfo.name;
				usage_dt.values[obj.id] = prop.second;
				usage[pinfo.name] = usage_dt;
			}

			// Make sure only properties with the same exact PropertyInfo data will appear.
			if (usage[pinfo.name].prop.first == pinfo) {
				usage[pinfo.name].qty++;
				usage[pinfo.name].values[obj.id] = prop.second;
			}
		}

		nc++;
	}
	for (HashMap<String, UsageData>::Iterator E = usage.begin(); E;) {
		HashMap<String, UsageData>::Iterator next = E;
		++next;

		UsageData usage_dt = E->value;
		if (nc != usage_dt.qty) {
			// Doesn't appear on all of them, remove it.
			usage.erase(E->key);
		}

		E = next;
	}

	int old_prop_size = remote_objects->prop_list.size();

	remote_objects->prop_list.clear();
	int new_props_added = 0;
	HashSet<String> changed;
	for (KeyValue<String, UsageData> &KV : usage) {
		const PropertyInfo &pinfo = KV.value.prop.first;
		Variant var = KV.value.values[remote_objects->remote_object_ids[0]];

		// Always add the property, since props may have been added or removed.
		remote_objects->prop_list.push_back(pinfo);

		if (!remote_objects->prop_values.has(pinfo.name)) {
			new_props_added++;
		} else if (bool(Variant::evaluate(Variant::OP_NOT_EQUAL, remote_objects->prop_values[pinfo.name], var))) {
			changed.insert(pinfo.name);
		}

		remote_objects->prop_values[pinfo.name] = KV.value.values;
	}

	if (old_prop_size == remote_objects->prop_list.size() && new_props_added == 0) {
		// Only some may have changed, if so, then update those, if they exist.
		for (const String &E : changed) {
			emit_signal(SNAME("object_property_updated"), remote_objects->get_instance_id(), E);
		}
	} else {
		// Full update, because props were added or removed.
		remote_objects->update();
	}

	return remote_objects;
}

void EditorDebuggerInspector::clear_remote_inspector() {
	if (remote_objects_list.is_empty()) {
		return;
	}

	const Object *obj = InspectorDock::get_inspector_singleton()->get_edited_object();
	// Check if the inspector holds remote items, and take it out if so.
	if (Object::cast_to<EditorDebuggerRemoteObjects>(obj)) {
		EditorNode::get_singleton()->push_item(nullptr);
	}
}

void EditorDebuggerInspector::clear_cache() {
	clear_remote_inspector();

	for (EditorDebuggerRemoteObjects *robjs : remote_objects_list) {
		memdelete(robjs);
	}
	remote_objects_list.clear();

	remote_dependencies.clear();
}

void EditorDebuggerInspector::invalidate_selection_from_cache(const TypedArray<uint64_t> &p_ids) {
	for (EditorDebuggerRemoteObjects *robjs : remote_objects_list) {
		if (robjs->remote_object_ids == p_ids) {
			const Object *obj = InspectorDock::get_inspector_singleton()->get_edited_object();
			if (obj == robjs) {
				EditorNode::get_singleton()->push_item(nullptr);
			}

			remote_objects_list.erase(robjs);
			memdelete(robjs);
			break;
		}
	}
}

void EditorDebuggerInspector::add_stack_variable(const Array &p_array, int p_offset) {
	DebuggerMarshalls::ScriptStackVariable var;
	var.deserialize(p_array);
	String n = var.name;
	Variant v = var.value;

	PropertyHint h = PROPERTY_HINT_NONE;
	String hs;

	if (var.var_type == Variant::OBJECT && v) {
		v = Object::cast_to<EncodedObjectAsID>(v)->get_object_id();
		h = PROPERTY_HINT_OBJECT_ID;
		hs = "Object";
	}
	String type;
	switch (var.type) {
		case 0:
			type = "Locals/";
			break;
		case 1:
			type = "Members/";
			break;
		case 2:
			type = "Globals/";
			break;
		case 3:
			type = "Evaluated/";
			break;
		default:
			type = "Unknown/";
	}

	PropertyInfo pinfo;
	// Encode special characters to avoid issues with expressions in Evaluator.
	// Dots are skipped by uri_encode(), but uri_decode() process them correctly when replaced with "%2E".
	pinfo.name = type + n.uri_encode().replace(".", "%2E");
	pinfo.type = v.get_type();
	pinfo.hint = h;
	pinfo.hint_string = hs;

	if ((p_offset == -1) || variables->prop_list.is_empty()) {
		variables->prop_list.push_back(pinfo);
	} else {
		List<PropertyInfo>::Element *current = variables->prop_list.front();
		for (int i = 0; i < p_offset; i++) {
			current = current->next();
		}
		variables->prop_list.insert_before(current, pinfo);
	}
	variables->prop_values[pinfo.name][0] = v;
	variables->update();
	edit(variables);
}

void EditorDebuggerInspector::clear_stack_variables() {
	variables->clear();
	variables->update();
}

String EditorDebuggerInspector::get_stack_variable(const String &p_var) {
	for (KeyValue<StringName, TypedDictionary<uint64_t, Variant>> &E : variables->prop_values) {
		String v = E.key.operator String();
		if (v.get_slicec('/', 1) == p_var) {
			return variables->get_variant(v);
		}
	}
	return String();
}
