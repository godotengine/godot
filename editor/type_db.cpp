/*************************************************************************/
/*  type_db.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/type_db.h"
#include "editor/editor_node.h"

void TypeDB::refresh() {
	type_map.clear();
	path_map.clear();
	const EditorData &ed = EditorNode::get_editor_data();

	List<StringName> types;
	ClassDB::get_class_list(&types);
	for (List<StringName>::Element *E = types.front(); E; E = E->next()) {
		TypeInfo ti;
		StringName n = E->get();

		ti.name = n;
		ti.base = ClassDB::get_parent_class(n);
		ti.native = n;
		ti.source = SOURCE_ENGINE;

		type_map[ti.name] = ti;
	}
	for (List<StringName>::Element *E = types.front(); E; E = E->next()) {
		StringName n = E->get();
		type_map[n].icon = EditorNode::get_singleton()->get_class_icon(n);
	}
	types.clear();
	ScriptServer::get_global_class_list(&types);
	for (List<StringName>::Element *E = types.front(); E; E = E->next()) {
		TypeInfo ti;
		StringName n = E->get();

		ti.name = n;
		ti.script = ResourceLoader::load(ScriptServer::get_global_class_path(n));
		if (ti.script.is_null())
			continue;
		String icon_path = ed.script_class_get_icon_path(n);
		if (icon_path.length())
			ti.icon = ResourceLoader::load(ScriptServer::get_global_class_path(n));
		ti.native = ti.script->get_instance_base_type();
		ti.base = ScriptServer::get_global_class_base(n);
		ti.source = SOURCE_SCRIPT_CLASS;

		type_map[ti.name] = ti;
		path_map[ti.script->get_path()] = ti.name;
	}
	types.clear();
	ed.get_custom_type_class_list(&types);
	for (List<StringName>::Element *E = types.front(); E; E = E->next()) {
		TypeInfo ti;
		StringName n = E->get();

		ti.name = n;
		ti.script = ResourceLoader::load(ScriptServer::get_global_class_path(n));
		if (ti.script.is_null())
			continue;
		RES icon = ed.get_custom_type_icon(n);
		if (icon.is_valid())
			ti.icon = icon;
		ti.native = ti.script->get_instance_base_type();
		ti.base = ed.get_custom_type_base(n);
		ti.source = SOURCE_CUSTOM_TYPE;

		type_map[ti.name] = ti;
		path_map[ti.script->get_path()] = ti.name;
	}
	//types.clear();
	//List<String> paths;
	//path_map.get_key_list(&paths);
	//for (List<String>::Element *E = paths.front(); E; E = E->next()) {
	//	TypeInfo &ti = type_map[path_map[E->get()]];
	//	if (ti.base == StringName())
	//		continue;
	//	Ref<Script> script = ti.script;
	//	if (script.is_null())
	//		continue;
	//	script = script->get_base_script();
	//	while (script.is_valid()) {
	//		const String &p = script->get_path();
	//		if (path_map.has(p)) {
	//			ti.base = path_map[p];
	//			break;
	//		}
	//	}
	//	if (script.is_null())
	//		ti.base = ti.native;
	//}
}

bool TypeDB::class_exists(const StringName &p_type) const {
	return type_map.has(p_type);
}

bool TypeDB::path_exists(const String &p_path) const {
	return path_map.has(p_path);
}

StringName TypeDB::get_class_by_path(const String &p_path) const {
	return path_exists(p_path) ? path_map[p_path] : StringName();
}

String TypeDB::get_path_by_class(const StringName &p_type) const {
	if (!class_exists(p_type))
		return String();
	Ref<Script> script = type_map[p_type].script;
	if (script.is_null())
		return String();
	return script->get_path();
}

StringName TypeDB::get_native(const StringName &p_type) const {
	ERR_FAIL_COND_V(!type_map.has(p_type), StringName());
	return type_map[p_type].native;
}

Object *TypeDB::instance(const StringName &p_type) const {
	ERR_FAIL_COND_V(!type_map.has(p_type), NULL);

	const TypeInfo &ti = type_map[p_type];

	switch (ti.source) {
		case SOURCE_ENGINE: {
			return ClassDB::instance(ti.name);
		} break;
		case SOURCE_SCRIPT_CLASS:
		case SOURCE_CUSTOM_TYPE: {
			Object *obj = ClassDB::instance(ti.native);
			if (!obj)
				return NULL;
			if (ti.script.is_null())
				return NULL;
			obj->set_script(ti.script.get_ref_ptr());
			return obj;
		} break;
		case SOURCE_NONE: {
			return NULL;
		} break;
	}
	return NULL;
}

RES TypeDB::get_icon(const StringName &p_type) const {
	ERR_FAIL_COND_V(!type_map.has(p_type), NULL);
	return type_map[p_type].icon;
}

bool TypeDB::can_instance(const StringName &p_type) const {
	ERR_FAIL_COND_V(!type_map.has(p_type), false);
	return ClassDB::can_instance(type_map[p_type].native);
}

void TypeDB::get_class_list(List<StringName> *p_class_list) const {
	type_map.get_key_list(p_class_list);
}

bool TypeDB::is_parent_class(const String &p_type, const String &p_inherits) const {
	ERR_FAIL_COND_V(!type_map.has(p_type), false);
	const TypeInfo *ti = &type_map[p_type];
	while (ti) {
		if (ti->name == p_inherits)
			return true;
		ti = type_map.has(ti->base) ? &type_map[ti->base] : NULL;
	}
	return false;
}

String TypeDB::get_parent_class(const String &p_type) const {
	ERR_FAIL_COND_V(!type_map.has(p_type), String());
	return type_map[p_type].base;
}

void TypeDB::get_inheritors_from_class(const String &p_type, List<StringName> *p_classes) const {
	ERR_FAIL_COND(!type_map.has(p_type));
	ERR_FAIL_COND(!p_classes);

	const TypeInfo &ti = type_map[p_type];
	List<StringName> types;
	type_map.get_key_list(&types);
	for (List<StringName>::Element *E = types.front(); E; E = E->next()) {
		const TypeInfo &a_ti = type_map[E->get()];
		if (is_parent_class(a_ti.name, ti.name))
			p_classes->push_back(a_ti.name);
	}
}

void TypeDB::_bind_methods() {}

TypeDB::TypeDB() {}