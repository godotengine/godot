/*************************************************************************/
/*  class_type.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "class_type.h"

#include "core/io/resource_loader.h"
#include "core/os/file_access.h"
#include "editor_node.h"
#include "scene/resources/packed_scene.h"

StringName ClassType::_get_name_by_path(const String &p_path, Source *r_source) {
	if (!is_valid_path(p_path)) {
		return StringName();
	}
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		StringName name = EditorNode::get_editor_data().script_class_get_name(p_path);
		if (r_source && name != StringName())
			*r_source = SOURCE_SCRIPT;
		return name;
	}
#else
	Ref<Script> script = ResourceLoader::load(p_path, "Script");
	if (script.is_valid()) {
		StringName name = script->get_language()->get_global_class_name(p_path);
		if (r_source && name != StringName())
			*r_source = SOURCE_SCRIPT;
		return name;
	}
#endif
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		StringName name = EditorNode::get_singleton()->get_project_settings()->get_scene_template_settings()->scene_template_get_name(p_path);
		if (r_source && name != StringName())
			*r_source = SOURCE_SCENE;
		return name;
	}
#endif
	if (r_source)
		*r_source = SOURCE_NONE;
	return StringName();
}

void ClassType::_update_path_by_name() {
	if (_source_dirty)
		_update_source_by_name();
	if (_name == "") { //shortcut
		_path = "";
	} else if (is_scene_template()) {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint())
			_path = EditorNode::get_singleton()->get_project_settings()->get_scene_template_settings()->scene_template_get_path(_name);
#else
		_path = "";
#endif
	} else if (is_script_class()) {
		_path = ScriptServer::get_global_class_path(_name);
	} else {
		_path = "";
	}
}

void ClassType::_update_source_by_name() {
	if (!_source_dirty)
		return;

	_source_dirty = false;
	_source = SOURCE_NONE;

	if (_name == StringName()) {
		return;
	} else if (ClassDB::class_exists(_name)) {
		_source = SOURCE_ENGINE;
	} else if (ScriptServer::is_global_class(_name)) {
		_source = SOURCE_SCRIPT;
	} else {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			if (EditorNode::get_singleton()->get_project_settings()->get_scene_template_settings()->scene_template_exists(_name)) {
				_source = SOURCE_SCENE;
			}
		}
#endif
	}
}

Array ClassType::_get_engine_class_list() const {
	List<StringName> names;
	get_engine_class_list(&names);
	names.sort_custom<StringName::AlphCompare>();
	Array ret;
	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

Array ClassType::_get_script_class_list() const {
	List<StringName> names;
	get_script_class_list(&names);
	names.sort_custom<StringName::AlphCompare>();
	Array ret;
	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

Array ClassType::_get_scene_template_list() const {
	List<StringName> names;
	get_scene_template_list(&names);
	names.sort_custom<StringName::AlphCompare>();
	Array ret;
	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

Array ClassType::_get_class_list() const {
	List<StringName> names;
	get_class_list(&names);
	names.sort_custom<StringName::AlphCompare>();
	Array ret;
	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

Array ClassType::_get_inheritors_list() const {
	List<StringName> names;
	get_inheritors_list(_name, &names);
	names.sort_custom<StringName::AlphCompare>();
	Array ret;
	for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

Array ClassType::_get_class_signal_list() const {
	List<MethodInfo> list;
	get_class_signal_list(&list);
	Array ret;
	for (List<MethodInfo>::Element *E = list.front(); E; E = E->next()) {
		ret.push_back(E->get().name);
	}
	return ret;
}

Array ClassType::_get_class_method_list() const {
	List<MethodInfo> list;
	get_class_method_list(&list);
	Array ret;
	for (List<MethodInfo>::Element *E = list.front(); E; E = E->next()) {
		ret.push_back(E->get().name);
	}
	return ret;
}

Array ClassType::_get_class_property_list() const {
	List<PropertyInfo> list;
	get_class_property_list(&list);
	Array ret;
	for (List<PropertyInfo>::Element *E = list.front(); E; E = E->next()) {
		ret.push_back(E->get().name);
	}
	return ret;
}

Array ClassType::_get_class_constant_list() const {
	List<String> list;
	get_class_constant_list(&list);
	Array ret;
	for (List<String>::Element *E = list.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

StringName ClassType::get_name_by_path(const String &p_path) {
	return _get_name_by_path(p_path, NULL);
}

StringName ClassType::get_name_by_object(const Object *p_object, String *r_path) {
	if (!p_object)
		return StringName();

	const Node *n = Object::cast_to<Node>(p_object);
	if (n && !n->get_filename().empty()) {
		if (r_path)
			*r_path = n->get_filename();
		return get_name_by_path(n->get_filename());
	}

	Ref<Script> s = p_object->get_script();
	if (s.is_null() && p_object->is_class("Script")) {
		s = p_object;
	}

	if (s.is_valid()) {
		if (r_path)
			*r_path = s->get_path();
		return get_name_by_path(s->get_path());
	}

	return p_object->get_class();
}

RES ClassType::get_scene_root_script(const String &p_path) {
	if (!is_valid_path(p_path))
		return NULL;
	Ref<PackedScene> scene = ResourceLoader::load(p_path, "PackedScene");
	if (scene.is_null())
		return NULL;
	Ref<SceneState> state = scene->get_state();
	if (state.is_null())
		return NULL;
	for (int i = 0; i < state->get_node_property_count(0); i++) {
		String name = state->get_node_property_name(0, i);
		if ("script" == name) {
			return state->get_node_property_value(0, i);
		}
	}
	return NULL;
}

RES ClassType::get_scene_root_scene(const String &p_path) {
	if (!is_valid_path(p_path))
		return NULL;
	Ref<PackedScene> scene = ResourceLoader::load(p_path, "PackedScene");
	if (scene.is_null())
		return NULL;
	Ref<SceneState> state = scene->get_state();
	if (state.is_null())
		return NULL;
	return state->get_node_instance(0);
}

void ClassType::set_name(const String &p_name) {
	_name = p_name;
	_source_dirty = true;
	_update_path_by_name();
}

String ClassType::get_name() const {
	return _name;
}

void ClassType::set_path(const String &p_path) {
	if (p_path == _path)
		return;
	_path = p_path;
	_source_dirty = true;
	_name = get_name_by_path(_path);
}

String ClassType::get_path() const {
	return _path;
}

bool ClassType::is_valid_path(const String &p_path) {
	return !p_path.empty() && FileAccess::exists(p_path);
}

bool ClassType::is_engine_class() const {
	return _source == SOURCE_ENGINE;
}

bool ClassType::is_script_class() const {
	return _source == SOURCE_SCRIPT;
}

bool ClassType::is_scene_template() const {
	return _source == SOURCE_SCENE;
}

bool ClassType::exists() const {
	return _source != SOURCE_NONE;
}

bool ClassType::is_non_class_res() const {
	return !_path.empty() && _name == StringName();
}

RES ClassType::get_res() const {
	if (!has_valid_path())
		return NULL;
	return ResourceLoader::load(_path);
}

RES ClassType::get_editor_icon() const {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() && exists())
		return EditorNode::get_singleton()->get_class_icon(_name);
#endif
	return NULL;
}

StringName ClassType::get_engine_parent() const {
	if (is_engine_class())
		return ClassDB::get_parent_class(_name);
	if (is_script_class())
		return ScriptServer::get_global_class_base(_name);
	if (is_scene_template()) {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
			Ref<PackedScene> scene = get_res();
			if (scene.is_null())
				return StringName();
			Ref<SceneState> state = scene->get_state();
			if (state.is_null())
				return StringName();
			return state->get_node_type(0);
		}
#endif
	}
	return StringName();
}

StringName ClassType::get_script_parent() const {
	if (is_engine_class())
		return StringName();
	if (is_script_class()) {

		Ref<Script> script = get_res();
		if (script.is_null())
			return StringName();

		script = script->get_base_script();
		while (script.is_valid()) {
			Ref<ClassType> base = ClassType::from_object(script.operator->());
			if (base->is_script_class())
				return base->get_name();
			script = script->get_base_script();
		}

		return get_engine_parent();
	}
	if (is_scene_template()) {
		Ref<Script> script = get_scene_root_script(_path);
		while (script.is_valid()) {
			StringName tmp = get_name_by_path(script->get_path());
			if (StringName() != tmp)
				return tmp;
			script = script->get_base_script();
		}
	}
	return StringName();
}

StringName ClassType::get_scene_parent() const {
	if (!is_scene_template())
		return StringName();
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		EditorSceneTemplateSettings *templates = EditorNode::get_singleton()->get_project_settings()->get_scene_template_settings();
		return templates->scene_template_exists(_name) ? templates->scene_template_get_base(_name) : StringName();
	}
#endif
	return StringName();
}

StringName ClassType::get_parent_class() const {
	StringName ret = get_scene_parent();
	if (StringName() != ret)
		return ret;
	else {
		ret = get_script_parent();
		if (StringName() != ret)
			return ret;
		else {
			ret = get_engine_parent();
			return ret;
		}
	}
}

Array ClassType::get_parents() const {
	Array ret;
	ret.resize(3);
	ret[0] = get_engine_parent();
	ret[1] = get_script_parent();
	ret[2] = get_scene_parent();
	return ret;
}

Ref<ClassType> ClassType::get_type_parent() const {
	switch (_source) {
		case SOURCE_ENGINE:
			return ClassType::from_name(get_engine_parent());
		case SOURCE_SCRIPT: {
			Ref<Script> script = get_res();
			script = script->get_base_script();
			if (script.is_valid())
				return ClassType::from_path(script->get_path());
			else
				return ClassType::from_name(get_engine_class());
		} break;
		case SOURCE_SCENE: {
			Ref<PackedScene> scene = get_scene_root_scene(_path);
			if (scene.is_valid())
				return ClassType::from_path(scene->get_path());
			else
				return ClassType::from_name(get_engine_class());
		} break;
		case SOURCE_NONE:
			return Ref<ClassType>();
	}
	return Ref<ClassType>();
}

RES ClassType::get_type_script() const {
	RES res = get_res();
	if (res.is_valid()) {
		Ref<Script> script = res;
		if (script.is_null())
			script = get_scene_root_script(_path);
		return script;
	}
	return res;
}

StringName ClassType::get_engine_class() const {
	if (has_valid_path())
		return get_engine_parent();
	else if (is_engine_class())
		return _name;
	return StringName();
}

bool ClassType::is_engine_parent(const StringName &p_class) const {
	StringName parent = get_engine_parent();
	return ClassDB::class_exists(parent) && ClassDB::is_parent_class(parent, p_class);
}

bool ClassType::is_script_parent(const StringName &p_class) const {
	if (is_engine_class())
		return false;

	Ref<Script> script;
	if (is_script_class())
		script = get_res();
	else if (is_scene_template())
		script = get_scene_root_script(_path);
	else if (is_non_class_res()) {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			List<String> exts;
			for (int i = 0; i < ScriptServer::get_language_count(); i++)
				ScriptServer::get_language(i)->get_recognized_extensions(&exts);
			if (exts.find(_path.get_extension()))
				script = get_res();
		}
	}

	while (script.is_valid()) {
		Ref<ClassType> script_type = ClassType::from_object(script.operator->());
		if (script_type->is_script_class() && script_type->get_name() == p_class)
			return true;
		script = script->get_base_script();
	}
	return false;
}

bool ClassType::is_scene_parent(const StringName &p_class) const {
	if (is_engine_class() || is_script_class())
		return false;
	Ref<PackedScene> scene = get_scene_root_scene(_path);
	while (scene.is_valid()) {
		Ref<ClassType> scene_type = ClassType::from_path(scene->get_path());
		if (scene_type->is_scene_template() && scene_type->get_name() == p_class)
			return true;
		scene = get_scene_root_scene(scene->get_path());
	}
	return false;
}

bool ClassType::is_parent_class(const StringName &p_class) const {
	return is_engine_parent(p_class) || is_script_parent(p_class) || is_scene_parent(p_class);
}

bool ClassType::can_instance() const {
	if (is_engine_class())
		return ClassDB::can_instance(_name);
	StringName base = get_engine_parent();
	if (StringName() != base)
		return ClassDB::can_instance(base);
	return false;
}

Object *ClassType::instance() const {
	if (is_engine_class())
		return ClassDB::instance(_name);
	if (!has_valid_path())
		return NULL;
	if (is_script_class()) {
		RES res = get_res();
		Object *obj = ClassType::from_name(get_engine_parent())->instance();
		if (!obj)
			return NULL;
		obj->set_script(res.get_ref_ptr());
		return obj;
	}
	if (is_scene_template()) {
		Ref<PackedScene> scene = get_res();
		if (scene.is_null())
			return NULL;
		return scene->instance();
	}
	if (is_non_class_res()) {
		RES res = ResourceLoader::load(_path);
		Ref<Script> script = res;
		if (script.is_valid()) {
			StringName base = script->get_instance_base_type();
			if (StringName() == base)
				return NULL;
			Object *obj = ClassDB::instance(base);
			obj->set_script(script.get_ref_ptr());
		}
		Ref<PackedScene> scene = res;
		if (scene.is_valid()) {
			return scene->instance();
		}
	}
	return NULL;
}

void ClassType::get_class_list(List<StringName> *p_list) {
	get_engine_class_list(p_list);
	get_script_class_list(p_list);
	get_scene_template_list(p_list);
}

void ClassType::get_engine_class_list(List<StringName> *p_list) {
	ClassDB::get_class_list(p_list);
}

void ClassType::get_script_class_list(List<StringName> *p_list) {
	ScriptServer::get_global_class_list(p_list);
}

void ClassType::get_scene_template_list(List<StringName> *p_list) {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint())
		EditorNode::get_singleton()->get_project_settings()->get_scene_template_settings()->get_scene_template_list(p_list);
#endif
}

void ClassType::get_inheritors_list(const StringName &p_class, List<StringName> *p_list) {
	Ref<ClassType> type = ClassType::from_name(p_class);
	if (type->is_engine_class()) {
		List<StringName> names;
		ClassDB::get_class_list(&names);
		for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
			if (ClassDB::is_parent_class(E->get(), p_class))
				p_list->push_back(E->get());
		}
	}
	RES res = type->get_res();
	Ref<Script> script = res;
	if (script.is_valid()) {
		Ref<ClassType> ancestor_script = ClassType::from_name(type->get_script_parent());
		if (ancestor_script.is_valid() && ancestor_script->has_valid_path()) {
			List<StringName> names;
			ScriptServer::get_global_class_list(&names);
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				Ref<ClassType> script_type = ClassType::from_name(E->get());
				if (script_type->is_parent_class(p_class))
					p_list->push_back(E->get());
			}
		}
	}
	Ref<PackedScene> scene = res;
	if (scene.is_valid()) {
		Ref<ClassType> ancestor_scene = ClassType::from_name(type->get_scene_parent());
		if (ancestor_scene.is_valid() && ancestor_scene->has_valid_path()) {
			List<StringName> names;
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint())
				EditorNode::get_singleton()->get_project_settings()->get_scene_template_settings()->get_scene_template_list(p_list);
#endif
			for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
				Ref<ClassType> scene_type = ClassType::from_name(E->get());
				if (scene_type->is_parent_class(p_class))
					p_list->push_back(E->get());
			}
		}
	}
}

bool ClassType::class_has_signal(const StringName &p_signal) const {
	Ref<Script> script = get_type_script();
	if (script.is_valid() && script->has_script_signal(p_signal))
		return true;
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return false;
	return ClassDB::has_signal(base, p_signal);
}

bool ClassType::class_has_method(const StringName &p_method) const {
	Ref<Script> script = get_type_script();
	if (script.is_valid() && script->has_method(p_method))
		return true;
	StringName base = _name;
	if (!is_engine_class())
		base = get_engine_parent();
	return ClassDB::has_method(base, p_method);
}

bool ClassType::class_has_property(const StringName &p_property) const {
	Ref<Script> script = get_type_script();
	if (script.is_valid()) {
		List<PropertyInfo> plist;
		script->get_script_property_list(&plist);
		for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {
			if (E->get().name == p_property)
				return true;
		}
	}
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return false;
	return ClassDB::has_property(base, p_property);
}

bool ClassType::class_has_constant(const StringName &p_constant) const {
	Ref<Script> script = get_type_script();
	if (script.is_valid()) {
		Map<StringName, Variant> map;
		script->get_constants(&map);
		if (map.has(p_constant))
			return true;
	}
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return false;
	List<String> list;
	ClassDB::get_integer_constant_list(base, &list);
	return list.find(p_constant);
}

void ClassType::get_class_signal_list(List<MethodInfo> *p_signals) const {
	ERR_FAIL_COND(!p_signals);
	Ref<Script> script = get_type_script();
	if (script.is_valid()) {
		script->get_script_signal_list(p_signals);
	}
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return;
	ClassDB::get_signal_list(base, p_signals);
}

void ClassType::get_class_method_list(List<MethodInfo> *p_methods) const {
	ERR_FAIL_COND(!p_methods);
	Ref<Script> script = get_type_script();
	if (script.is_valid()) {
		script->get_script_method_list(p_methods);
	}
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return;
	ClassDB::get_method_list(base, p_methods);
}

void ClassType::get_class_property_list(List<PropertyInfo> *p_properties) const {
	ERR_FAIL_COND(!p_properties);
	Ref<Script> script = get_type_script();
	if (script.is_valid()) {
		script->get_script_property_list(p_properties);
	}
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return;
	ClassDB::get_property_list(base, p_properties);
}

void ClassType::get_class_constant_list(List<String> *p_constants) const {
	ERR_FAIL_COND(!p_constants);
	Ref<Script> script = get_type_script();
	if (script.is_valid()) {
		Map<StringName, Variant> map;
		script->get_constants(&map);
		for (Map<StringName, Variant>::Element *E = map.front(); E; E = E->next()) {
			p_constants->push_back(E->key());
		}
	}
	StringName base = get_engine_class();
	if (!ClassDB::class_exists(base))
		return;
	ClassDB::get_integer_constant_list(base, p_constants);
}

void ClassType::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &ClassType::get_name);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &ClassType::set_name);
	ClassDB::bind_method(D_METHOD("get_path"), &ClassType::get_path);
	ClassDB::bind_method(D_METHOD("set_path", "path"), &ClassType::set_path);
	ClassDB::bind_method(D_METHOD("has_valid_path"), &ClassType::has_valid_path);

	ClassDB::bind_method(D_METHOD("is_engine_class"), &ClassType::is_engine_class);
	ClassDB::bind_method(D_METHOD("is_script_class"), &ClassType::is_script_class);
	ClassDB::bind_method(D_METHOD("is_scene_template"), &ClassType::is_scene_template);
	ClassDB::bind_method(D_METHOD("exists"), &ClassType::exists);
	ClassDB::bind_method(D_METHOD("is_non_class_res"), &ClassType::is_non_class_res);

	ClassDB::bind_method(D_METHOD("get_res"), &ClassType::get_res);
	ClassDB::bind_method(D_METHOD("get_editor_icon"), &ClassType::get_editor_icon);

	ClassDB::bind_method(D_METHOD("get_engine_parent"), &ClassType::get_engine_parent);
	ClassDB::bind_method(D_METHOD("get_script_parent"), &ClassType::get_script_parent);
	ClassDB::bind_method(D_METHOD("get_scene_parent"), &ClassType::get_scene_parent);
	ClassDB::bind_method(D_METHOD("get_parent_class"), &ClassType::get_parent_class);
	ClassDB::bind_method(D_METHOD("get_parents"), &ClassType::get_parents);

	ClassDB::bind_method(D_METHOD("get_type_script"), &ClassType::get_type_script);
	ClassDB::bind_method(D_METHOD("get_engine_class"), &ClassType::get_engine_class);

	ClassDB::bind_method(D_METHOD("is_engine_parent", "p_class"), &ClassType::is_engine_parent);
	ClassDB::bind_method(D_METHOD("is_script_parent", "p_class"), &ClassType::is_script_parent);
	ClassDB::bind_method(D_METHOD("is_scene_parent", "p_class"), &ClassType::is_scene_parent);
	ClassDB::bind_method(D_METHOD("is_parent_class", "p_class"), &ClassType::is_parent_class);

	ClassDB::bind_method(D_METHOD("get_engine_class_list"), &ClassType::_get_engine_class_list);
	ClassDB::bind_method(D_METHOD("get_script_class_list"), &ClassType::_get_script_class_list);
	ClassDB::bind_method(D_METHOD("get_scene_template_list"), &ClassType::_get_scene_template_list);
	ClassDB::bind_method(D_METHOD("get_class_list"), &ClassType::_get_class_list);

	ClassDB::bind_method(D_METHOD("get_inheritors_list"), &ClassType::_get_inheritors_list);

	ClassDB::bind_method(D_METHOD("instance"), &ClassType::instance);

	ClassDB::bind_method(D_METHOD("class_has_signal"), &ClassType::class_has_signal);
	ClassDB::bind_method(D_METHOD("class_has_method"), &ClassType::class_has_method);
	ClassDB::bind_method(D_METHOD("class_has_property"), &ClassType::class_has_property);
	ClassDB::bind_method(D_METHOD("class_has_constant"), &ClassType::class_has_constant);

	ClassDB::bind_method(D_METHOD("get_class_signal_list"), &ClassType::_get_class_signal_list);
	ClassDB::bind_method(D_METHOD("get_class_method_list"), &ClassType::_get_class_method_list);
	ClassDB::bind_method(D_METHOD("get_class_property_list"), &ClassType::_get_class_property_list);
	ClassDB::bind_method(D_METHOD("get_class_constant_list"), &ClassType::_get_class_constant_list);

	ClassDB::bind_method(D_METHOD("become", "p_object"), &ClassType::_become);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_name", "get_name");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "path"), "set_path", "get_path");
}

Ref<ClassType> ClassType::from_name(const StringName &p_name) {
	Ref<ClassType> ret = memnew(ClassType);
	ret->set_name(p_name);
	return ret;
}

Ref<ClassType> ClassType::from_path(const String &p_path) {
	Ref<ClassType> ret = memnew(ClassType);
	ret->set_path(p_path);
	return ret;
}

Ref<ClassType> ClassType::from_object(const Object *p_object) {
	Ref<ClassType> ret = memnew(ClassType);
	ret->set_name(get_name_by_object(p_object));
	return ret;
}

ClassType::ClassType() {
	_name = StringName();
	_path = String();
	_source = SOURCE_NONE;
	_source_dirty = false;
}
