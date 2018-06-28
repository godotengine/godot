/*************************************************************************/
/*  editor_data.cpp                                                      */
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

#include "editor_data.h"

#include "editor_node.h"
#include "editor_settings.h"
#include "io/resource_loader.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "project_settings.h"
#include "scene/resources/packed_scene.h"

#include "../modules/gdscript/gdscript.h"
#include "../modules/regex/regex.h"

void EditorHistory::cleanup_history() {

	for (int i = 0; i < history.size(); i++) {

		bool fail = false;

		for (int j = 0; j < history[i].path.size(); j++) {
			if (!history[i].path[j].ref.is_null())
				continue;

			Object *obj = ObjectDB::get_instance(history[i].path[j].object);
			if (obj) {
				Node *n = Object::cast_to<Node>(obj);
				if (n && n->is_inside_tree())
					continue;
				if (!n) // Possibly still alive
					continue;
			}

			if (j <= history[i].level) {
				//before or equal level, complete fail
				fail = true;
			} else {
				//after level, clip
				history[i].path.resize(j);
			}

			break;
		}

		if (fail) {
			history.remove(i);
			i--;
		}
	}

	if (current >= history.size())
		current = history.size() - 1;
}

void EditorHistory::_add_object(ObjectID p_object, const String &p_property, int p_level_change, bool p_inspector_only) {

	Object *obj = ObjectDB::get_instance(p_object);
	ERR_FAIL_COND(!obj);
	Reference *r = Object::cast_to<Reference>(obj);
	Obj o;
	if (r)
		o.ref = REF(r);
	o.object = p_object;
	o.property = p_property;
	o.inspector_only = p_inspector_only;

	History h;

	bool has_prev = current >= 0 && current < history.size();

	if (has_prev) {
		history.resize(current + 1); //clip history to next
	}

	if (p_property != "" && has_prev) {
		//add a sub property
		History &pr = history[current];
		h = pr;
		h.path.resize(h.level + 1);
		h.path.push_back(o);
		h.level++;
	} else if (p_level_change != -1 && has_prev) {
		//add a sub property
		History &pr = history[current];
		h = pr;
		ERR_FAIL_INDEX(p_level_change, h.path.size());
		h.level = p_level_change;
	} else {
		//add a new node
		h.path.push_back(o);
		h.level = 0;
	}

	history.push_back(h);
	current++;
}

void EditorHistory::add_object_inspector_only(ObjectID p_object) {

	_add_object(p_object, "", -1, true);
}

void EditorHistory::add_object(ObjectID p_object) {

	_add_object(p_object, "", -1);
}

void EditorHistory::add_object(ObjectID p_object, const String &p_subprop) {

	_add_object(p_object, p_subprop, -1);
}

void EditorHistory::add_object(ObjectID p_object, int p_relevel) {

	_add_object(p_object, "", p_relevel);
}

int EditorHistory::get_history_len() {
	return history.size();
}
int EditorHistory::get_history_pos() {
	return current;
}

bool EditorHistory::is_history_obj_inspector_only(int p_obj) const {

	ERR_FAIL_INDEX_V(p_obj, history.size(), false);
	ERR_FAIL_INDEX_V(history[p_obj].level, history[p_obj].path.size(), false);
	return history[p_obj].path[history[p_obj].level].inspector_only;
}

ObjectID EditorHistory::get_history_obj(int p_obj) const {
	ERR_FAIL_INDEX_V(p_obj, history.size(), 0);
	ERR_FAIL_INDEX_V(history[p_obj].level, history[p_obj].path.size(), 0);
	return history[p_obj].path[history[p_obj].level].object;
}

bool EditorHistory::is_at_beginning() const {
	return current <= 0;
}
bool EditorHistory::is_at_end() const {

	return ((current + 1) >= history.size());
}

bool EditorHistory::next() {

	cleanup_history();

	if ((current + 1) < history.size())
		current++;
	else
		return false;

	return true;
}

bool EditorHistory::previous() {

	cleanup_history();

	if (current > 0)
		current--;
	else
		return false;

	return true;
}

bool EditorHistory::is_current_inspector_only() const {

	if (current < 0 || current >= history.size())
		return false;

	const History &h = history[current];
	return h.path[h.level].inspector_only;
}
ObjectID EditorHistory::get_current() {

	if (current < 0 || current >= history.size())
		return 0;

	History &h = history[current];
	Object *obj = ObjectDB::get_instance(h.path[h.level].object);
	if (!obj)
		return 0;

	return obj->get_instance_id();
}

int EditorHistory::get_path_size() const {

	if (current < 0 || current >= history.size())
		return 0;

	const History &h = history[current];
	return h.path.size();
}

ObjectID EditorHistory::get_path_object(int p_index) const {

	if (current < 0 || current >= history.size())
		return 0;

	const History &h = history[current];

	ERR_FAIL_INDEX_V(p_index, h.path.size(), 0);

	Object *obj = ObjectDB::get_instance(h.path[p_index].object);
	if (!obj)
		return 0;

	return obj->get_instance_id();
}

String EditorHistory::get_path_property(int p_index) const {

	if (current < 0 || current >= history.size())
		return "";

	const History &h = history[current];

	ERR_FAIL_INDEX_V(p_index, h.path.size(), "");

	return h.path[p_index].property;
}

void EditorHistory::clear() {

	history.clear();
	current = -1;
}

EditorHistory::EditorHistory() {

	current = -1;
}

EditorPlugin *EditorData::get_editor(Object *p_object) {

	for (int i = 0; i < editor_plugins.size(); i++) {

		if (editor_plugins[i]->has_main_screen() && editor_plugins[i]->handles(p_object))
			return editor_plugins[i];
	}

	return NULL;
}

EditorPlugin *EditorData::get_subeditor(Object *p_object) {

	for (int i = 0; i < editor_plugins.size(); i++) {

		if (!editor_plugins[i]->has_main_screen() && editor_plugins[i]->handles(p_object))
			return editor_plugins[i];
	}

	return NULL;
}

Vector<EditorPlugin *> EditorData::get_subeditors(Object *p_object) {
	Vector<EditorPlugin *> sub_plugins;
	for (int i = 0; i < editor_plugins.size(); i++) {
		if (!editor_plugins[i]->has_main_screen() && editor_plugins[i]->handles(p_object)) {
			sub_plugins.push_back(editor_plugins[i]);
		}
	}
	return sub_plugins;
}

EditorPlugin *EditorData::get_editor(String p_name) {

	for (int i = 0; i < editor_plugins.size(); i++) {

		if (editor_plugins[i]->get_name() == p_name)
			return editor_plugins[i];
	}

	return NULL;
}

void EditorData::copy_object_params(Object *p_object) {

	clipboard.clear();

	List<PropertyInfo> pinfo;
	p_object->get_property_list(&pinfo);

	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_EDITOR))
			continue;

		PropertyData pd;
		pd.name = E->get().name;
		pd.value = p_object->get(pd.name);
		clipboard.push_back(pd);
	}
}

void EditorData::get_editor_breakpoints(List<String> *p_breakpoints) {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->get_breakpoints(p_breakpoints);
	}
}

Dictionary EditorData::get_editor_states() const {

	Dictionary metadata;
	for (int i = 0; i < editor_plugins.size(); i++) {

		Dictionary state = editor_plugins[i]->get_state();
		if (state.empty())
			continue;
		metadata[editor_plugins[i]->get_name()] = state;
	}

	return metadata;
}

Dictionary EditorData::get_scene_editor_states(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), Dictionary());
	EditedScene es = edited_scene[p_idx];
	return es.editor_states;
}

void EditorData::set_editor_states(const Dictionary &p_states) {

	List<Variant> keys;
	p_states.get_key_list(&keys);

	List<Variant>::Element *E = keys.front();
	for (; E; E = E->next()) {

		String name = E->get();
		int idx = -1;
		for (int i = 0; i < editor_plugins.size(); i++) {

			if (editor_plugins[i]->get_name() == name) {
				idx = i;
				break;
			}
		}

		if (idx == -1)
			continue;
		editor_plugins[idx]->set_state(p_states[name]);
	}
}

void EditorData::notify_edited_scene_changed() {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->edited_scene_changed();
		editor_plugins[i]->notify_scene_changed(get_edited_scene_root());
	}
}

void EditorData::notify_resource_saved(const Ref<Resource> &p_resource) {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->notify_resource_saved(p_resource);
	}
}

void EditorData::clear_editor_states() {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->clear();
	}
}

void EditorData::save_editor_external_data() {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->save_external_data();
	}
}

void EditorData::apply_changes_in_editors() {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->apply_changes();
	}
}

void EditorData::save_editor_global_states() {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->save_global_state();
	}
}

void EditorData::restore_editor_global_states() {

	for (int i = 0; i < editor_plugins.size(); i++) {

		editor_plugins[i]->restore_global_state();
	}
}

void EditorData::paste_object_params(Object *p_object) {

	for (List<PropertyData>::Element *E = clipboard.front(); E; E = E->next()) {

		p_object->set(E->get().name, E->get().value);
	}
}

bool EditorData::call_build() {

	bool result = true;

	for (int i = 0; i < editor_plugins.size() && result; i++) {

		result &= editor_plugins[i]->build();
	}

	return result;
}

UndoRedo &EditorData::get_undo_redo() {

	return undo_redo;
}

void EditorData::remove_editor_plugin(EditorPlugin *p_plugin) {

	p_plugin->undo_redo = NULL;
	editor_plugins.erase(p_plugin);
}

void EditorData::add_editor_plugin(EditorPlugin *p_plugin) {

	p_plugin->undo_redo = &undo_redo;
	editor_plugins.push_back(p_plugin);
}

int EditorData::get_editor_plugin_count() const {
	return editor_plugins.size();
}

EditorPlugin *EditorData::get_editor_plugin(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, editor_plugins.size(), NULL);
	return editor_plugins[p_idx];
}

void EditorData::TypeDB::_add_type(const StringName &p_type, const StringName &p_inherits, const String &p_path, const Ref<Texture> &p_icon, bool p_is_abstract, bool p_is_custom, TypeDbTypes p_db_type) {

	ERR_FAIL_COND(_scripts.has(p_type));
	ERR_FAIL_COND(!FileAccess::exists(p_path));

	TypeInfo ti;
	ti.type_name = p_type;
	ti.inherits = p_inherits;
	ti.icon = p_icon;
	ti.is_abstract = p_is_abstract;
	ti.is_custom = p_is_custom;
	ti.type = p_db_type;
	ti.path = String(p_path);
	ti.disabled = false;

	TypeMap *tm;
	DependencyMap *dm;
	switch (p_db_type) {
		case TYPEDB_TYPE_SCRIPT:
			tm = &_scripts;
			dm = &_script_deps;
			break;
		case TYPEDB_TYPE_SCENE:
			tm = &_scenes;
			dm = &_scene_deps;
			break;
	}

	if (!tm->has(p_inherits) && !ClassDB::class_exists(p_inherits)) {
		if (!dm->has(p_inherits))
			(*dm)[p_inherits] = Vector<TypeInfo>();
		(*dm)[p_inherits].push_back(ti);
		return;
	}
	if (tm->has(ti.inherits)) {
		TypeInfo &bti = (*tm)[ti.inherits];
		ti.base = bti.base;
		bti.sub_types.push_back(ti.type_name);
	} else {
		ti.base = ti.inherits;
	}

	_register_type_info(ti, p_db_type);

	_check_for_deps(ti.type_name, p_db_type);
}

void EditorData::TypeDB::_register_type_info(TypeInfo &p_ti, TypeDbTypes p_db_type) {

	TypeMap *tm;
	switch (p_db_type) {
		case TYPEDB_TYPE_SCRIPT:
			tm = &_scripts;
			break;
		case TYPEDB_TYPE_SCENE:
			tm = &_scenes;
			break;
	}
	(*tm)[p_ti.type_name] = p_ti;
	_paths[p_ti.path] = &(*tm)[p_ti.type_name];

	_globals_dirty = true;
}

void EditorData::TypeDB::_check_for_deps(const StringName &p_type, TypeDbTypes p_db_type) {

	TypeMap *tm;
	DependencyMap *dm;
	switch (p_db_type) {
		case TYPEDB_TYPE_SCRIPT:
			tm = &_scripts;
			dm = &_script_deps;
			break;
		case TYPEDB_TYPE_SCENE:
			tm = &_scenes;
			dm = &_scene_deps;
			break;
	}

	if (!dm->has(p_type))
		return;

	TypeInfo &ti = (*tm)[p_type];
	Vector<TypeInfo> &tis = (*dm)[p_type];

	for (int i = 0; i < tis.size(); i++) {
		TypeInfo &dep = tis[i];

		dep.base = ti.base;
		ti.sub_types.push_back(dep.type_name);
		_register_type_info(dep);
		_check_for_deps(dep.type_name);
	}

	dm->erase(p_type);
}

void EditorData::TypeDB::add_script(const StringName &p_type, const StringName &p_inherits, const Ref<Script> &p_script, const Ref<Texture> &p_icon, bool p_is_abstract, bool p_is_custom) {
	ERR_FAIL_COND(!p_script.is_valid());
	_add_type(p_type, p_inherits, p_script->get_path(), p_icon, p_is_abstract, p_is_custom, TYPEDB_TYPE_SCRIPT);
}

void EditorData::TypeDB::add_scene(const StringName &p_type, const StringName &p_inherits, const Ref<PackedScene> &p_scene, const Ref<Texture> &p_icon, bool p_is_abstract, bool p_is_custom) {
	ERR_FAIL_COND(!p_scene.is_valid());
	_add_type(p_type, p_inherits, p_scene->get_path(), p_icon, p_is_abstract, p_is_custom, TYPEDB_TYPE_SCENE);
}

void EditorData::TypeDB::remove_custom_type(const StringName &p_type, TypeDbTypes p_db_types) {
	TypeMap *tm;
	switch (p_db_types) {
		case TYPEDB_TYPE_SCRIPT:
			tm = &_scripts;
			break;
		case TYPEDB_TYPE_SCENE:
			tm = &_scenes;
			break;
	}

	TypeInfo &ti = (*tm)[p_type];
	(*tm)[ti.inherits].sub_types.erase(ti.type_name);
	tm->erase(p_type);
}

bool EditorData::TypeDB::class_exists(const StringName &p_type, TypeDbTypes p_db_types) const {
	if (p_db_types & TYPEDB_TYPE_SCRIPT && _scripts.has(p_type))
		return !_scripts[p_type].disabled;
	if (p_db_types & TYPEDB_TYPE_SCENE && _scenes.has(p_type))
		return !_scenes[p_type].disabled;
	return false;
}

bool EditorData::TypeDB::can_instance(const StringName &p_type, TypeDbTypes p_db_types) const {

	if (p_db_types & TYPEDB_TYPE_SCRIPT && _scripts.has(p_type)) {
		const EditorData::TypeInfo &ti = _scripts[p_type];
		if (ti.is_abstract || ti.disabled) {
			return false;
		}
		RES res = ResourceLoader::load(_scripts[p_type].path);
		if (res->has_method("can_instance")) {
			bool script_can_instance = res->call("can_instance").operator bool();
			if (!script_can_instance) {
				return false;
			}
		}
		return true;
	}

	if (p_db_types & TYPEDB_TYPE_SCENE && _scenes.has(p_type)) {
		const EditorData::TypeInfo &ti = _scenes[p_type];
		return !(ti.is_abstract || ti.disabled);
	}

	return false;
}

Object *EditorData::TypeDB::instance(const StringName &p_type, TypeDbTypes p_db_type) {
	ERR_FAIL_COND_V(!class_exists(p_type, p_db_type), NULL);

	const EditorData::TypeInfo *ti = get_type_info(p_type, p_db_type);
	ERR_FAIL_COND_V(!ti, NULL);
	ERR_FAIL_COND_V(ti->type != p_db_type, NULL);
	ERR_FAIL_COND_V(ti->is_abstract, NULL);
	ERR_FAIL_COND_V(ti->disabled, NULL);

	if (ti->type == TYPEDB_TYPE_SCRIPT) {
		Object *obj = ClassDB::instance(ti->base);
		ERR_FAIL_COND_V(!obj, NULL);

		Ref<Script> script = ResourceLoader::load(ti->path, "Script");
		ERR_FAIL_COND_V(script.is_null(), NULL);

		obj->set_script(script.get_ref_ptr());
		return obj;
	} else if (ti->type == TYPEDB_TYPE_SCENE) {
		Ref<PackedScene> scene = ResourceLoader::load(ti->path, "PackedScene");
		ERR_FAIL_COND_V(scene.is_null(), NULL);

		Object *obj = scene->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
		ERR_FAIL_COND_V(!obj, NULL);

		return obj;
	}
	return NULL;
}

bool EditorData::TypeDB::is_custom(const StringName &p_type, TypeDbTypes p_db_types) const {

	if (p_db_types & TYPEDB_TYPE_SCRIPT && _scripts.has(p_type)) {
		const EditorData::TypeInfo &ti = _scripts[p_type];
		return ti.is_custom && !ti.disabled;
	}
	if (p_db_types & TYPEDB_TYPE_SCENE && _scenes.has(p_type)) {
		const EditorData::TypeInfo &ti = _scenes[p_type];
		return ti.is_custom && !ti.disabled;
	}
	return false;
}

bool EditorData::TypeDB::is_custom_path(const String &p_path, TypeDbTypes p_db_types) const {
	EditorData::TypeInfo *ti = _paths[p_path];
	if (!ti)
		return false;
	return is_custom(ti->type_name, p_db_types);
}

StringName EditorData::TypeDB::get_type_name(const String &p_path, EditorData::TypeDbTypes p_db_types) const {

	if (p_db_types == TYPEDB_TYPE_SCRIPT) {
		Ref<Script> script = ResourceLoader::load(p_path, "Script");
		Dictionary meta = script->get_script_metadata();
		if (!meta.empty() && meta.has("type_name")) {
			return meta["type_name"];
		}
	}
	//TODO: scenes metadata extraction
	return StringName();
}

bool EditorData::TypeDB::is_parent_class(const StringName &p_type, const StringName &p_inherits, TypeDbTypes p_db_types) const {

	StringName inherits = p_type;

	while (inherits.operator String().length()) {

		if (inherits == p_inherits)
			return true;
		inherits = get_parent_class(inherits, p_db_types);
	}

	const EditorData::TypeInfo *ti = get_type_info(p_type, p_db_types);
	if (ti) {
		return ti->base == p_inherits || ClassDB::is_parent_class(ti->base, p_inherits);
	}

	return false;
}

StringName EditorData::TypeDB::get_parent_class(const StringName &p_type, TypeDbTypes p_db_types) const {
	if (p_db_types & TYPEDB_TYPE_SCRIPT)
		return _scripts.has(p_type) && !_scripts[p_type].disabled ? _scripts[p_type].inherits : StringName();
	if (p_db_types & TYPEDB_TYPE_SCENE)
		return _scenes.has(p_type) && !_scenes[p_type].disabled ? _scenes[p_type].inherits : StringName();
	return StringName();
}

void EditorData::TypeDB::get_class_list(List<StringName> *p_types, TypeDbTypes p_db_types) const {

	if (p_db_types & TYPEDB_TYPE_SCRIPT) {
		const StringName *k = NULL;

		while ((k = _scripts.next(k))) {
			if (_scripts[*k].disabled)
				continue;
			p_types->push_back(*k);
		}
	}

	if (p_db_types & TYPEDB_TYPE_SCENE) {
		const StringName *k = NULL;

		while ((k = _scenes.next(k))) {
			if (_scenes[*k].disabled)
				continue;
			p_types->push_back(*k);
		}
	}

	p_types->sort();
}

const EditorData::TypeInfo *EditorData::TypeDB::get_type_info(const StringName &p_type, TypeDbTypes p_db_types) const {
	if (p_db_types & TYPEDB_TYPE_SCRIPT && _scripts.has(p_type) && !_scripts[p_type].disabled)
		return &_scripts[p_type];
	if (p_db_types & TYPEDB_TYPE_SCENE && _scenes.has(p_type) && !_scenes[p_type].disabled)
		return &_scenes[p_type];
	return NULL;
}

void EditorData::TypeDB::toggle_namespace(const String &p_namespace, bool p_active) {
	List<StringName> keys;
	_scripts.get_key_list(&keys);
	for (List<StringName>::Element *E = keys.front(); E; E = E->next()) {
		if (String(E->get()).find(p_namespace + ".", 0) == 0)
			_scripts[E->get()].disabled = !p_active;
	}
	keys.clear();
	_scenes.get_key_list(&keys);
	for (List<StringName>::Element *E = keys.front(); E; E = E->next()) {
		if (String(E->get()).find(p_namespace + ".", 0) == 0)
			_scenes[E->get()].disabled = !p_active;
	}
	_globals_dirty = true;
}

void EditorData::TypeDB::toggle_directory(const String &p_directory, bool p_active) {
	List<StringName> keys;
	_paths.get_key_list(&keys);
	for (List<StringName>::Element *E = keys.front(); E; E = E->next()) {
		if (String(E->get()).find(p_directory, 0) == 0)
			_paths[E->get()]->disabled = !p_active;
	}
	_globals_dirty = true;
}

void EditorData::TypeDB::update_res(const String &p_path) {

	TypeDbTypes t = TYPEDB_TYPE_NONE;
	String ext = p_path.get_extension();
	if ("tscn" == ext || "scn" == ext) {
		t = TYPEDB_TYPE_SCENE;
	} else {
		for (int i = 0; i < ScriptServer::get_language_count(); i++) {
			ScriptLanguage *lang = ScriptServer::get_language(i);
			if (ext == lang->get_extension()) {
				t = TYPEDB_TYPE_SCRIPT;
				break;
			}
		}
	}

	if (_paths.has(p_path) && !FileAccess::exists(p_path)) {
		_paths.erase(p_path); // will this situation even happen if it's already in the ResourceCache?
	}

	StringName type_name = get_type_name(p_path, t);
	if (StringName() == type_name) {
		if (_paths.has(p_path)) {
			EditorData::TypeInfo *ti = _paths[p_path];
			if (ti) {
				if (ti->type == TYPEDB_TYPE_SCRIPT)
					_scripts.erase(ti->type_name);
				else if (ti->type == TYPEDB_TYPE_SCENE)
					_scenes.erase(ti->type_name);
			}
			_paths.erase(p_path);
		}
		return;
	}

	if (_paths.has(p_path)) {

		EditorData::TypeInfo *ti = _paths[p_path];
		if (ti->type == TYPEDB_TYPE_SCRIPT) {
			Ref<Script> script;
			script = ResourceLoader::load(ti->path, "Script");

			Dictionary meta = script->get_script_metadata();
			if (!meta.empty()) {

				StringName script_type_name = meta["type_name"];
				StringName base_name = meta["base_name"].operator StringName();

				String icon_path = meta["icon_path"];
				Ref<Texture> icon = FileAccess::exists(String(meta["icon_path"])) ? ResourceLoader::load(icon_path, "Texture") : NULL;

				bool is_abstract = meta.has("is_abstract") ? meta["is_abstract"].operator bool() : false;
				bool is_custom = meta.has("is_custom") ? meta["is_custom"].operator bool() : false;

				if (base_name == StringName()) {
					Ref<Script> base_script = script->get_base_script();
					if (base_script.is_valid() && _paths.has(base_script->get_path())) {
						EditorData::TypeInfo *ti = _paths[base_script->get_path()];
						if (ti)
							base_name = ti->type_name;
					}
				}

				ti->type_name = script_type_name;
				ti->inherits = base_name;
				ti->icon = icon;
				ti->is_abstract = is_abstract;
				ti->is_custom = is_custom;
			}
			_globals_dirty = true;
		}
	} else {
		const EditorData::TypeInfo *cti = get_type_info(type_name, t);
		if (!cti) {

			if (t == TYPEDB_TYPE_SCRIPT) {
				Ref<Script> script = ResourceLoader::load(p_path, "Script");
				if (script.is_valid()) {

					Dictionary meta = script->get_script_metadata();
					if (!meta.empty()) {

						StringName script_type_name = meta["type_name"];
						StringName base_name = meta["base_name"].get_type() == Variant::NIL ? StringName() : meta["base_name"].operator StringName();
						if (String(base_name).find("Scripts.", 0) == 0) {
							base_name = StringName(String(base_name).replace("Scripts.", ""));
						}

						String icon_path = meta["icon_path"];
						Ref<Texture> icon = FileAccess::exists(String(meta["icon_path"])) ? ResourceLoader::load(icon_path, "Texture") : NULL;

						bool is_abstract = meta.has("is_abstract") ? meta["is_abstract"].operator bool() : false;
						bool is_custom = meta.has("is_custom") ? meta["is_custom"].operator bool() : false;

						if (base_name == StringName()) {
							String base_path = meta["base_path"];
							Ref<Script> base_script = script->get_base_script();
							if (base_script.is_valid() && _paths.has(base_script->get_path())) {
								EditorData::TypeInfo *ti = _paths[base_script->get_path()];
								if (ti)
									base_name = ti->type_name;
							}
						}

						_add_type(script_type_name, base_name, p_path, icon, is_abstract, is_custom, TYPEDB_TYPE_SCRIPT);
					}
				}
			}
		} else {
			String original_path = cti->path;
			if (_paths.has(original_path)) {
				EditorData::TypeInfo *ti = _paths[original_path];
				ti->path = p_path;
				_paths.erase(original_path);
				_paths[p_path] = ti;
				_globals_dirty = true;
			}
		}
	}
}

void EditorData::TypeDB::update_globals() {
	if (!_globals_dirty)
		return;

	Dictionary scripts;
	List<StringName> keys;
	_scripts.get_key_list(&keys);
	for (List<StringName>::Element *E = keys.front(); E; E = E->next()) {
		EditorData::TypeInfo &ti = _scripts[E->get()];
		if (ti.disabled)
			continue;
		scripts[E->get()] = ti.path;
	}
	ProjectSettings::get_singleton()->set_setting("typedb/scripts", scripts);

	keys.clear();

	Dictionary scenes;
	_scenes.get_key_list(&keys);
	for (List<StringName>::Element *E = keys.front(); E; E = E->next()) {
		EditorData::TypeInfo &ti = _scenes[E->get()];
		if (ti.disabled)
			continue;
		scenes[E->get()] = ti.path;
	}
	ProjectSettings::get_singleton()->set_setting("typedb/scenes", scenes);
	keys.clear();

	ProjectSettings::get_singleton()->save();
	_globals_dirty = false;
}

String EditorData::TypeDB::get_script_create_dialog_inherits_field(const EditorData::TypeInfo &p_ti) const {
	if (p_ti.path.get_extension() == "gd") {
		return "Scripts." + String(p_ti.type_name);
	}
	return "";
}

EditorData::TypeDB::TypeDB() {
}

EditorData::TypeDB::~TypeDB() {
}

int EditorData::add_edited_scene(int p_at_pos) {

	if (p_at_pos < 0)
		p_at_pos = edited_scene.size();
	EditedScene es;
	es.root = NULL;
	es.history_current = -1;
	es.version = 0;
	es.live_edit_root = NodePath(String("/root"));

	if (p_at_pos == edited_scene.size())
		edited_scene.push_back(es);
	else
		edited_scene.insert(p_at_pos, es);

	if (current_edited_scene < 0)
		current_edited_scene = 0;
	return p_at_pos;
}

void EditorData::move_edited_scene_index(int p_idx, int p_to_idx) {

	ERR_FAIL_INDEX(p_idx, edited_scene.size());
	ERR_FAIL_INDEX(p_to_idx, edited_scene.size());
	SWAP(edited_scene[p_idx], edited_scene[p_to_idx]);
}
void EditorData::remove_scene(int p_idx) {
	ERR_FAIL_INDEX(p_idx, edited_scene.size());
	if (edited_scene[p_idx].root) {

		for (int i = 0; i < editor_plugins.size(); i++) {
			editor_plugins[i]->notify_scene_closed(edited_scene[p_idx].root->get_filename());
		}

		memdelete(edited_scene[p_idx].root);
	}

	if (current_edited_scene > p_idx)
		current_edited_scene--;
	else if (current_edited_scene == p_idx && current_edited_scene > 0) {
		current_edited_scene--;
	}

	edited_scene.remove(p_idx);
}

bool EditorData::_find_updated_instances(Node *p_root, Node *p_node, Set<String> &checked_paths) {

	/*
	if (p_root!=p_node && p_node->get_owner()!=p_root && !p_root->is_editable_instance(p_node->get_owner()))
		return false;
	*/

	Ref<SceneState> ss;

	if (p_node == p_root) {
		ss = p_node->get_scene_inherited_state();
	} else if (p_node->get_filename() != String()) {
		ss = p_node->get_scene_instance_state();
	}

	if (ss.is_valid()) {
		String path = ss->get_path();

		if (!checked_paths.has(path)) {

			uint64_t modified_time = FileAccess::get_modified_time(path);
			if (modified_time != ss->get_last_modified_time()) {
				return true; //external scene changed
			}

			checked_paths.insert(path);
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {

		bool found = _find_updated_instances(p_root, p_node->get_child(i), checked_paths);
		if (found)
			return true;
	}

	return false;
}

bool EditorData::check_and_update_scene(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), false);
	if (!edited_scene[p_idx].root)
		return false;

	Set<String> checked_scenes;

	bool must_reload = _find_updated_instances(edited_scene[p_idx].root, edited_scene[p_idx].root, checked_scenes);

	if (must_reload) {
		Ref<PackedScene> pscene;
		pscene.instance();

		EditorProgress ep("update_scene", TTR("Updating Scene"), 2);
		ep.step(TTR("Storing local changes..."), 0);
		//pack first, so it stores diffs to previous version of saved scene
		Error err = pscene->pack(edited_scene[p_idx].root);
		ERR_FAIL_COND_V(err != OK, false);
		ep.step(TTR("Updating scene..."), 1);
		Node *new_scene = pscene->instance(PackedScene::GEN_EDIT_STATE_MAIN);
		ERR_FAIL_COND_V(!new_scene, false);

		//transfer selection
		List<Node *> new_selection;
		for (List<Node *>::Element *E = edited_scene[p_idx].selection.front(); E; E = E->next()) {
			NodePath p = edited_scene[p_idx].root->get_path_to(E->get());
			Node *new_node = new_scene->get_node(p);
			if (new_node)
				new_selection.push_back(new_node);
		}

		new_scene->set_filename(edited_scene[p_idx].root->get_filename());

		memdelete(edited_scene[p_idx].root);
		edited_scene[p_idx].root = new_scene;
		edited_scene[p_idx].selection = new_selection;

		return true;
	}

	return false;
}

int EditorData::get_edited_scene() const {

	return current_edited_scene;
}
void EditorData::set_edited_scene(int p_idx) {

	ERR_FAIL_INDEX(p_idx, edited_scene.size());
	current_edited_scene = p_idx;
	//swap
}
Node *EditorData::get_edited_scene_root(int p_idx) {
	if (p_idx < 0) {
		ERR_FAIL_INDEX_V(current_edited_scene, edited_scene.size(), NULL);
		return edited_scene[current_edited_scene].root;
	} else {
		ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), NULL);
		return edited_scene[p_idx].root;
	}
}
void EditorData::set_edited_scene_root(Node *p_root) {

	ERR_FAIL_INDEX(current_edited_scene, edited_scene.size());
	edited_scene[current_edited_scene].root = p_root;
}

int EditorData::get_edited_scene_count() const {

	return edited_scene.size();
}

Vector<EditorData::EditedScene> EditorData::get_edited_scenes() const {

	Vector<EditedScene> out_edited_scenes_list = Vector<EditedScene>();

	for (int i = 0; i < edited_scene.size(); i++) {
		out_edited_scenes_list.push_back(edited_scene[i]);
	}

	return out_edited_scenes_list;
}

void EditorData::set_edited_scene_version(uint64_t version, int p_scene_idx) {
	ERR_FAIL_INDEX(current_edited_scene, edited_scene.size());
	if (p_scene_idx < 0) {
		edited_scene[current_edited_scene].version = version;
	} else {
		ERR_FAIL_INDEX(p_scene_idx, edited_scene.size());
		edited_scene[p_scene_idx].version = version;
	}
}

uint64_t EditorData::get_edited_scene_version() const {

	ERR_FAIL_INDEX_V(current_edited_scene, edited_scene.size(), 0);
	return edited_scene[current_edited_scene].version;
}
uint64_t EditorData::get_scene_version(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), false);
	return edited_scene[p_idx].version;
}

String EditorData::get_scene_type(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), String());
	if (!edited_scene[p_idx].root)
		return "";
	return edited_scene[p_idx].root->get_class();
}
void EditorData::move_edited_scene_to_index(int p_idx) {

	ERR_FAIL_INDEX(current_edited_scene, edited_scene.size());
	ERR_FAIL_INDEX(p_idx, edited_scene.size());

	EditedScene es = edited_scene[current_edited_scene];
	edited_scene.remove(current_edited_scene);
	edited_scene.insert(p_idx, es);
	current_edited_scene = p_idx;
}

Ref<Script> EditorData::get_scene_root_script(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), Ref<Script>());
	if (!edited_scene[p_idx].root)
		return Ref<Script>();
	Ref<Script> s = edited_scene[p_idx].root->get_script();
	if (!s.is_valid() && edited_scene[p_idx].root->get_child_count()) {
		Node *n = edited_scene[p_idx].root->get_child(0);
		while (!s.is_valid() && n && n->get_filename() == String()) {
			s = n->get_script();
			n = n->get_parent();
		}
	}
	return s;
}

String EditorData::get_scene_title(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), String());
	if (!edited_scene[p_idx].root)
		return TTR("[empty]");
	if (edited_scene[p_idx].root->get_filename() == "")
		return TTR("[unsaved]");
	bool show_ext = EDITOR_DEF("interface/scene_tabs/show_extension", false);
	String name = edited_scene[p_idx].root->get_filename().get_file();
	if (!show_ext) {
		name = name.get_basename();
	}
	return name;
}

void EditorData::set_scene_path(int p_idx, const String &p_path) {

	ERR_FAIL_INDEX(p_idx, edited_scene.size());

	if (!edited_scene[p_idx].root)
		return;
	edited_scene[p_idx].root->set_filename(p_path);
}

String EditorData::get_scene_path(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, edited_scene.size(), String());

	if (!edited_scene[p_idx].root)
		return "";
	return edited_scene[p_idx].root->get_filename();
}

void EditorData::set_edited_scene_live_edit_root(const NodePath &p_root) {
	ERR_FAIL_INDEX(current_edited_scene, edited_scene.size());

	edited_scene[current_edited_scene].live_edit_root = p_root;
}
NodePath EditorData::get_edited_scene_live_edit_root() {

	ERR_FAIL_INDEX_V(current_edited_scene, edited_scene.size(), String());

	return edited_scene[current_edited_scene].live_edit_root;
}

void EditorData::save_edited_scene_state(EditorSelection *p_selection, EditorHistory *p_history, const Dictionary &p_custom) {

	ERR_FAIL_INDEX(current_edited_scene, edited_scene.size());

	EditedScene &es = edited_scene[current_edited_scene];
	es.selection = p_selection->get_selected_node_list();
	es.history_current = p_history->current;
	es.history_stored = p_history->history;
	es.editor_states = get_editor_states();
	es.custom_state = p_custom;
}

Dictionary EditorData::restore_edited_scene_state(EditorSelection *p_selection, EditorHistory *p_history) {
	ERR_FAIL_INDEX_V(current_edited_scene, edited_scene.size(), Dictionary());

	EditedScene &es = edited_scene[current_edited_scene];

	p_history->current = es.history_current;
	p_history->history = es.history_stored;

	p_selection->clear();
	for (List<Node *>::Element *E = es.selection.front(); E; E = E->next()) {
		p_selection->add_node(E->get());
	}
	set_editor_states(es.editor_states);

	return es.custom_state;
}

void EditorData::clear_edited_scenes() {

	for (int i = 0; i < edited_scene.size(); i++) {
		if (edited_scene[i].root) {
			memdelete(edited_scene[i].root);
		}
	}
	edited_scene.clear();
}

void EditorData::set_plugin_window_layout(Ref<ConfigFile> p_layout) {
	for (int i = 0; i < editor_plugins.size(); i++) {
		editor_plugins[i]->set_window_layout(p_layout);
	}
}

void EditorData::get_plugin_window_layout(Ref<ConfigFile> p_layout) {
	for (int i = 0; i < editor_plugins.size(); i++) {
		editor_plugins[i]->get_window_layout(p_layout);
	}
}

EditorData::EditorData() {

	current_edited_scene = -1;

	//load_imported_scenes_from_globals();
}

///////////
void EditorSelection::_node_removed(Node *p_node) {

	if (!selection.has(p_node))
		return;

	Object *meta = selection[p_node];
	if (meta)
		memdelete(meta);
	selection.erase(p_node);
	changed = true;
	nl_changed = true;
}

void EditorSelection::add_node(Node *p_node) {

	ERR_FAIL_NULL(p_node);
	ERR_FAIL_COND(!p_node->is_inside_tree());
	if (selection.has(p_node))
		return;

	changed = true;
	nl_changed = true;
	Object *meta = NULL;
	for (List<Object *>::Element *E = editor_plugins.front(); E; E = E->next()) {

		meta = E->get()->call("_get_editor_data", p_node);
		if (meta) {
			break;
		}
	}
	selection[p_node] = meta;

	p_node->connect("tree_exiting", this, "_node_removed", varray(p_node), CONNECT_ONESHOT);

	//emit_signal("selection_changed");
}

void EditorSelection::remove_node(Node *p_node) {

	ERR_FAIL_NULL(p_node);

	if (!selection.has(p_node))
		return;

	changed = true;
	nl_changed = true;
	Object *meta = selection[p_node];
	if (meta)
		memdelete(meta);
	selection.erase(p_node);
	p_node->disconnect("tree_exiting", this, "_node_removed");
	//emit_signal("selection_changed");
}
bool EditorSelection::is_selected(Node *p_node) const {

	return selection.has(p_node);
}

Array EditorSelection::_get_transformable_selected_nodes() {

	Array ret;

	for (List<Node *>::Element *E = selected_node_list.front(); E; E = E->next()) {

		ret.push_back(E->get());
	}

	return ret;
}

Array EditorSelection::get_selected_nodes() {

	Array ret;

	for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

		ret.push_back(E->key());
	}

	return ret;
}

void EditorSelection::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_node_removed"), &EditorSelection::_node_removed);
	ClassDB::bind_method(D_METHOD("clear"), &EditorSelection::clear);
	ClassDB::bind_method(D_METHOD("add_node", "node"), &EditorSelection::add_node);
	ClassDB::bind_method(D_METHOD("remove_node", "node"), &EditorSelection::remove_node);
	ClassDB::bind_method(D_METHOD("get_selected_nodes"), &EditorSelection::get_selected_nodes);
	ClassDB::bind_method(D_METHOD("get_transformable_selected_nodes"), &EditorSelection::_get_transformable_selected_nodes);
	ClassDB::bind_method(D_METHOD("_emit_change"), &EditorSelection::_emit_change);
	ADD_SIGNAL(MethodInfo("selection_changed"));
}

void EditorSelection::add_editor_plugin(Object *p_object) {

	editor_plugins.push_back(p_object);
}

void EditorSelection::_update_nl() {

	if (!nl_changed)
		return;

	selected_node_list.clear();

	for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

		Node *parent = E->key();
		parent = parent->get_parent();
		bool skip = false;
		while (parent) {
			if (selection.has(parent)) {
				skip = true;
				break;
			}
			parent = parent->get_parent();
		}

		if (skip)
			continue;
		selected_node_list.push_back(E->key());
	}

	nl_changed = true;
}

void EditorSelection::update() {

	_update_nl();

	if (!changed)
		return;
	changed = false;
	if (!emitted) {
		emitted = true;
		call_deferred("_emit_change");
	}
}

void EditorSelection::_emit_change() {
	emit_signal("selection_changed");
	emitted = false;
}

List<Node *> &EditorSelection::get_selected_node_list() {

	if (changed)
		update();
	else
		_update_nl();
	return selected_node_list;
}

void EditorSelection::clear() {

	while (!selection.empty()) {

		remove_node(selection.front()->key());
	}

	changed = true;
	nl_changed = true;
}
EditorSelection::EditorSelection() {

	emitted = false;
	changed = false;
	nl_changed = false;
}

EditorSelection::~EditorSelection() {

	clear();
}
