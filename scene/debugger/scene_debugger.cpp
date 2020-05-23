/*************************************************************************/
/*  scene_debugger.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene_debugger.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"
#include "core/script_language.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/packed_scene.h"

void SceneDebugger::initialize() {
#ifdef DEBUG_ENABLED
	LiveEditor::singleton = memnew(LiveEditor);
	EngineDebugger::register_message_capture("scene", EngineDebugger::Capture(nullptr, SceneDebugger::parse_message));
#endif
}

void SceneDebugger::deinitialize() {
#ifdef DEBUG_ENABLED
	if (LiveEditor::singleton) {
		// Should be removed automatically when deiniting debugger, but just in case
		if (EngineDebugger::has_capture("scene")) {
			EngineDebugger::unregister_message_capture("scene");
		}
		memdelete(LiveEditor::singleton);
		LiveEditor::singleton = nullptr;
	}
#endif
}

#ifdef DEBUG_ENABLED
Error SceneDebugger::parse_message(void *p_user, const String &p_msg, const Array &p_args, bool &r_captured) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return ERR_UNCONFIGURED;
	}
	LiveEditor *live_editor = LiveEditor::get_singleton();
	if (!live_editor) {
		return ERR_UNCONFIGURED;
	}

	r_captured = true;
	if (p_msg == "request_scene_tree") { // Scene tree
		live_editor->_send_tree();

	} else if (p_msg == "save_node") { // Save node.
		ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
		_save_node(p_args[0], p_args[1]);

	} else if (p_msg == "inspect_object") { // Object Inspect
		ERR_FAIL_COND_V(p_args.size() < 1, ERR_INVALID_DATA);
		ObjectID id = p_args[0];
		_send_object_id(id);

	} else if (p_msg == "override_camera_2D:set") { // Camera
		ERR_FAIL_COND_V(p_args.size() < 1, ERR_INVALID_DATA);
		bool enforce = p_args[0];
		scene_tree->get_root()->enable_canvas_transform_override(enforce);

	} else if (p_msg == "override_camera_2D:transform") {
		ERR_FAIL_COND_V(p_args.size() < 1, ERR_INVALID_DATA);
		Transform2D transform = p_args[1];
		scene_tree->get_root()->set_canvas_transform_override(transform);

	} else if (p_msg == "override_camera_3D:set") {
		ERR_FAIL_COND_V(p_args.size() < 1, ERR_INVALID_DATA);
		bool enable = p_args[0];
		scene_tree->get_root()->enable_camera_override(enable);

	} else if (p_msg == "override_camera_3D:transform") {
		ERR_FAIL_COND_V(p_args.size() < 5, ERR_INVALID_DATA);
		Transform transform = p_args[0];
		bool is_perspective = p_args[1];
		float size_or_fov = p_args[2];
		float near = p_args[3];
		float far = p_args[4];
		if (is_perspective) {
			scene_tree->get_root()->set_camera_override_perspective(size_or_fov, near, far);
		} else {
			scene_tree->get_root()->set_camera_override_orthogonal(size_or_fov, near, far);
		}
		scene_tree->get_root()->set_camera_override_transform(transform);

	} else if (p_msg == "set_object_property") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		_set_object_property(p_args[0], p_args[1], p_args[2]);

	} else if (!p_msg.begins_with("live_")) { // Live edits below.
		return ERR_SKIP;
	} else if (p_msg == "live_set_root") {
		ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
		live_editor->_root_func(p_args[0], p_args[1]);

	} else if (p_msg == "live_node_path") {
		ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
		live_editor->_node_path_func(p_args[0], p_args[1]);

	} else if (p_msg == "live_res_path") {
		ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
		live_editor->_res_path_func(p_args[0], p_args[1]);

	} else if (p_msg == "live_node_prop_res") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_node_set_res_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_node_prop") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_node_set_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_res_prop_res") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_res_set_res_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_res_prop") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_res_set_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_node_call") {
		ERR_FAIL_COND_V(p_args.size() < 7, ERR_INVALID_DATA);
		live_editor->_node_call_func(p_args[0], p_args[1], p_args[2], p_args[3], p_args[4], p_args[5], p_args[6]);

	} else if (p_msg == "live_res_call") {
		ERR_FAIL_COND_V(p_args.size() < 7, ERR_INVALID_DATA);
		live_editor->_res_call_func(p_args[0], p_args[1], p_args[2], p_args[3], p_args[4], p_args[5], p_args[6]);

	} else if (p_msg == "live_create_node") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_create_node_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_instance_node") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_instance_node_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_remove_node") {
		ERR_FAIL_COND_V(p_args.size() < 1, ERR_INVALID_DATA);
		live_editor->_remove_node_func(p_args[0]);

	} else if (p_msg == "live_remove_and_keep_node") {
		ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
		live_editor->_remove_and_keep_node_func(p_args[0], p_args[1]);

	} else if (p_msg == "live_restore_node") {
		ERR_FAIL_COND_V(p_args.size() < 3, ERR_INVALID_DATA);
		live_editor->_restore_node_func(p_args[0], p_args[1], p_args[2]);

	} else if (p_msg == "live_duplicate_node") {
		ERR_FAIL_COND_V(p_args.size() < 2, ERR_INVALID_DATA);
		live_editor->_duplicate_node_func(p_args[0], p_args[1]);

	} else if (p_msg == "live_reparent_node") {
		ERR_FAIL_COND_V(p_args.size() < 4, ERR_INVALID_DATA);
		live_editor->_reparent_node_func(p_args[0], p_args[1], p_args[2], p_args[3]);
	} else {
		r_captured = false;
	}
	return OK;
}

void SceneDebugger::_save_node(ObjectID id, const String &p_path) {
	Node *node = Object::cast_to<Node>(ObjectDB::get_instance(id));
	ERR_FAIL_COND(!node);

	Ref<PackedScene> ps = memnew(PackedScene);
	ps->pack(node);
	ResourceSaver::save(p_path, ps);
}

void SceneDebugger::_send_object_id(ObjectID p_id, int p_max_size) {
	SceneDebuggerObject obj(p_id);
	if (obj.id.is_null()) {
		return;
	}

	Array arr;
	obj.serialize(arr);
	EngineDebugger::get_singleton()->send_message("scene:inspect_object", arr);
}

void SceneDebugger::_set_object_property(ObjectID p_id, const String &p_property, const Variant &p_value) {
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return;
	}

	String prop_name = p_property;
	if (p_property.begins_with("Members/")) {
		Vector<String> ss = p_property.split("/");
		prop_name = ss[ss.size() - 1];
	}

	obj->set(prop_name, p_value);
}

void SceneDebugger::add_to_cache(const String &p_filename, Node *p_node) {
	LiveEditor *debugger = LiveEditor::get_singleton();
	if (!debugger) {
		return;
	}

	if (EngineDebugger::get_script_debugger() && p_filename != String()) {
		debugger->live_scene_edit_cache[p_filename].insert(p_node);
	}
}

void SceneDebugger::remove_from_cache(const String &p_filename, Node *p_node) {
	LiveEditor *debugger = LiveEditor::get_singleton();
	if (!debugger) {
		return;
	}

	Map<String, Set<Node *>> &edit_cache = debugger->live_scene_edit_cache;
	Map<String, Set<Node *>>::Element *E = edit_cache.find(p_filename);
	if (E) {
		E->get().erase(p_node);
		if (E->get().size() == 0) {
			edit_cache.erase(E);
		}
	}

	Map<Node *, Map<ObjectID, Node *>> &remove_list = debugger->live_edit_remove_list;
	Map<Node *, Map<ObjectID, Node *>>::Element *F = remove_list.find(p_node);
	if (F) {
		for (Map<ObjectID, Node *>::Element *G = F->get().front(); G; G = G->next()) {
			memdelete(G->get());
		}
		remove_list.erase(F);
	}
}

/// SceneDebuggerObject
SceneDebuggerObject::SceneDebuggerObject(ObjectID p_id) {
	id = ObjectID();
	Object *obj = ObjectDB::get_instance(p_id);
	if (!obj) {
		return;
	}

	id = p_id;
	class_name = obj->get_class();

	if (ScriptInstance *si = obj->get_script_instance()) {
		// Read script instance constants and variables
		if (!si->get_script().is_null()) {
			Script *s = si->get_script().ptr();
			_parse_script_properties(s, si);
		}
	}

	if (Node *node = Object::cast_to<Node>(obj)) {
		// Add specialized NodePath info (if inside tree).
		if (node->is_inside_tree()) {
			PropertyInfo pi(Variant::NODE_PATH, String("Node/path"));
			properties.push_back(SceneDebuggerProperty(pi, node->get_path()));
		} else { // Can't ask for path if a node is not in tree.
			PropertyInfo pi(Variant::STRING, String("Node/path"));
			properties.push_back(SceneDebuggerProperty(pi, "[Orphan]"));
		}
	} else if (Script *s = Object::cast_to<Script>(obj)) {
		// Add script constants (no instance).
		_parse_script_properties(s, nullptr);
	}

	// Add base object properties.
	List<PropertyInfo> pinfo;
	obj->get_property_list(&pinfo, true);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		if (E->get().usage & (PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_CATEGORY)) {
			properties.push_back(SceneDebuggerProperty(E->get(), obj->get(E->get().name)));
		}
	}
}

void SceneDebuggerObject::_parse_script_properties(Script *p_script, ScriptInstance *p_instance) {
	typedef Map<const Script *, Set<StringName>> ScriptMemberMap;
	typedef Map<const Script *, Map<StringName, Variant>> ScriptConstantsMap;

	ScriptMemberMap members;
	if (p_instance) {
		members[p_script] = Set<StringName>();
		p_script->get_members(&(members[p_script]));
	}

	ScriptConstantsMap constants;
	constants[p_script] = Map<StringName, Variant>();
	p_script->get_constants(&(constants[p_script]));

	Ref<Script> base = p_script->get_base_script();
	while (base.is_valid()) {
		if (p_instance) {
			members[base.ptr()] = Set<StringName>();
			base->get_members(&(members[base.ptr()]));
		}

		constants[base.ptr()] = Map<StringName, Variant>();
		base->get_constants(&(constants[base.ptr()]));

		base = base->get_base_script();
	}

	// Members
	for (ScriptMemberMap::Element *sm = members.front(); sm; sm = sm->next()) {
		for (Set<StringName>::Element *E = sm->get().front(); E; E = E->next()) {
			Variant m;
			if (p_instance->get(E->get(), m)) {
				String script_path = sm->key() == p_script ? "" : sm->key()->get_path().get_file() + "/";
				PropertyInfo pi(m.get_type(), "Members/" + script_path + E->get());
				properties.push_back(SceneDebuggerProperty(pi, m));
			}
		}
	}
	// Constants
	for (ScriptConstantsMap::Element *sc = constants.front(); sc; sc = sc->next()) {
		for (Map<StringName, Variant>::Element *E = sc->get().front(); E; E = E->next()) {
			String script_path = sc->key() == p_script ? "" : sc->key()->get_path().get_file() + "/";
			if (E->value().get_type() == Variant::OBJECT) {
				Variant id = ((Object *)E->value())->get_instance_id();
				PropertyInfo pi(id.get_type(), "Constants/" + E->key(), PROPERTY_HINT_OBJECT_ID, "Object");
				properties.push_back(SceneDebuggerProperty(pi, id));
			} else {
				PropertyInfo pi(E->value().get_type(), "Constants/" + script_path + E->key());
				properties.push_back(SceneDebuggerProperty(pi, E->value()));
			}
		}
	}
}

void SceneDebuggerObject::serialize(Array &r_arr, int p_max_size) {
	Array send_props;
	for (int i = 0; i < properties.size(); i++) {
		const PropertyInfo &pi = properties[i].first;
		Variant &var = properties[i].second;

		RES res = var;

		Array prop;
		prop.push_back(pi.name);
		prop.push_back(pi.type);

		PropertyHint hint = pi.hint;
		String hint_string = pi.hint_string;
		if (!res.is_null()) {
			var = res->get_path();
		} else { //only send information that can be sent..
			int len = 0; //test how big is this to encode
			encode_variant(var, nullptr, len);
			if (len > p_max_size) { //limit to max size
				hint = PROPERTY_HINT_OBJECT_TOO_BIG;
				hint_string = "";
				var = Variant();
			}
		}
		prop.push_back(hint);
		prop.push_back(hint_string);
		prop.push_back(pi.usage);
		prop.push_back(var);
		send_props.push_back(prop);
	}
	r_arr.push_back(uint64_t(id));
	r_arr.push_back(class_name);
	r_arr.push_back(send_props);
}

void SceneDebuggerObject::deserialize(const Array &p_arr) {
#define CHECK_TYPE(p_what, p_type) ERR_FAIL_COND(p_what.get_type() != Variant::p_type);
	ERR_FAIL_COND(p_arr.size() < 3);
	CHECK_TYPE(p_arr[0], INT);
	CHECK_TYPE(p_arr[1], STRING);
	CHECK_TYPE(p_arr[2], ARRAY);

	id = uint64_t(p_arr[0]);
	class_name = p_arr[1];
	Array props = p_arr[2];

	for (int i = 0; i < props.size(); i++) {
		CHECK_TYPE(props[i], ARRAY);
		Array prop = props[i];

		ERR_FAIL_COND(prop.size() != 6);
		CHECK_TYPE(prop[0], STRING);
		CHECK_TYPE(prop[1], INT);
		CHECK_TYPE(prop[2], INT);
		CHECK_TYPE(prop[3], STRING);
		CHECK_TYPE(prop[4], INT);

		PropertyInfo pinfo;
		pinfo.name = prop[0];
		pinfo.type = Variant::Type(int(prop[1]));
		pinfo.hint = PropertyHint(int(prop[2]));
		pinfo.hint_string = prop[3];
		pinfo.usage = PropertyUsageFlags(int(prop[4]));
		Variant var = prop[5];

		if (pinfo.type == Variant::OBJECT) {
			if (var.is_zero()) {
				var = RES();
			} else if (var.get_type() == Variant::OBJECT) {
				if (((Object *)var)->is_class("EncodedObjectAsID")) {
					var = Object::cast_to<EncodedObjectAsID>(var)->get_object_id();
					pinfo.type = var.get_type();
					pinfo.hint = PROPERTY_HINT_OBJECT_ID;
					pinfo.hint_string = "Object";
				}
			}
		}
		properties.push_back(SceneDebuggerProperty(pinfo, var));
	}
}

/// SceneDebuggerTree
SceneDebuggerTree::SceneDebuggerTree(Node *p_root) {
	// Flatten tree into list, depth first, use stack to avoid recursion.
	List<Node *> stack;
	stack.push_back(p_root);
	while (stack.size()) {
		Node *n = stack[0];
		stack.pop_front();
		int count = n->get_child_count();
		nodes.push_back(RemoteNode(count, n->get_name(), n->get_class(), n->get_instance_id()));
		for (int i = 0; i < count; i++) {
			stack.push_front(n->get_child(count - i - 1));
		}
	}
}

void SceneDebuggerTree::serialize(Array &p_arr) {
	for (List<RemoteNode>::Element *E = nodes.front(); E; E = E->next()) {
		RemoteNode &n = E->get();
		p_arr.push_back(n.child_count);
		p_arr.push_back(n.name);
		p_arr.push_back(n.type_name);
		p_arr.push_back(n.id);
	}
}

void SceneDebuggerTree::deserialize(const Array &p_arr) {
	int idx = 0;
	while (p_arr.size() > idx) {
		ERR_FAIL_COND(p_arr.size() < 4);
		CHECK_TYPE(p_arr[idx], INT);
		CHECK_TYPE(p_arr[idx + 1], STRING);
		CHECK_TYPE(p_arr[idx + 2], STRING);
		CHECK_TYPE(p_arr[idx + 3], INT);
		nodes.push_back(RemoteNode(p_arr[idx], p_arr[idx + 1], p_arr[idx + 2], p_arr[idx + 3]));
		idx += 4;
	}
}

/// LiveEditor
LiveEditor *LiveEditor::singleton = nullptr;
LiveEditor *LiveEditor::get_singleton() {
	return singleton;
}

void LiveEditor::_send_tree() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Array arr;
	// Encoded as a flat list depth fist.
	SceneDebuggerTree tree(scene_tree->root);
	tree.serialize(arr);
	EngineDebugger::get_singleton()->send_message("scene:scene_tree", arr);
}

void LiveEditor::_node_path_func(const NodePath &p_path, int p_id) {
	live_edit_node_path_cache[p_id] = p_path;
}

void LiveEditor::_res_path_func(const String &p_path, int p_id) {
	live_edit_resource_cache[p_id] = p_path;
}

void LiveEditor::_node_set_func(int p_id, const StringName &p_prop, const Variant &p_value) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	if (!live_edit_node_path_cache.has(p_id)) {
		return;
	}

	NodePath np = live_edit_node_path_cache[p_id];
	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {
		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(np)) {
			continue;
		}
		Node *n2 = n->get_node(np);

		n2->set(p_prop, p_value);
	}
}

void LiveEditor::_node_set_res_func(int p_id, const StringName &p_prop, const String &p_value) {
	RES r = ResourceLoader::load(p_value);
	if (!r.is_valid()) {
		return;
	}
	_node_set_func(p_id, p_prop, r);
}

void LiveEditor::_node_call_func(int p_id, const StringName &p_method, VARIANT_ARG_DECLARE) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}
	if (!live_edit_node_path_cache.has(p_id)) {
		return;
	}

	NodePath np = live_edit_node_path_cache[p_id];
	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {
		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(np)) {
			continue;
		}
		Node *n2 = n->get_node(np);

		n2->call(p_method, VARIANT_ARG_PASS);
	}
}

void LiveEditor::_res_set_func(int p_id, const StringName &p_prop, const Variant &p_value) {
	if (!live_edit_resource_cache.has(p_id)) {
		return;
	}

	String resp = live_edit_resource_cache[p_id];

	if (!ResourceCache::has(resp)) {
		return;
	}

	RES r = ResourceCache::get(resp);
	if (!r.is_valid()) {
		return;
	}

	r->set(p_prop, p_value);
}

void LiveEditor::_res_set_res_func(int p_id, const StringName &p_prop, const String &p_value) {
	RES r = ResourceLoader::load(p_value);
	if (!r.is_valid()) {
		return;
	}
	_res_set_func(p_id, p_prop, r);
}

void LiveEditor::_res_call_func(int p_id, const StringName &p_method, VARIANT_ARG_DECLARE) {
	if (!live_edit_resource_cache.has(p_id)) {
		return;
	}

	String resp = live_edit_resource_cache[p_id];

	if (!ResourceCache::has(resp)) {
		return;
	}

	RES r = ResourceCache::get(resp);
	if (!r.is_valid()) {
		return;
	}

	r->call(p_method, VARIANT_ARG_PASS);
}

void LiveEditor::_root_func(const NodePath &p_scene_path, const String &p_scene_from) {
	live_edit_root = p_scene_path;
	live_edit_scene = p_scene_from;
}

void LiveEditor::_create_node_func(const NodePath &p_parent, const String &p_type, const String &p_name) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {
		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_parent)) {
			continue;
		}
		Node *n2 = n->get_node(p_parent);

		Node *no = Object::cast_to<Node>(ClassDB::instance(p_type));
		if (!no) {
			continue;
		}

		no->set_name(p_name);
		n2->add_child(no);
	}
}

void LiveEditor::_instance_node_func(const NodePath &p_parent, const String &p_path, const String &p_name) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Ref<PackedScene> ps = ResourceLoader::load(p_path);

	if (!ps.is_valid()) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {
		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_parent)) {
			continue;
		}
		Node *n2 = n->get_node(p_parent);

		Node *no = ps->instance();
		if (!no) {
			continue;
		}

		no->set_name(p_name);
		n2->add_child(no);
	}
}

void LiveEditor::_remove_node_func(const NodePath &p_at) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F;) {
		Set<Node *>::Element *N = F->next();

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *n2 = n->get_node(p_at);

		memdelete(n2);

		F = N;
	}
}

void LiveEditor::_remove_and_keep_node_func(const NodePath &p_at, ObjectID p_keep_id) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F;) {
		Set<Node *>::Element *N = F->next();

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}

		Node *n2 = n->get_node(p_at);

		n2->get_parent()->remove_child(n2);

		live_edit_remove_list[n][p_keep_id] = n2;

		F = N;
	}
}

void LiveEditor::_restore_node_func(ObjectID p_id, const NodePath &p_at, int p_at_pos) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F;) {
		Set<Node *>::Element *N = F->next();

		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *n2 = n->get_node(p_at);

		Map<Node *, Map<ObjectID, Node *>>::Element *EN = live_edit_remove_list.find(n);

		if (!EN) {
			continue;
		}

		Map<ObjectID, Node *>::Element *FN = EN->get().find(p_id);

		if (!FN) {
			continue;
		}
		n2->add_child(FN->get());

		EN->get().erase(FN);

		if (EN->get().size() == 0) {
			live_edit_remove_list.erase(EN);
		}

		F = N;
	}
}

void LiveEditor::_duplicate_node_func(const NodePath &p_at, const String &p_new_name) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {
		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *n2 = n->get_node(p_at);

		Node *dup = n2->duplicate(Node::DUPLICATE_SIGNALS | Node::DUPLICATE_GROUPS | Node::DUPLICATE_SCRIPTS);

		if (!dup) {
			continue;
		}

		dup->set_name(p_new_name);
		n2->get_parent()->add_child(dup);
	}
}

void LiveEditor::_reparent_node_func(const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos) {
	SceneTree *scene_tree = SceneTree::get_singleton();
	if (!scene_tree) {
		return;
	}

	Node *base = nullptr;
	if (scene_tree->root->has_node(live_edit_root)) {
		base = scene_tree->root->get_node(live_edit_root);
	}

	Map<String, Set<Node *>>::Element *E = live_scene_edit_cache.find(live_edit_scene);
	if (!E) {
		return; //scene not editable
	}

	for (Set<Node *>::Element *F = E->get().front(); F; F = F->next()) {
		Node *n = F->get();

		if (base && !base->is_a_parent_of(n)) {
			continue;
		}

		if (!n->has_node(p_at)) {
			continue;
		}
		Node *nfrom = n->get_node(p_at);

		if (!n->has_node(p_new_place)) {
			continue;
		}
		Node *nto = n->get_node(p_new_place);

		nfrom->get_parent()->remove_child(nfrom);
		nfrom->set_name(p_new_name);

		nto->add_child(nfrom);
		if (p_at_pos >= 0) {
			nto->move_child(nfrom, p_at_pos);
		}
	}
}

#endif
