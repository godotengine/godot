/*************************************************************************/
/*  visual_script_func_nodes.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "visual_script_func_nodes.h"

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "visual_script_nodes.h"

//////////////////////////////////////////
////////////////CALL//////////////////////
//////////////////////////////////////////

int VisualScriptFunctionCall::get_output_sequence_port_count() const {
	if ((method_cache.flags & METHOD_FLAG_CONST && call_mode != CALL_MODE_INSTANCE) || (call_mode == CALL_MODE_BASIC_TYPE && Variant::is_builtin_method_const(basic_type, function))) {
		return 0;
	} else {
		return 1;
	}
}

bool VisualScriptFunctionCall::has_input_sequence_port() const {
	return !((method_cache.flags & METHOD_FLAG_CONST && call_mode != CALL_MODE_INSTANCE) || (call_mode == CALL_MODE_BASIC_TYPE && Variant::is_builtin_method_const(basic_type, function)));
}
#ifdef TOOLS_ENABLED

static Node *_find_script_node(Node *p_edited_scene, Node *p_current_node, const Ref<Script> &script) {
	if (p_edited_scene != p_current_node && p_current_node->get_owner() != p_edited_scene) {
		return nullptr;
	}

	Ref<Script> scr = p_current_node->get_script();

	if (scr.is_valid() && scr == script) {
		return p_current_node;
	}

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *n = _find_script_node(p_edited_scene, p_current_node->get_child(i), script);
		if (n) {
			return n;
		}
	}

	return nullptr;
}

#endif
Node *VisualScriptFunctionCall::_get_base_node() const {
#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid()) {
		return nullptr;
	}

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);

	if (!scene_tree) {
		return nullptr;
	}

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene) {
		return nullptr;
	}

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node) {
		return nullptr;
	}

	if (!script_node->has_node(base_path)) {
		return nullptr;
	}

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return nullptr;
#endif
}

StringName VisualScriptFunctionCall::_get_base_type() const {
	if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
		return get_visual_script()->get_instance_base_type();
	} else if (call_mode == CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path) {
			return path->get_class();
		}
	}

	return base_type;
}

int VisualScriptFunctionCall::get_input_value_port_count() const {
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		Vector<Variant::Type> types;
		int argc = Variant::get_builtin_method_argument_count(basic_type, function);
		for (int i = 0; i < argc; i++) {
			types.push_back(Variant::get_builtin_method_argument_type(basic_type, function, i));
		}
		return types.size() + (rpc_call_mode >= RPC_RELIABLE_TO_ID ? 1 : 0) + 1;

	} else {
		MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
		if (mb) {
			int defaulted_args = mb->get_argument_count() < use_default_args ? mb->get_argument_count() : use_default_args;
			return mb->get_argument_count() + (call_mode == CALL_MODE_INSTANCE ? 1 : 0) + (rpc_call_mode >= RPC_RELIABLE_TO_ID ? 1 : 0) - defaulted_args;
		}

		int defaulted_args = method_cache.arguments.size() < use_default_args ? method_cache.arguments.size() : use_default_args;
		return method_cache.arguments.size() + (call_mode == CALL_MODE_INSTANCE ? 1 : 0) + (rpc_call_mode >= RPC_RELIABLE_TO_ID ? 1 : 0) - defaulted_args;
	}
}

int VisualScriptFunctionCall::get_output_value_port_count() const {
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		bool returns = Variant::has_builtin_method_return_value(basic_type, function);
		return returns ? 1 : 0;

	} else {
		int ret;
		MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
		if (mb) {
			ret = mb->has_return() ? 1 : 0;
		} else {
			ret = 1; //it is assumed that script always returns something
		}

		if (call_mode == CALL_MODE_INSTANCE) {
			ret++;
		}

		return ret;
	}
}

String VisualScriptFunctionCall::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptFunctionCall::get_input_value_port_info(int p_idx) const {
	if (call_mode == CALL_MODE_INSTANCE || call_mode == CALL_MODE_BASIC_TYPE) {
		if (p_idx == 0) {
			PropertyInfo pi;
			pi.type = (call_mode == CALL_MODE_INSTANCE ? Variant::OBJECT : basic_type);
			pi.name = (call_mode == CALL_MODE_INSTANCE ? String("instance") : Variant::get_type_name(basic_type).to_lower());
			return pi;
		} else {
			p_idx--;
		}
	}

	if (rpc_call_mode >= RPC_RELIABLE_TO_ID) {
		if (p_idx == 0) {
			return PropertyInfo(Variant::INT, "peer_id");
		} else {
			p_idx--;
		}
	}

#ifdef DEBUG_METHODS_ENABLED

	if (call_mode == CALL_MODE_BASIC_TYPE) {
		return PropertyInfo(Variant::get_builtin_method_argument_type(basic_type, function, p_idx), Variant::get_builtin_method_argument_name(basic_type, function, p_idx));
	} else {
		MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
		if (mb) {
			return mb->get_argument_info(p_idx);
		}

		if (p_idx >= 0 && p_idx < method_cache.arguments.size()) {
			return method_cache.arguments[p_idx];
		}

		return PropertyInfo();
	}
#else
	return PropertyInfo();
#endif
}

PropertyInfo VisualScriptFunctionCall::get_output_value_port_info(int p_idx) const {
#ifdef DEBUG_METHODS_ENABLED

	if (call_mode == CALL_MODE_BASIC_TYPE) {
		return PropertyInfo(Variant::get_builtin_method_return_type(basic_type, function), "");
	} else {
		if (call_mode == CALL_MODE_INSTANCE) {
			if (p_idx == 0) {
				return PropertyInfo(Variant::OBJECT, "pass", PROPERTY_HINT_TYPE_STRING, get_base_type());
			} else {
				p_idx--;
			}
		}

		PropertyInfo ret;

		/*MethodBind *mb = ClassDB::get_method(_get_base_type(),function);
		if (mb) {
			ret = mb->get_argument_info(-1);
		} else {*/

		ret = method_cache.return_val;

		//}

		if (call_mode == CALL_MODE_INSTANCE) {
			ret.name = "return";
		} else {
			ret.name = "";
		}
		return ret;
	}
#else
	return PropertyInfo();
#endif
}

String VisualScriptFunctionCall::get_caption() const {
	return "  " + String(function) + "()";
}

String VisualScriptFunctionCall::get_text() const {
	String text;

	if (call_mode == CALL_MODE_BASIC_TYPE) {
		text = String("On ") + Variant::get_type_name(basic_type);
	} else if (call_mode == CALL_MODE_INSTANCE) {
		text = String("On ") + base_type;
	} else if (call_mode == CALL_MODE_NODE_PATH) {
		text = "[" + String(base_path.simplified()) + "]";
	} else if (call_mode == CALL_MODE_SELF) {
		text = "On Self";
	} else if (call_mode == CALL_MODE_SINGLETON) {
		text = String(singleton) + ":" + String(function) + "()";
	}

	if (rpc_call_mode) {
		text += " RPC";
		if (rpc_call_mode == RPC_UNRELIABLE || rpc_call_mode == RPC_UNRELIABLE_TO_ID) {
			text += " UNREL";
		}
	}

	return text;
}

void VisualScriptFunctionCall::set_basic_type(Variant::Type p_type) {
	if (basic_type == p_type) {
		return;
	}
	basic_type = p_type;

	notify_property_list_changed();
	ports_changed_notify();
}

Variant::Type VisualScriptFunctionCall::get_basic_type() const {
	return basic_type;
}

void VisualScriptFunctionCall::set_base_type(const StringName &p_type) {
	if (base_type == p_type) {
		return;
	}

	base_type = p_type;
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptFunctionCall::get_base_type() const {
	return base_type;
}

void VisualScriptFunctionCall::set_base_script(const String &p_path) {
	if (base_script == p_path) {
		return;
	}

	base_script = p_path;
	notify_property_list_changed();
	ports_changed_notify();
}

String VisualScriptFunctionCall::get_base_script() const {
	return base_script;
}

void VisualScriptFunctionCall::set_singleton(const StringName &p_type) {
	if (singleton == p_type) {
		return;
	}

	singleton = p_type;
	Object *obj = Engine::get_singleton()->get_singleton_object(singleton);
	if (obj) {
		base_type = obj->get_class();
	}

	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptFunctionCall::get_singleton() const {
	return singleton;
}

void VisualScriptFunctionCall::_update_method_cache() {
	StringName type;
	Ref<Script> script;

	if (call_mode == CALL_MODE_NODE_PATH) {
		Node *node = _get_base_node();
		if (node) {
			type = node->get_class();
			base_type = type; //cache, too
			script = node->get_script();
		}
	} else if (call_mode == CALL_MODE_SELF) {
		if (get_visual_script().is_valid()) {
			type = get_visual_script()->get_instance_base_type();
			base_type = type; //cache, too
			script = get_visual_script();
		}

	} else if (call_mode == CALL_MODE_SINGLETON) {
		Object *obj = Engine::get_singleton()->get_singleton_object(singleton);
		if (obj) {
			type = obj->get_class();
			script = obj->get_script();
		}

	} else if (call_mode == CALL_MODE_INSTANCE) {
		type = base_type;
		if (!base_script.is_empty()) {
			if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {
				ScriptServer::edit_request_func(base_script); //make sure it's loaded
			}

			if (ResourceCache::has(base_script)) {
				script = Ref<Resource>(ResourceCache::get(base_script));
			} else {
				return;
			}
		}
	}

	MethodBind *mb = ClassDB::get_method(type, function);
	if (mb) {
		use_default_args = mb->get_default_argument_count();
		method_cache = MethodInfo();
		for (int i = 0; i < mb->get_argument_count(); i++) {
#ifdef DEBUG_METHODS_ENABLED
			method_cache.arguments.push_back(mb->get_argument_info(i));
#else
			method_cache.arguments.push_back(PropertyInfo());
#endif
		}

		if (mb->is_const()) {
			method_cache.flags |= METHOD_FLAG_CONST;
		}

#ifdef DEBUG_METHODS_ENABLED

		method_cache.return_val = mb->get_return_info();
#endif

		if (mb->is_vararg()) {
			//for vararg just give it 10 arguments (should be enough for most use cases)
			for (int i = 0; i < 10; i++) {
				method_cache.arguments.push_back(PropertyInfo(Variant::NIL, "arg" + itos(i)));
				use_default_args++;
			}
		}
	} else if (script.is_valid() && script->has_method(function)) {
		method_cache = script->get_method_info(function);
		use_default_args = method_cache.default_arguments.size();
	}
}

void VisualScriptFunctionCall::set_function(const StringName &p_type) {
	if (function == p_type) {
		return;
	}

	function = p_type;

	if (call_mode == CALL_MODE_BASIC_TYPE) {
		use_default_args = Variant::get_builtin_method_default_arguments(basic_type, function).size();
	} else {
		//update all caches

		_update_method_cache();
	}

	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptFunctionCall::get_function() const {
	return function;
}

void VisualScriptFunctionCall::set_base_path(const NodePath &p_type) {
	if (base_path == p_type) {
		return;
	}

	base_path = p_type;
	notify_property_list_changed();
	ports_changed_notify();
}

NodePath VisualScriptFunctionCall::get_base_path() const {
	return base_path;
}

void VisualScriptFunctionCall::set_call_mode(CallMode p_mode) {
	if (call_mode == p_mode) {
		return;
	}

	call_mode = p_mode;
	notify_property_list_changed();
	ports_changed_notify();
}

VisualScriptFunctionCall::CallMode VisualScriptFunctionCall::get_call_mode() const {
	return call_mode;
}

void VisualScriptFunctionCall::set_use_default_args(int p_amount) {
	if (use_default_args == p_amount) {
		return;
	}

	use_default_args = p_amount;
	ports_changed_notify();
}

void VisualScriptFunctionCall::set_rpc_call_mode(VisualScriptFunctionCall::RPCCallMode p_mode) {
	if (rpc_call_mode == p_mode) {
		return;
	}
	rpc_call_mode = p_mode;
	ports_changed_notify();
	notify_property_list_changed();
}

VisualScriptFunctionCall::RPCCallMode VisualScriptFunctionCall::get_rpc_call_mode() const {
	return rpc_call_mode;
}

int VisualScriptFunctionCall::get_use_default_args() const {
	return use_default_args;
}

void VisualScriptFunctionCall::set_validate(bool p_amount) {
	validate = p_amount;
}

bool VisualScriptFunctionCall::get_validate() const {
	return validate;
}

void VisualScriptFunctionCall::_set_argument_cache(const Dictionary &p_cache) {
	//so everything works in case all else fails
	method_cache = MethodInfo::from_dict(p_cache);
}

Dictionary VisualScriptFunctionCall::_get_argument_cache() const {
	return method_cache;
}

void VisualScriptFunctionCall::_validate_property(PropertyInfo &property) const {
	if (property.name == "base_type") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "base_script") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (property.name == "basic_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (property.name == "singleton") {
		if (call_mode != CALL_MODE_SINGLETON) {
			property.usage = PROPERTY_USAGE_NONE;
		} else {
			List<Engine::Singleton> names;
			Engine::get_singleton()->get_singletons(&names);
			property.hint = PROPERTY_HINT_ENUM;
			String sl;
			for (const Engine::Singleton &E : names) {
				if (!sl.is_empty()) {
					sl += ",";
				}
				sl += E.name;
			}
			property.hint_string = sl;
		}
	}

	if (property.name == "node_path") {
		if (call_mode != CALL_MODE_NODE_PATH) {
			property.usage = PROPERTY_USAGE_NONE;
		} else {
			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string = bnode->get_path(); //convert to long string
			}
		}
	}

	if (property.name == "function") {
		if (call_mode == CALL_MODE_BASIC_TYPE) {
			property.hint = PROPERTY_HINT_METHOD_OF_VARIANT_TYPE;
			property.hint_string = Variant::get_type_name(basic_type);

		} else if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
			property.hint = PROPERTY_HINT_METHOD_OF_SCRIPT;
			property.hint_string = itos(get_visual_script()->get_instance_id());
		} else if (call_mode == CALL_MODE_SINGLETON) {
			Object *obj = Engine::get_singleton()->get_singleton_object(singleton);
			if (obj) {
				property.hint = PROPERTY_HINT_METHOD_OF_INSTANCE;
				property.hint_string = itos(obj->get_instance_id());
			} else {
				property.hint = PROPERTY_HINT_METHOD_OF_BASE_TYPE;
				property.hint_string = base_type; //should be cached
			}
		} else if (call_mode == CALL_MODE_INSTANCE) {
			property.hint = PROPERTY_HINT_METHOD_OF_BASE_TYPE;
			property.hint_string = base_type;

			if (!base_script.is_empty()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {
					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {
					Ref<Script> script = Ref<Resource>(ResourceCache::get(base_script));
					if (script.is_valid()) {
						property.hint = PROPERTY_HINT_METHOD_OF_SCRIPT;
						property.hint_string = itos(script->get_instance_id());
					}
				}
			}

		} else if (call_mode == CALL_MODE_NODE_PATH) {
			Node *node = _get_base_node();
			if (node) {
				property.hint = PROPERTY_HINT_METHOD_OF_INSTANCE;
				property.hint_string = itos(node->get_instance_id());
			} else {
				property.hint = PROPERTY_HINT_METHOD_OF_BASE_TYPE;
				property.hint_string = get_base_type();
			}
		}
	}

	if (property.name == "use_default_args") {
		property.hint = PROPERTY_HINT_RANGE;

		int mc = 0;

		if (call_mode == CALL_MODE_BASIC_TYPE) {
			mc = Variant::get_builtin_method_default_arguments(basic_type, function).size();
		} else {
			MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
			if (mb) {
				mc = mb->get_default_argument_count();
			}
		}

		if (mc == 0) {
			property.usage = PROPERTY_USAGE_NONE; //do not show
		} else {
			property.hint_string = "0," + itos(mc) + ",1";
		}
	}

	if (property.name == "rpc_call_mode") {
		if (call_mode == CALL_MODE_BASIC_TYPE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void VisualScriptFunctionCall::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_base_type", "base_type"), &VisualScriptFunctionCall::set_base_type);
	ClassDB::bind_method(D_METHOD("get_base_type"), &VisualScriptFunctionCall::get_base_type);

	ClassDB::bind_method(D_METHOD("set_base_script", "base_script"), &VisualScriptFunctionCall::set_base_script);
	ClassDB::bind_method(D_METHOD("get_base_script"), &VisualScriptFunctionCall::get_base_script);

	ClassDB::bind_method(D_METHOD("set_basic_type", "basic_type"), &VisualScriptFunctionCall::set_basic_type);
	ClassDB::bind_method(D_METHOD("get_basic_type"), &VisualScriptFunctionCall::get_basic_type);

	ClassDB::bind_method(D_METHOD("set_singleton", "singleton"), &VisualScriptFunctionCall::set_singleton);
	ClassDB::bind_method(D_METHOD("get_singleton"), &VisualScriptFunctionCall::get_singleton);

	ClassDB::bind_method(D_METHOD("set_function", "function"), &VisualScriptFunctionCall::set_function);
	ClassDB::bind_method(D_METHOD("get_function"), &VisualScriptFunctionCall::get_function);

	ClassDB::bind_method(D_METHOD("set_call_mode", "mode"), &VisualScriptFunctionCall::set_call_mode);
	ClassDB::bind_method(D_METHOD("get_call_mode"), &VisualScriptFunctionCall::get_call_mode);

	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &VisualScriptFunctionCall::set_base_path);
	ClassDB::bind_method(D_METHOD("get_base_path"), &VisualScriptFunctionCall::get_base_path);

	ClassDB::bind_method(D_METHOD("set_use_default_args", "amount"), &VisualScriptFunctionCall::set_use_default_args);
	ClassDB::bind_method(D_METHOD("get_use_default_args"), &VisualScriptFunctionCall::get_use_default_args);

	ClassDB::bind_method(D_METHOD("_set_argument_cache", "argument_cache"), &VisualScriptFunctionCall::_set_argument_cache);
	ClassDB::bind_method(D_METHOD("_get_argument_cache"), &VisualScriptFunctionCall::_get_argument_cache);

	ClassDB::bind_method(D_METHOD("set_rpc_call_mode", "mode"), &VisualScriptFunctionCall::set_rpc_call_mode);
	ClassDB::bind_method(D_METHOD("get_rpc_call_mode"), &VisualScriptFunctionCall::get_rpc_call_mode);

	ClassDB::bind_method(D_METHOD("set_validate", "enable"), &VisualScriptFunctionCall::set_validate);
	ClassDB::bind_method(D_METHOD("get_validate"), &VisualScriptFunctionCall::get_validate);

	String bt;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0) {
			bt += ",";
		}

		bt += Variant::get_type_name(Variant::Type(i));
	}

	List<String> script_extensions;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (const String &E : script_extensions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E;
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "call_mode", PROPERTY_HINT_ENUM, "Self,Node Path,Instance,Basic Type,Singleton"), "set_call_mode", "get_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "singleton"), "set_singleton", "get_singleton");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "basic_type", PROPERTY_HINT_ENUM, bt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "argument_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_argument_cache", "_get_argument_cache");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function"), "set_function", "get_function"); //when set, if loaded properly, will override argument count.
	ADD_PROPERTY(PropertyInfo(Variant::INT, "use_default_args"), "set_use_default_args", "get_use_default_args");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "validate"), "set_validate", "get_validate");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rpc_call_mode", PROPERTY_HINT_ENUM, "Disabled,Reliable,Unreliable,ReliableToID,UnreliableToID"), "set_rpc_call_mode", "get_rpc_call_mode"); //when set, if loaded properly, will override argument count.

	BIND_ENUM_CONSTANT(CALL_MODE_SELF);
	BIND_ENUM_CONSTANT(CALL_MODE_NODE_PATH);
	BIND_ENUM_CONSTANT(CALL_MODE_INSTANCE);
	BIND_ENUM_CONSTANT(CALL_MODE_BASIC_TYPE);
	BIND_ENUM_CONSTANT(CALL_MODE_SINGLETON);

	BIND_ENUM_CONSTANT(RPC_DISABLED);
	BIND_ENUM_CONSTANT(RPC_RELIABLE);
	BIND_ENUM_CONSTANT(RPC_UNRELIABLE);
	BIND_ENUM_CONSTANT(RPC_RELIABLE_TO_ID);
	BIND_ENUM_CONSTANT(RPC_UNRELIABLE_TO_ID);
}

class VisualScriptNodeInstanceFunctionCall : public VisualScriptNodeInstance {
public:
	VisualScriptFunctionCall::CallMode call_mode;
	NodePath node_path;
	int input_args;
	bool validate;
	int returns;
	VisualScriptFunctionCall::RPCCallMode rpc_mode;
	StringName function;
	StringName singleton;

	VisualScriptFunctionCall *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	_FORCE_INLINE_ bool call_rpc(Object *p_base, const Variant **p_args, int p_argcount) {
		if (!p_base) {
			return false;
		}

		Node *node = Object::cast_to<Node>(p_base);
		if (!node) {
			return false;
		}

		int to_id = 0;
		//bool reliable = true;

		if (rpc_mode >= VisualScriptFunctionCall::RPC_RELIABLE_TO_ID) {
			to_id = *p_args[0];
			p_args += 1;
			p_argcount -= 1;
			//if (rpc_mode == VisualScriptFunctionCall::RPC_UNRELIABLE_TO_ID) {
			//reliable = false;
			//}
		}
		//else if (rpc_mode == VisualScriptFunctionCall::RPC_UNRELIABLE) {
		//reliable = false;
		//}

		// TODO reliable?
		node->rpcp(to_id, function, p_args, p_argcount);

		return true;
	}

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		switch (call_mode) {
			case VisualScriptFunctionCall::CALL_MODE_SELF: {
				Object *object = instance->get_owner_ptr();

				if (rpc_mode) {
					call_rpc(object, p_inputs, input_args);
				} else if (returns) {
					*p_outputs[0] = object->call(function, p_inputs, input_args, r_error);
				} else {
					object->call(function, p_inputs, input_args, r_error);
				}
			} break;
			case VisualScriptFunctionCall::CALL_MODE_NODE_PATH: {
				Node *node = Object::cast_to<Node>(instance->get_owner_ptr());
				if (!node) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Base object is not a Node!";
					return 0;
				}

				Node *another = node->get_node(node_path);
				if (!another) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Path does not lead Node!";
					return 0;
				}

				if (rpc_mode) {
					call_rpc(node, p_inputs, input_args);
				} else if (returns) {
					*p_outputs[0] = another->call(function, p_inputs, input_args, r_error);
				} else {
					another->call(function, p_inputs, input_args, r_error);
				}

			} break;
			case VisualScriptFunctionCall::CALL_MODE_INSTANCE:
			case VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE: {
				Variant v = *p_inputs[0];

				if (rpc_mode) {
					Object *obj = v;
					if (obj) {
						call_rpc(obj, p_inputs + 1, input_args - 1);
					}
				} else if (returns) {
					if (call_mode == VisualScriptFunctionCall::CALL_MODE_INSTANCE) {
						if (returns >= 2) {
							v.call(function, p_inputs + 1, input_args, *p_outputs[1], r_error);
						} else if (returns == 1) {
							Variant ret;
							v.call(function, p_inputs + 1, input_args, ret, r_error);
						} else {
							r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
							r_error_str = "Invalid returns count for call_mode == CALL_MODE_INSTANCE";
							return 0;
						}
					} else {
						v.call(function, p_inputs + 1, input_args, *p_outputs[0], r_error);
					}
				} else {
					Variant ret;
					v.call(function, p_inputs + 1, input_args, ret, r_error);
				}

				if (call_mode == VisualScriptFunctionCall::CALL_MODE_INSTANCE) {
					*p_outputs[0] = *p_inputs[0];
				}

			} break;
			case VisualScriptFunctionCall::CALL_MODE_SINGLETON: {
				Object *object = Engine::get_singleton()->get_singleton_object(singleton);
				if (!object) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid singleton name: '" + String(singleton) + "'";
					return 0;
				}

				if (rpc_mode) {
					call_rpc(object, p_inputs, input_args);
				} else if (returns) {
					*p_outputs[0] = object->call(function, p_inputs, input_args, r_error);
				} else {
					object->call(function, p_inputs, input_args, r_error);
				}
			} break;
		}

		if (!validate) {
			//ignore call errors if validation is disabled
			r_error.error = Callable::CallError::CALL_OK;
			r_error_str = String();
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptFunctionCall::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceFunctionCall *instance = memnew(VisualScriptNodeInstanceFunctionCall);
	instance->node = this;
	instance->instance = p_instance;
	instance->singleton = singleton;
	instance->function = function;
	instance->call_mode = call_mode;
	instance->returns = get_output_value_port_count();
	instance->node_path = base_path;
	instance->input_args = get_input_value_port_count() - ((call_mode == CALL_MODE_BASIC_TYPE || call_mode == CALL_MODE_INSTANCE) ? 1 : 0);
	instance->rpc_mode = rpc_call_mode;
	instance->validate = validate;
	return instance;
}

VisualScriptFunctionCall::TypeGuess VisualScriptFunctionCall::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	if (p_output == 0 && call_mode == CALL_MODE_INSTANCE) {
		return p_inputs[0];
	}

	return VisualScriptNode::guess_output_type(p_inputs, p_output);
}

VisualScriptFunctionCall::VisualScriptFunctionCall() {
	validate = true;
	call_mode = CALL_MODE_SELF;
	basic_type = Variant::NIL;
	use_default_args = 0;
	base_type = "Object";
	rpc_call_mode = RPC_DISABLED;
}

template <VisualScriptFunctionCall::CallMode cmode>
static Ref<VisualScriptNode> create_function_call_node(const String &p_name) {
	Ref<VisualScriptFunctionCall> node;
	node.instantiate();
	node->set_call_mode(cmode);
	return node;
}

//////////////////////////////////////////
////////////////SET//////////////////////
//////////////////////////////////////////

int VisualScriptPropertySet::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptPropertySet::has_input_sequence_port() const {
	return 1;
}

Node *VisualScriptPropertySet::_get_base_node() const {
#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid()) {
		return nullptr;
	}

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();

	SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);

	if (!scene_tree) {
		return nullptr;
	}

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene) {
		return nullptr;
	}

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node) {
		return nullptr;
	}

	if (!script_node->has_node(base_path)) {
		return nullptr;
	}

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return nullptr;
#endif
}

StringName VisualScriptPropertySet::_get_base_type() const {
	if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
		return get_visual_script()->get_instance_base_type();
	} else if (call_mode == CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path) {
			return path->get_class();
		}
	}

	return base_type;
}

int VisualScriptPropertySet::get_input_value_port_count() const {
	int pc = (call_mode == CALL_MODE_BASIC_TYPE || call_mode == CALL_MODE_INSTANCE) ? 2 : 1;

	return pc;
}

int VisualScriptPropertySet::get_output_value_port_count() const {
	return (call_mode == CALL_MODE_BASIC_TYPE || call_mode == CALL_MODE_INSTANCE) ? 1 : 0;
}

String VisualScriptPropertySet::get_output_sequence_port_text(int p_port) const {
	return String();
}

void VisualScriptPropertySet::_adjust_input_index(PropertyInfo &pinfo) const {
	if (index != StringName()) {
		Variant v;
		Callable::CallError ce;
		Variant::construct(pinfo.type, v, nullptr, 0, ce);
		Variant i = v.get(index);
		pinfo.type = i.get_type();
	}
}

PropertyInfo VisualScriptPropertySet::get_input_value_port_info(int p_idx) const {
	if (call_mode == CALL_MODE_INSTANCE || call_mode == CALL_MODE_BASIC_TYPE) {
		if (p_idx == 0) {
			PropertyInfo pi;
			pi.type = (call_mode == CALL_MODE_INSTANCE ? Variant::OBJECT : basic_type);
			pi.name = "instance";
			return pi;
		}
	}

	List<PropertyInfo> props;
	ClassDB::get_property_list(_get_base_type(), &props, false);
	for (const PropertyInfo &E : props) {
		if (E.name == property) {
			String detail_prop_name = property;
			if (index != StringName()) {
				detail_prop_name += "." + String(index);
			}
			PropertyInfo pinfo = PropertyInfo(E.type, detail_prop_name, E.hint, E.hint_string);
			_adjust_input_index(pinfo);
			return pinfo;
		}
	}

	PropertyInfo pinfo = type_cache;
	_adjust_input_index(pinfo);
	return pinfo;
}

PropertyInfo VisualScriptPropertySet::get_output_value_port_info(int p_idx) const {
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		return PropertyInfo(basic_type, "pass");
	} else if (call_mode == CALL_MODE_INSTANCE) {
		return PropertyInfo(Variant::OBJECT, "pass", PROPERTY_HINT_TYPE_STRING, get_base_type());
	} else {
		return PropertyInfo();
	}
}

String VisualScriptPropertySet::get_caption() const {
	static const char *opname[ASSIGN_OP_MAX] = {
		"Set", "Add", "Subtract", "Multiply", "Divide", "Mod", "ShiftLeft", "ShiftRight", "BitAnd", "BitOr", "BitXor"
	};

	String prop = String(opname[assign_op]) + " " + property;
	if (index != StringName()) {
		prop += "." + String(index);
	}

	return prop;
}

String VisualScriptPropertySet::get_text() const {
	if (!has_input_sequence_port()) {
		return "";
	}
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		return String("On ") + Variant::get_type_name(basic_type);
	} else if (call_mode == CALL_MODE_INSTANCE) {
		return String("On ") + base_type;
	} else if (call_mode == CALL_MODE_NODE_PATH) {
		return " [" + String(base_path.simplified()) + "]";
	} else {
		return "On Self";
	}
}

void VisualScriptPropertySet::_update_base_type() {
	//cache it because this information may not be available on load
	if (call_mode == CALL_MODE_NODE_PATH) {
		Node *node = _get_base_node();
		if (node) {
			base_type = node->get_class();
		}
	} else if (call_mode == CALL_MODE_SELF) {
		if (get_visual_script().is_valid()) {
			base_type = get_visual_script()->get_instance_base_type();
		}
	}
}

void VisualScriptPropertySet::set_basic_type(Variant::Type p_type) {
	if (basic_type == p_type) {
		return;
	}
	basic_type = p_type;

	notify_property_list_changed();
	_update_base_type();
	ports_changed_notify();
}

Variant::Type VisualScriptPropertySet::get_basic_type() const {
	return basic_type;
}

void VisualScriptPropertySet::set_base_type(const StringName &p_type) {
	if (base_type == p_type) {
		return;
	}

	base_type = p_type;
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptPropertySet::get_base_type() const {
	return base_type;
}

void VisualScriptPropertySet::set_base_script(const String &p_path) {
	if (base_script == p_path) {
		return;
	}

	base_script = p_path;
	notify_property_list_changed();
	ports_changed_notify();
}

String VisualScriptPropertySet::get_base_script() const {
	return base_script;
}

void VisualScriptPropertySet::_update_cache() {
	if (!Object::cast_to<SceneTree>(OS::get_singleton()->get_main_loop())) {
		return;
	}

	if (!Engine::get_singleton()->is_editor_hint()) { //only update cache if editor exists, it's pointless otherwise
		return;
	}

	if (call_mode == CALL_MODE_BASIC_TYPE) {
		//not super efficient..

		Variant v;
		Callable::CallError ce;
		Variant::construct(basic_type, v, nullptr, 0, ce);

		List<PropertyInfo> pinfo;
		v.get_property_list(&pinfo);

		for (const PropertyInfo &E : pinfo) {
			if (E.name == property) {
				type_cache = E;
			}
		}

	} else {
		StringName type;
		Ref<Script> script;
		Node *node = nullptr;

		if (call_mode == CALL_MODE_NODE_PATH) {
			node = _get_base_node();
			if (node) {
				type = node->get_class();
				base_type = type; //cache, too
				script = node->get_script();
			}
		} else if (call_mode == CALL_MODE_SELF) {
			if (get_visual_script().is_valid()) {
				type = get_visual_script()->get_instance_base_type();
				base_type = type; //cache, too
				script = get_visual_script();
			}
		} else if (call_mode == CALL_MODE_INSTANCE) {
			type = base_type;
			if (!base_script.is_empty()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {
					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {
					script = Ref<Resource>(ResourceCache::get(base_script));
				} else {
					return;
				}
			}
		}

		List<PropertyInfo> pinfo;

		if (node) {
			node->get_property_list(&pinfo);
		} else {
			ClassDB::get_property_list(type, &pinfo);
		}

		if (script.is_valid()) {
			script->get_script_property_list(&pinfo);
		}

		for (const PropertyInfo &E : pinfo) {
			if (E.name == property) {
				type_cache = E;
				return;
			}
		}
	}
}

void VisualScriptPropertySet::set_property(const StringName &p_type) {
	if (property == p_type) {
		return;
	}

	property = p_type;
	index = StringName();
	_update_cache();
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptPropertySet::get_property() const {
	return property;
}

void VisualScriptPropertySet::set_base_path(const NodePath &p_type) {
	if (base_path == p_type) {
		return;
	}

	base_path = p_type;
	_update_base_type();
	notify_property_list_changed();
	ports_changed_notify();
}

NodePath VisualScriptPropertySet::get_base_path() const {
	return base_path;
}

void VisualScriptPropertySet::set_call_mode(CallMode p_mode) {
	if (call_mode == p_mode) {
		return;
	}

	call_mode = p_mode;
	_update_base_type();
	notify_property_list_changed();
	ports_changed_notify();
}

VisualScriptPropertySet::CallMode VisualScriptPropertySet::get_call_mode() const {
	return call_mode;
}

void VisualScriptPropertySet::_set_type_cache(const Dictionary &p_type) {
	type_cache = PropertyInfo::from_dict(p_type);
}

Dictionary VisualScriptPropertySet::_get_type_cache() const {
	return type_cache;
}

void VisualScriptPropertySet::set_index(const StringName &p_type) {
	if (index == p_type) {
		return;
	}
	index = p_type;
	_update_cache();
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptPropertySet::get_index() const {
	return index;
}

void VisualScriptPropertySet::set_assign_op(AssignOp p_op) {
	ERR_FAIL_INDEX(p_op, ASSIGN_OP_MAX);
	if (assign_op == p_op) {
		return;
	}

	assign_op = p_op;
	_update_cache();
	notify_property_list_changed();
	ports_changed_notify();
}

VisualScriptPropertySet::AssignOp VisualScriptPropertySet::get_assign_op() const {
	return assign_op;
}

void VisualScriptPropertySet::_validate_property(PropertyInfo &property) const {
	if (property.name == "base_type") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "base_script") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (property.name == "basic_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (property.name == "node_path") {
		if (call_mode != CALL_MODE_NODE_PATH) {
			property.usage = PROPERTY_USAGE_NONE;
		} else {
			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string = bnode->get_path(); //convert to long string
			}
		}
	}

	if (property.name == "property") {
		if (call_mode == CALL_MODE_BASIC_TYPE) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE;
			property.hint_string = Variant::get_type_name(basic_type);

		} else if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
			property.hint_string = itos(get_visual_script()->get_instance_id());
		} else if (call_mode == CALL_MODE_INSTANCE) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
			property.hint_string = base_type;

			if (!base_script.is_empty()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {
					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {
					Ref<Script> script = Ref<Resource>(ResourceCache::get(base_script));
					if (script.is_valid()) {
						property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
						property.hint_string = itos(script->get_instance_id());
					}
				}
			}

		} else if (call_mode == CALL_MODE_NODE_PATH) {
			Node *node = _get_base_node();
			if (node) {
				property.hint = PROPERTY_HINT_PROPERTY_OF_INSTANCE;
				property.hint_string = itos(node->get_instance_id());
			} else {
				property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
				property.hint_string = get_base_type();
			}
		}
	}

	if (property.name == "index") {
		Callable::CallError ce;
		Variant v;
		Variant::construct(type_cache.type, v, nullptr, 0, ce);
		List<PropertyInfo> plist;
		v.get_property_list(&plist);
		String options = "";
		for (const PropertyInfo &E : plist) {
			options += "," + E.name;
		}

		property.hint = PROPERTY_HINT_ENUM;
		property.hint_string = options;
		property.type = Variant::STRING;
		if (options.is_empty()) {
			property.usage = PROPERTY_USAGE_NONE; //hide if type has no usable index
		}
	}
}

void VisualScriptPropertySet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_base_type", "base_type"), &VisualScriptPropertySet::set_base_type);
	ClassDB::bind_method(D_METHOD("get_base_type"), &VisualScriptPropertySet::get_base_type);

	ClassDB::bind_method(D_METHOD("set_base_script", "base_script"), &VisualScriptPropertySet::set_base_script);
	ClassDB::bind_method(D_METHOD("get_base_script"), &VisualScriptPropertySet::get_base_script);

	ClassDB::bind_method(D_METHOD("set_basic_type", "basic_type"), &VisualScriptPropertySet::set_basic_type);
	ClassDB::bind_method(D_METHOD("get_basic_type"), &VisualScriptPropertySet::get_basic_type);

	ClassDB::bind_method(D_METHOD("_set_type_cache", "type_cache"), &VisualScriptPropertySet::_set_type_cache);
	ClassDB::bind_method(D_METHOD("_get_type_cache"), &VisualScriptPropertySet::_get_type_cache);

	ClassDB::bind_method(D_METHOD("set_property", "property"), &VisualScriptPropertySet::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &VisualScriptPropertySet::get_property);

	ClassDB::bind_method(D_METHOD("set_call_mode", "mode"), &VisualScriptPropertySet::set_call_mode);
	ClassDB::bind_method(D_METHOD("get_call_mode"), &VisualScriptPropertySet::get_call_mode);

	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &VisualScriptPropertySet::set_base_path);
	ClassDB::bind_method(D_METHOD("get_base_path"), &VisualScriptPropertySet::get_base_path);

	ClassDB::bind_method(D_METHOD("set_index", "index"), &VisualScriptPropertySet::set_index);
	ClassDB::bind_method(D_METHOD("get_index"), &VisualScriptPropertySet::get_index);

	ClassDB::bind_method(D_METHOD("set_assign_op", "assign_op"), &VisualScriptPropertySet::set_assign_op);
	ClassDB::bind_method(D_METHOD("get_assign_op"), &VisualScriptPropertySet::get_assign_op);

	String bt;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0) {
			bt += ",";
		}

		bt += Variant::get_type_name(Variant::Type(i));
	}

	List<String> script_extensions;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (const String &E : script_extensions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E;
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "set_mode", PROPERTY_HINT_ENUM, "Self,Node Path,Instance,Basic Type"), "set_call_mode", "get_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_type_cache", "_get_type_cache");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "basic_type", PROPERTY_HINT_ENUM, bt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property"), "set_property", "get_property");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "index"), "set_index", "get_index");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "assign_op", PROPERTY_HINT_ENUM, "Assign,Add,Sub,Mul,Div,Mod,ShiftLeft,ShiftRight,BitAnd,BitOr,Bitxor"), "set_assign_op", "get_assign_op");

	BIND_ENUM_CONSTANT(CALL_MODE_SELF);
	BIND_ENUM_CONSTANT(CALL_MODE_NODE_PATH);
	BIND_ENUM_CONSTANT(CALL_MODE_INSTANCE);
	BIND_ENUM_CONSTANT(CALL_MODE_BASIC_TYPE);

	BIND_ENUM_CONSTANT(ASSIGN_OP_NONE);
	BIND_ENUM_CONSTANT(ASSIGN_OP_ADD);
	BIND_ENUM_CONSTANT(ASSIGN_OP_SUB);
	BIND_ENUM_CONSTANT(ASSIGN_OP_MUL);
	BIND_ENUM_CONSTANT(ASSIGN_OP_DIV);
	BIND_ENUM_CONSTANT(ASSIGN_OP_MOD);
	BIND_ENUM_CONSTANT(ASSIGN_OP_SHIFT_LEFT);
	BIND_ENUM_CONSTANT(ASSIGN_OP_SHIFT_RIGHT);
	BIND_ENUM_CONSTANT(ASSIGN_OP_BIT_AND);
	BIND_ENUM_CONSTANT(ASSIGN_OP_BIT_OR);
	BIND_ENUM_CONSTANT(ASSIGN_OP_BIT_XOR);
}

class VisualScriptNodeInstancePropertySet : public VisualScriptNodeInstance {
public:
	VisualScriptPropertySet::CallMode call_mode;
	NodePath node_path;
	StringName property;

	VisualScriptPropertySet *node;
	VisualScriptInstance *instance;
	VisualScriptPropertySet::AssignOp assign_op;
	StringName index;
	bool needs_get;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	_FORCE_INLINE_ void _process_get(Variant &source, const Variant &p_argument, bool &valid) {
		if (index != StringName() && assign_op == VisualScriptPropertySet::ASSIGN_OP_NONE) {
			source.set_named(index, p_argument, valid);
		} else {
			Variant value;
			if (index != StringName()) {
				value = source.get_named(index, valid);
			} else {
				value = source;
			}

			switch (assign_op) {
				case VisualScriptPropertySet::ASSIGN_OP_NONE: {
					//should never get here
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_ADD: {
					value = Variant::evaluate(Variant::OP_ADD, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_SUB: {
					value = Variant::evaluate(Variant::OP_SUBTRACT, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_MUL: {
					value = Variant::evaluate(Variant::OP_MULTIPLY, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_DIV: {
					value = Variant::evaluate(Variant::OP_DIVIDE, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_MOD: {
					value = Variant::evaluate(Variant::OP_MODULE, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_SHIFT_LEFT: {
					value = Variant::evaluate(Variant::OP_SHIFT_LEFT, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_SHIFT_RIGHT: {
					value = Variant::evaluate(Variant::OP_SHIFT_RIGHT, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_BIT_AND: {
					value = Variant::evaluate(Variant::OP_BIT_AND, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_BIT_OR: {
					value = Variant::evaluate(Variant::OP_BIT_OR, value, p_argument);
				} break;
				case VisualScriptPropertySet::ASSIGN_OP_BIT_XOR: {
					value = Variant::evaluate(Variant::OP_BIT_XOR, value, p_argument);
				} break;
				default: {
				}
			}

			if (index != StringName()) {
				source.set_named(index, value, valid);
			} else {
				source = value;
			}
		}
	}

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		switch (call_mode) {
			case VisualScriptPropertySet::CALL_MODE_SELF: {
				Object *object = instance->get_owner_ptr();

				bool valid;

				if (needs_get) {
					Variant value = object->get(property, &valid);
					_process_get(value, *p_inputs[0], valid);
					object->set(property, value, &valid);
				} else {
					object->set(property, *p_inputs[0], &valid);
				}

				if (!valid) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid set value '" + String(*p_inputs[0]) + "' on property '" + String(property) + "' of type " + object->get_class();
				}
			} break;
			case VisualScriptPropertySet::CALL_MODE_NODE_PATH: {
				Node *node = Object::cast_to<Node>(instance->get_owner_ptr());
				if (!node) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Base object is not a Node!";
					return 0;
				}

				Node *another = node->get_node(node_path);
				if (!another) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Path does not lead Node!";
					return 0;
				}

				bool valid;

				if (needs_get) {
					Variant value = another->get(property, &valid);
					_process_get(value, *p_inputs[0], valid);
					another->set(property, value, &valid);
				} else {
					another->set(property, *p_inputs[0], &valid);
				}

				if (!valid) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid set value '" + String(*p_inputs[0]) + "' on property '" + String(property) + "' of type " + another->get_class();
				}

			} break;
			case VisualScriptPropertySet::CALL_MODE_INSTANCE:
			case VisualScriptPropertySet::CALL_MODE_BASIC_TYPE: {
				Variant v = *p_inputs[0];

				bool valid;

				if (needs_get) {
					Variant value = v.get_named(property, valid);
					_process_get(value, *p_inputs[1], valid);
					v.set_named(property, value, valid);

				} else {
					v.set_named(property, *p_inputs[1], valid);
				}

				if (!valid) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid set value '" + String(*p_inputs[1]) + "' (" + Variant::get_type_name(p_inputs[1]->get_type()) + ") on property '" + String(property) + "' of type " + Variant::get_type_name(v.get_type());
				}

				*p_outputs[0] = v;

			} break;
		}
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptPropertySet::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstancePropertySet *instance = memnew(VisualScriptNodeInstancePropertySet);
	instance->node = this;
	instance->instance = p_instance;
	instance->property = property;
	instance->call_mode = call_mode;
	instance->node_path = base_path;
	instance->assign_op = assign_op;
	instance->index = index;
	instance->needs_get = index != StringName() || assign_op != ASSIGN_OP_NONE;
	return instance;
}

VisualScriptPropertySet::TypeGuess VisualScriptPropertySet::guess_output_type(TypeGuess *p_inputs, int p_output) const {
	if (p_output == 0 && call_mode == CALL_MODE_INSTANCE) {
		return p_inputs[0];
	}

	return VisualScriptNode::guess_output_type(p_inputs, p_output);
}

VisualScriptPropertySet::VisualScriptPropertySet() {
	assign_op = ASSIGN_OP_NONE;
	call_mode = CALL_MODE_SELF;
	base_type = "Object";
	basic_type = Variant::NIL;
}

template <VisualScriptPropertySet::CallMode cmode>
static Ref<VisualScriptNode> create_property_set_node(const String &p_name) {
	Ref<VisualScriptPropertySet> node;
	node.instantiate();
	node->set_call_mode(cmode);
	return node;
}

//////////////////////////////////////////
////////////////GET//////////////////////
//////////////////////////////////////////

int VisualScriptPropertyGet::get_output_sequence_port_count() const {
	return 0; // (call_mode==CALL_MODE_SELF || call_mode==CALL_MODE_NODE_PATH)?0:1;
}

bool VisualScriptPropertyGet::has_input_sequence_port() const {
	return false; //(call_mode==CALL_MODE_SELF || call_mode==CALL_MODE_NODE_PATH)?false:true;
}

void VisualScriptPropertyGet::_update_base_type() {
	//cache it because this information may not be available on load
	if (call_mode == CALL_MODE_NODE_PATH) {
		Node *node = _get_base_node();
		if (node) {
			base_type = node->get_class();
		}
	} else if (call_mode == CALL_MODE_SELF) {
		if (get_visual_script().is_valid()) {
			base_type = get_visual_script()->get_instance_base_type();
		}
	}
}

Node *VisualScriptPropertyGet::_get_base_node() const {
#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid()) {
		return nullptr;
	}

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();

	SceneTree *scene_tree = Object::cast_to<SceneTree>(main_loop);

	if (!scene_tree) {
		return nullptr;
	}

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene) {
		return nullptr;
	}

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node) {
		return nullptr;
	}

	if (!script_node->has_node(base_path)) {
		return nullptr;
	}

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return nullptr;
#endif
}

StringName VisualScriptPropertyGet::_get_base_type() const {
	if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
		return get_visual_script()->get_instance_base_type();
	} else if (call_mode == CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path) {
			return path->get_class();
		}
	}

	return base_type;
}

int VisualScriptPropertyGet::get_input_value_port_count() const {
	return (call_mode == CALL_MODE_BASIC_TYPE || call_mode == CALL_MODE_INSTANCE) ? 1 : 0;
}

int VisualScriptPropertyGet::get_output_value_port_count() const {
	int pc = (call_mode == CALL_MODE_BASIC_TYPE || call_mode == CALL_MODE_INSTANCE) ? 2 : 1;

	return pc;
}

String VisualScriptPropertyGet::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptPropertyGet::get_input_value_port_info(int p_idx) const {
	if (call_mode == CALL_MODE_INSTANCE || call_mode == CALL_MODE_BASIC_TYPE) {
		if (p_idx == 0) {
			PropertyInfo pi;
			pi.type = (call_mode == CALL_MODE_INSTANCE ? Variant::OBJECT : basic_type);
			pi.name = (call_mode == CALL_MODE_INSTANCE ? String("instance") : Variant::get_type_name(basic_type).to_lower());
			return pi;
		}
	}
	return PropertyInfo();
}

PropertyInfo VisualScriptPropertyGet::get_output_value_port_info(int p_idx) const {
	if (call_mode == CALL_MODE_BASIC_TYPE && p_idx == 0) {
		return PropertyInfo(basic_type, "pass");
	} else if (call_mode == CALL_MODE_INSTANCE && p_idx == 0) {
		return PropertyInfo(Variant::OBJECT, "pass", PROPERTY_HINT_TYPE_STRING, get_base_type());
	} else {
		List<PropertyInfo> props;
		ClassDB::get_property_list(_get_base_type(), &props, false);
		for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
			if (E->get().name == property) {
				PropertyInfo pinfo = PropertyInfo(E->get().type, String(property) + "." + String(index), E->get().hint, E->get().hint_string);
				_adjust_input_index(pinfo);
				return pinfo;
			}
		}
	}

	PropertyInfo pinfo = PropertyInfo(type_cache, "value");
	_adjust_input_index(pinfo);
	return pinfo;
}

String VisualScriptPropertyGet::get_caption() const {
	String prop = String("Get ") + property;
	if (index != StringName()) {
		prop += "." + String(index);
	}

	return prop;
}

String VisualScriptPropertyGet::get_text() const {
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		return String("On ") + Variant::get_type_name(basic_type);
	} else if (call_mode == CALL_MODE_INSTANCE) {
		return String("On ") + base_type;
	} else if (call_mode == CALL_MODE_NODE_PATH) {
		return " [" + String(base_path.simplified()) + "]";
	} else {
		return "On Self";
	}
}

void VisualScriptPropertyGet::set_base_type(const StringName &p_type) {
	if (base_type == p_type) {
		return;
	}

	base_type = p_type;
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptPropertyGet::get_base_type() const {
	return base_type;
}

void VisualScriptPropertyGet::set_base_script(const String &p_path) {
	if (base_script == p_path) {
		return;
	}

	base_script = p_path;
	notify_property_list_changed();
	ports_changed_notify();
}

String VisualScriptPropertyGet::get_base_script() const {
	return base_script;
}

void VisualScriptPropertyGet::_update_cache() {
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		//not super efficient..

		Variant v;
		Callable::CallError ce;
		Variant::construct(basic_type, v, nullptr, 0, ce);

		List<PropertyInfo> pinfo;
		v.get_property_list(&pinfo);

		for (const PropertyInfo &E : pinfo) {
			if (E.name == property) {
				type_cache = E.type;
				return;
			}
		}

	} else {
		StringName type;
		Ref<Script> script;
		Node *node = nullptr;

		if (call_mode == CALL_MODE_NODE_PATH) {
			node = _get_base_node();
			if (node) {
				type = node->get_class();
				base_type = type; //cache, too
				script = node->get_script();
			}
		} else if (call_mode == CALL_MODE_SELF) {
			if (get_visual_script().is_valid()) {
				type = get_visual_script()->get_instance_base_type();
				base_type = type; //cache, too
				script = get_visual_script();
			}
		} else if (call_mode == CALL_MODE_INSTANCE) {
			type = base_type;
			if (!base_script.is_empty()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {
					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {
					script = Ref<Resource>(ResourceCache::get(base_script));
				} else {
					return;
				}
			}
		}

		bool valid = false;

		Variant::Type type_ret;

		type_ret = ClassDB::get_property_type(base_type, property, &valid);

		if (valid) {
			type_cache = type_ret;
			return; //all dandy
		}

		if (node) {
			Variant prop = node->get(property, &valid);
			if (valid) {
				type_cache = prop.get_type();
				return; //all dandy again
			}
		}

		if (script.is_valid()) {
			type_ret = script->get_static_property_type(property, &valid);

			if (valid) {
				type_cache = type_ret;
				return; //all dandy
			}
		}
	}
}

void VisualScriptPropertyGet::set_property(const StringName &p_type) {
	if (property == p_type) {
		return;
	}

	property = p_type;

	_update_cache();
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptPropertyGet::get_property() const {
	return property;
}

void VisualScriptPropertyGet::set_base_path(const NodePath &p_type) {
	if (base_path == p_type) {
		return;
	}

	base_path = p_type;
	notify_property_list_changed();
	_update_base_type();
	ports_changed_notify();
}

NodePath VisualScriptPropertyGet::get_base_path() const {
	return base_path;
}

void VisualScriptPropertyGet::set_call_mode(CallMode p_mode) {
	if (call_mode == p_mode) {
		return;
	}

	call_mode = p_mode;
	notify_property_list_changed();
	_update_base_type();
	ports_changed_notify();
}

VisualScriptPropertyGet::CallMode VisualScriptPropertyGet::get_call_mode() const {
	return call_mode;
}

void VisualScriptPropertyGet::set_basic_type(Variant::Type p_type) {
	if (basic_type == p_type) {
		return;
	}
	basic_type = p_type;

	notify_property_list_changed();
	ports_changed_notify();
}

Variant::Type VisualScriptPropertyGet::get_basic_type() const {
	return basic_type;
}

void VisualScriptPropertyGet::_set_type_cache(Variant::Type p_type) {
	type_cache = p_type;
}

Variant::Type VisualScriptPropertyGet::_get_type_cache() const {
	return type_cache;
}

void VisualScriptPropertyGet::_adjust_input_index(PropertyInfo &pinfo) const {
	if (index != StringName()) {
		Variant v;
		Callable::CallError ce;
		Variant::construct(pinfo.type, v, nullptr, 0, ce);
		Variant i = v.get(index);
		pinfo.type = i.get_type();
		pinfo.name = String(property) + "." + index;
	} else {
		pinfo.name = String(property);
	}
}

void VisualScriptPropertyGet::set_index(const StringName &p_type) {
	if (index == p_type) {
		return;
	}
	index = p_type;
	_update_cache();
	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptPropertyGet::get_index() const {
	return index;
}

void VisualScriptPropertyGet::_validate_property(PropertyInfo &property) const {
	if (property.name == "base_type") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL;
		}
	}

	if (property.name == "base_script") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (property.name == "basic_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}

	if (property.name == "node_path") {
		if (call_mode != CALL_MODE_NODE_PATH) {
			property.usage = PROPERTY_USAGE_NONE;
		} else {
			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string = bnode->get_path(); //convert to long string
			}
		}
	}

	if (property.name == "property") {
		if (call_mode == CALL_MODE_BASIC_TYPE) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE;
			property.hint_string = Variant::get_type_name(basic_type);

		} else if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
			property.hint_string = itos(get_visual_script()->get_instance_id());
		} else if (call_mode == CALL_MODE_INSTANCE) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
			property.hint_string = base_type;

			if (!base_script.is_empty()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {
					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {
					Ref<Script> script = Ref<Resource>(ResourceCache::get(base_script));
					if (script.is_valid()) {
						property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
						property.hint_string = itos(script->get_instance_id());
					}
				}
			}
		} else if (call_mode == CALL_MODE_NODE_PATH) {
			Node *node = _get_base_node();
			if (node) {
				property.hint = PROPERTY_HINT_PROPERTY_OF_INSTANCE;
				property.hint_string = itos(node->get_instance_id());
			} else {
				property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
				property.hint_string = get_base_type();
			}
		}
	}

	if (property.name == "index") {
		Callable::CallError ce;
		Variant v;
		Variant::construct(type_cache, v, nullptr, 0, ce);
		List<PropertyInfo> plist;
		v.get_property_list(&plist);
		String options = "";
		for (const PropertyInfo &E : plist) {
			options += "," + E.name;
		}

		property.hint = PROPERTY_HINT_ENUM;
		property.hint_string = options;
		property.type = Variant::STRING;
		if (options.is_empty()) {
			property.usage = PROPERTY_USAGE_NONE; //hide if type has no usable index
		}
	}
}

void VisualScriptPropertyGet::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_base_type", "base_type"), &VisualScriptPropertyGet::set_base_type);
	ClassDB::bind_method(D_METHOD("get_base_type"), &VisualScriptPropertyGet::get_base_type);

	ClassDB::bind_method(D_METHOD("set_base_script", "base_script"), &VisualScriptPropertyGet::set_base_script);
	ClassDB::bind_method(D_METHOD("get_base_script"), &VisualScriptPropertyGet::get_base_script);

	ClassDB::bind_method(D_METHOD("set_basic_type", "basic_type"), &VisualScriptPropertyGet::set_basic_type);
	ClassDB::bind_method(D_METHOD("get_basic_type"), &VisualScriptPropertyGet::get_basic_type);

	ClassDB::bind_method(D_METHOD("_set_type_cache", "type_cache"), &VisualScriptPropertyGet::_set_type_cache);
	ClassDB::bind_method(D_METHOD("_get_type_cache"), &VisualScriptPropertyGet::_get_type_cache);

	ClassDB::bind_method(D_METHOD("set_property", "property"), &VisualScriptPropertyGet::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &VisualScriptPropertyGet::get_property);

	ClassDB::bind_method(D_METHOD("set_call_mode", "mode"), &VisualScriptPropertyGet::set_call_mode);
	ClassDB::bind_method(D_METHOD("get_call_mode"), &VisualScriptPropertyGet::get_call_mode);

	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &VisualScriptPropertyGet::set_base_path);
	ClassDB::bind_method(D_METHOD("get_base_path"), &VisualScriptPropertyGet::get_base_path);

	ClassDB::bind_method(D_METHOD("set_index", "index"), &VisualScriptPropertyGet::set_index);
	ClassDB::bind_method(D_METHOD("get_index"), &VisualScriptPropertyGet::get_index);

	String bt;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0) {
			bt += ",";
		}

		bt += Variant::get_type_name(Variant::Type(i));
	}

	List<String> script_extensions;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (const String &E : script_extensions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "." + E;
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "set_mode", PROPERTY_HINT_ENUM, "Self,Node Path,Instance,Basic Type"), "set_call_mode", "get_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_type_cache", "_get_type_cache");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "basic_type", PROPERTY_HINT_ENUM, bt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property"), "set_property", "get_property");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "index", PROPERTY_HINT_ENUM), "set_index", "get_index");

	BIND_ENUM_CONSTANT(CALL_MODE_SELF);
	BIND_ENUM_CONSTANT(CALL_MODE_NODE_PATH);
	BIND_ENUM_CONSTANT(CALL_MODE_INSTANCE);
	BIND_ENUM_CONSTANT(CALL_MODE_BASIC_TYPE);
}

class VisualScriptNodeInstancePropertyGet : public VisualScriptNodeInstance {
public:
	VisualScriptPropertyGet::CallMode call_mode;
	NodePath node_path;
	StringName property;
	StringName index;

	VisualScriptPropertyGet *node;
	VisualScriptInstance *instance;

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		switch (call_mode) {
			case VisualScriptPropertyGet::CALL_MODE_SELF: {
				Object *object = instance->get_owner_ptr();

				bool valid;

				*p_outputs[0] = object->get(property, &valid);

				if (index != StringName()) {
					*p_outputs[0] = p_outputs[0]->get_named(index, valid);
				}

				if (!valid) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Invalid index property name.");
					return 0;
				}
			} break;
			case VisualScriptPropertyGet::CALL_MODE_NODE_PATH: {
				Node *node = Object::cast_to<Node>(instance->get_owner_ptr());
				if (!node) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Base object is not a Node!");
					return 0;
				}

				Node *another = node->get_node(node_path);
				if (!another) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Path does not lead Node!");
					return 0;
				}

				bool valid;

				*p_outputs[0] = another->get(property, &valid);

				if (index != StringName()) {
					*p_outputs[0] = p_outputs[0]->get_named(index, valid);
				}

				if (!valid) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = vformat(RTR("Invalid index property name '%s' in node %s."), String(property), another->get_name());
					return 0;
				}

			} break;
			default: {
				bool valid;
				Variant v = *p_inputs[0];

				*p_outputs[1] = v.get(property, &valid);
				if (index != StringName()) {
					*p_outputs[1] = p_outputs[1]->get_named(index, valid);
				}

				if (!valid) {
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Invalid index property name.");
				}

				*p_outputs[0] = v;
			};
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptPropertyGet::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstancePropertyGet *instance = memnew(VisualScriptNodeInstancePropertyGet);
	instance->node = this;
	instance->instance = p_instance;
	instance->property = property;
	instance->call_mode = call_mode;
	instance->node_path = base_path;
	instance->index = index;

	return instance;
}

VisualScriptPropertyGet::VisualScriptPropertyGet() {
	call_mode = CALL_MODE_SELF;
	base_type = "Object";
	basic_type = Variant::NIL;
	type_cache = Variant::NIL;
}

template <VisualScriptPropertyGet::CallMode cmode>
static Ref<VisualScriptNode> create_property_get_node(const String &p_name) {
	Ref<VisualScriptPropertyGet> node;
	node.instantiate();
	node->set_call_mode(cmode);
	return node;
}

//////////////////////////////////////////
////////////////EMIT//////////////////////
//////////////////////////////////////////

int VisualScriptEmitSignal::get_output_sequence_port_count() const {
	return 1;
}

bool VisualScriptEmitSignal::has_input_sequence_port() const {
	return true;
}

int VisualScriptEmitSignal::get_input_value_port_count() const {
	Ref<VisualScript> vs = get_visual_script();
	if (vs.is_valid()) {
		if (!vs->has_custom_signal(name)) {
			return 0;
		}

		return vs->custom_signal_get_argument_count(name);
	}

	return 0;
}

int VisualScriptEmitSignal::get_output_value_port_count() const {
	return 0;
}

String VisualScriptEmitSignal::get_output_sequence_port_text(int p_port) const {
	return String();
}

PropertyInfo VisualScriptEmitSignal::get_input_value_port_info(int p_idx) const {
	Ref<VisualScript> vs = get_visual_script();
	if (vs.is_valid()) {
		if (!vs->has_custom_signal(name)) {
			return PropertyInfo();
		}

		return PropertyInfo(vs->custom_signal_get_argument_type(name, p_idx), vs->custom_signal_get_argument_name(name, p_idx));
	}

	return PropertyInfo();
}

PropertyInfo VisualScriptEmitSignal::get_output_value_port_info(int p_idx) const {
	return PropertyInfo();
}

String VisualScriptEmitSignal::get_caption() const {
	return "Emit " + String(name);
}

void VisualScriptEmitSignal::set_signal(const StringName &p_type) {
	if (name == p_type) {
		return;
	}

	name = p_type;

	notify_property_list_changed();
	ports_changed_notify();
}

StringName VisualScriptEmitSignal::get_signal() const {
	return name;
}

void VisualScriptEmitSignal::_validate_property(PropertyInfo &property) const {
	if (property.name == "signal") {
		property.hint = PROPERTY_HINT_ENUM;

		List<StringName> sigs;
		List<MethodInfo> base_sigs;

		Ref<VisualScript> vs = get_visual_script();
		if (vs.is_valid()) {
			vs->get_custom_signal_list(&sigs);
			ClassDB::get_signal_list(vs->get_instance_base_type(), &base_sigs);
		}

		String ml;
		for (const StringName &E : sigs) {
			if (!ml.is_empty()) {
				ml += ",";
			}
			ml += E;
		}
		for (const MethodInfo &E : base_sigs) {
			if (!ml.is_empty()) {
				ml += ",";
			}
			ml += E.name;
		}

		property.hint_string = ml;
	}
}

void VisualScriptEmitSignal::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_signal", "name"), &VisualScriptEmitSignal::set_signal);
	ClassDB::bind_method(D_METHOD("get_signal"), &VisualScriptEmitSignal::get_signal);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "signal"), "set_signal", "get_signal");
}

class VisualScriptNodeInstanceEmitSignal : public VisualScriptNodeInstance {
public:
	VisualScriptEmitSignal *node;
	VisualScriptInstance *instance;
	int argcount;
	StringName name;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Callable::CallError &r_error, String &r_error_str) {
		Object *obj = instance->get_owner_ptr();

		obj->emit_signal(name, p_inputs, argcount);

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptEmitSignal::instantiate(VisualScriptInstance *p_instance) {
	VisualScriptNodeInstanceEmitSignal *instance = memnew(VisualScriptNodeInstanceEmitSignal);
	instance->node = this;
	instance->instance = p_instance;
	instance->name = name;
	instance->argcount = get_input_value_port_count();
	return instance;
}

VisualScriptEmitSignal::VisualScriptEmitSignal() {
}

static Ref<VisualScriptNode> create_basic_type_call_node(const String &p_name) {
	Vector<String> path = p_name.split("/");
	ERR_FAIL_COND_V(path.size() < 4, Ref<VisualScriptNode>());
	String base_type = path[2];
	String method = path[3];

	Ref<VisualScriptFunctionCall> node;
	node.instantiate();

	Variant::Type type = Variant::VARIANT_MAX;

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (Variant::get_type_name(Variant::Type(i)) == base_type) {
			type = Variant::Type(i);
			break;
		}
	}

	ERR_FAIL_COND_V(type == Variant::VARIANT_MAX, Ref<VisualScriptNode>());

	node->set_call_mode(VisualScriptFunctionCall::CALL_MODE_BASIC_TYPE);
	node->set_basic_type(type);
	node->set_function(method);

	return node;
}

void register_visual_script_func_nodes() {
	VisualScriptLanguage::singleton->add_register_func("functions/call", create_node_generic<VisualScriptFunctionCall>);
	VisualScriptLanguage::singleton->add_register_func("functions/set", create_node_generic<VisualScriptPropertySet>);
	VisualScriptLanguage::singleton->add_register_func("functions/get", create_node_generic<VisualScriptPropertyGet>);

	//VisualScriptLanguage::singleton->add_register_func("functions/call_script/call_self",create_script_call_node<VisualScriptScriptCall::CALL_MODE_SELF>);
	//VisualScriptLanguage::singleton->add_register_func("functions/call_script/call_node",create_script_call_node<VisualScriptScriptCall::CALL_MODE_NODE_PATH>);
	VisualScriptLanguage::singleton->add_register_func("functions/emit_signal", create_node_generic<VisualScriptEmitSignal>);

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		Variant::Type t = Variant::Type(i);
		String type_name = Variant::get_type_name(t);
		Callable::CallError ce;
		Variant vt;
		Variant::construct(t, vt, nullptr, 0, ce);
		List<MethodInfo> ml;
		vt.get_method_list(&ml);

		for (const MethodInfo &E : ml) {
			VisualScriptLanguage::singleton->add_register_func("functions/by_type/" + type_name + "/" + E.name, create_basic_type_call_node);
		}
	}
}
