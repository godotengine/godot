/*************************************************************************/
/*  visual_script_func_nodes.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "visual_script_func_nodes.h"

#include "global_config.h"
#include "io/resource_loader.h"
#include "os/os.h"
#include "scene/main/node.h"
#include "scene/main/scene_main_loop.h"
#include "visual_script_nodes.h"

//////////////////////////////////////////
////////////////CALL//////////////////////
//////////////////////////////////////////

int VisualScriptFunctionCall::get_output_sequence_port_count() const {

	if (method_cache.flags & METHOD_FLAG_CONST || call_mode == CALL_MODE_BASIC_TYPE)
		return 0;
	else
		return 1;
}

bool VisualScriptFunctionCall::has_input_sequence_port() const {

	if (method_cache.flags & METHOD_FLAG_CONST || call_mode == CALL_MODE_BASIC_TYPE)
		return false;
	else
		return true;
}
#ifdef TOOLS_ENABLED

static Node *_find_script_node(Node *p_edited_scene, Node *p_current_node, const Ref<Script> &script) {

	if (p_edited_scene != p_current_node && p_current_node->get_owner() != p_edited_scene)
		return NULL;

	Ref<Script> scr = p_current_node->get_script();

	if (scr.is_valid() && scr == script)
		return p_current_node;

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *n = _find_script_node(p_edited_scene, p_current_node->get_child(i), script);
		if (n)
			return n;
	}

	return NULL;
}

#endif
Node *VisualScriptFunctionCall::_get_base_node() const {

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid())
		return NULL;

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}

StringName VisualScriptFunctionCall::_get_base_type() const {

	if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else if (call_mode == CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path)
			return path->get_class();
	}

	return base_type;
}

int VisualScriptFunctionCall::get_input_value_port_count() const {

	if (call_mode == CALL_MODE_BASIC_TYPE) {

		Vector<StringName> names = Variant::get_method_argument_names(basic_type, function);
		return names.size() + (rpc_call_mode >= RPC_RELIABLE_TO_ID ? 1 : 0) + 1;

	} else {

		MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
		if (mb) {
			return mb->get_argument_count() + (call_mode == CALL_MODE_INSTANCE ? 1 : 0) + (rpc_call_mode >= RPC_RELIABLE_TO_ID ? 1 : 0) - use_default_args;
		}

		return method_cache.arguments.size() + (call_mode == CALL_MODE_INSTANCE ? 1 : 0) + (rpc_call_mode >= RPC_RELIABLE_TO_ID ? 1 : 0) - use_default_args;
	}
}
int VisualScriptFunctionCall::get_output_value_port_count() const {

	if (call_mode == CALL_MODE_BASIC_TYPE) {

		bool returns = false;
		Variant::get_method_return_type(basic_type, function, &returns);
		return returns ? 1 : 0;

	} else {
		int ret;
		MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
		if (mb) {
			ret = mb->has_return() ? 1 : 0;
		} else
			ret = 1; //it is assumed that script always returns something

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

		Vector<StringName> names = Variant::get_method_argument_names(basic_type, function);
		Vector<Variant::Type> types = Variant::get_method_argument_types(basic_type, function);
		return PropertyInfo(types[p_idx], names[p_idx]);

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

		return PropertyInfo(Variant::get_method_return_type(basic_type, function), "");
	} else {

		if (call_mode == CALL_MODE_INSTANCE) {
			if (p_idx == 0) {
				return PropertyInfo(Variant::OBJECT, "pass");
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

	static const char *cname[5] = {
		"CallSelf",
		"CallNode",
		"CallInstance",
		"CallBasic",
		"CallSingleton"
	};

	String caption = cname[call_mode];

	if (rpc_call_mode) {
		caption += " (RPC)";
	}

	return caption;
}

String VisualScriptFunctionCall::get_text() const {

	if (call_mode == CALL_MODE_SELF)
		return "  " + String(function) + "()";
	if (call_mode == CALL_MODE_SINGLETON)
		return String(singleton) + ":" + String(function) + "()";
	else if (call_mode == CALL_MODE_BASIC_TYPE)
		return Variant::get_type_name(basic_type) + "." + String(function) + "()";
	else if (call_mode == CALL_MODE_NODE_PATH)
		return " [" + String(base_path.simplified()) + "]." + String(function) + "()";
	else
		return "  " + base_type + "." + String(function) + "()";
}

void VisualScriptFunctionCall::set_basic_type(Variant::Type p_type) {

	if (basic_type == p_type)
		return;
	basic_type = p_type;

	_change_notify();
	ports_changed_notify();
}

Variant::Type VisualScriptFunctionCall::get_basic_type() const {

	return basic_type;
}

void VisualScriptFunctionCall::set_base_type(const StringName &p_type) {

	if (base_type == p_type)
		return;

	base_type = p_type;
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptFunctionCall::get_base_type() const {

	return base_type;
}

void VisualScriptFunctionCall::set_base_script(const String &p_path) {

	if (base_script == p_path)
		return;

	base_script = p_path;
	_change_notify();
	ports_changed_notify();
}

String VisualScriptFunctionCall::get_base_script() const {

	return base_script;
}

void VisualScriptFunctionCall::set_singleton(const StringName &p_path) {

	if (singleton == p_path)
		return;

	singleton = p_path;
	Object *obj = GlobalConfig::get_singleton()->get_singleton_object(singleton);
	if (obj) {
		base_type = obj->get_class();
	}

	_change_notify();
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

		Object *obj = GlobalConfig::get_singleton()->get_singleton_object(singleton);
		if (obj) {
			type = obj->get_class();
			script = obj->get_script();
		}

	} else if (call_mode == CALL_MODE_INSTANCE) {

		type = base_type;
		if (base_script != String()) {

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

	//print_line("BASE: "+String(type)+" FUNC: "+String(function));
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

		method_cache.return_val = mb->get_argument_info(-1);
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

	if (function == p_type)
		return;

	function = p_type;

	if (call_mode == CALL_MODE_BASIC_TYPE) {
		use_default_args = Variant::get_method_default_arguments(basic_type, function).size();
	} else {
		//update all caches

		_update_method_cache();
	}

	_change_notify();
	ports_changed_notify();
}
StringName VisualScriptFunctionCall::get_function() const {

	return function;
}

void VisualScriptFunctionCall::set_base_path(const NodePath &p_type) {

	if (base_path == p_type)
		return;

	base_path = p_type;
	_change_notify();
	ports_changed_notify();
}

NodePath VisualScriptFunctionCall::get_base_path() const {

	return base_path;
}

void VisualScriptFunctionCall::set_call_mode(CallMode p_mode) {

	if (call_mode == p_mode)
		return;

	call_mode = p_mode;
	_change_notify();
	ports_changed_notify();
}
VisualScriptFunctionCall::CallMode VisualScriptFunctionCall::get_call_mode() const {

	return call_mode;
}

void VisualScriptFunctionCall::set_use_default_args(int p_amount) {

	if (use_default_args == p_amount)
		return;

	use_default_args = p_amount;
	ports_changed_notify();
}

void VisualScriptFunctionCall::set_rpc_call_mode(VisualScriptFunctionCall::RPCCallMode p_mode) {

	if (rpc_call_mode == p_mode)
		return;
	rpc_call_mode = p_mode;
	ports_changed_notify();
	_change_notify();
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

	if (property.name == "function/base_type") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name == "function/base_script") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = 0;
		}
	}

	if (property.name == "function/basic_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE) {
			property.usage = 0;
		}
	}

	if (property.name == "function/singleton") {
		if (call_mode != CALL_MODE_SINGLETON) {
			property.usage = 0;
		} else {
			List<GlobalConfig::Singleton> names;
			GlobalConfig::get_singleton()->get_singletons(&names);
			property.hint = PROPERTY_HINT_ENUM;
			String sl;
			for (List<GlobalConfig::Singleton>::Element *E = names.front(); E; E = E->next()) {
				if (sl != String())
					sl += ",";
				sl += E->get().name;
			}
			property.hint_string = sl;
		}
	}

	if (property.name == "function/node_path") {
		if (call_mode != CALL_MODE_NODE_PATH) {
			property.usage = 0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string = bnode->get_path(); //convert to loong string
			} else {
			}
		}
	}

	if (property.name == "function/function") {

		if (call_mode == CALL_MODE_BASIC_TYPE) {

			property.hint = PROPERTY_HINT_METHOD_OF_VARIANT_TYPE;
			property.hint_string = Variant::get_type_name(basic_type);

		} else if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
			property.hint = PROPERTY_HINT_METHOD_OF_SCRIPT;
			property.hint_string = itos(get_visual_script()->get_instance_ID());
		} else if (call_mode == CALL_MODE_SINGLETON) {

			Object *obj = GlobalConfig::get_singleton()->get_singleton_object(singleton);
			if (obj) {
				property.hint = PROPERTY_HINT_METHOD_OF_INSTANCE;
				property.hint_string = itos(obj->get_instance_ID());
			} else {

				property.hint = PROPERTY_HINT_METHOD_OF_BASE_TYPE;
				property.hint_string = base_type; //should be cached
			}
		} else if (call_mode == CALL_MODE_INSTANCE) {
			property.hint = PROPERTY_HINT_METHOD_OF_BASE_TYPE;
			property.hint_string = base_type;

			if (base_script != String()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {

					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {

					Ref<Script> script = Ref<Resource>(ResourceCache::get(base_script));
					if (script.is_valid()) {

						property.hint = PROPERTY_HINT_METHOD_OF_SCRIPT;
						property.hint_string = itos(script->get_instance_ID());
					}
				}
			}

		} else if (call_mode == CALL_MODE_NODE_PATH) {
			Node *node = _get_base_node();
			if (node) {
				property.hint = PROPERTY_HINT_METHOD_OF_INSTANCE;
				property.hint_string = itos(node->get_instance_ID());
			} else {
				property.hint = PROPERTY_HINT_METHOD_OF_BASE_TYPE;
				property.hint_string = get_base_type();
			}
		}
	}

	if (property.name == "function/use_default_args") {

		property.hint = PROPERTY_HINT_RANGE;

		int mc = 0;

		if (call_mode == CALL_MODE_BASIC_TYPE) {

			mc = Variant::get_method_default_arguments(basic_type, function).size();
		} else {
			MethodBind *mb = ClassDB::get_method(_get_base_type(), function);
			if (mb) {

				mc = mb->get_default_argument_count();
			}
		}

		if (mc == 0) {
			property.usage = 0; //do not show
		} else {

			property.hint_string = "0," + itos(mc) + ",1";
		}
	}

	if (property.name == "rpc/call_mode") {
		if (call_mode == CALL_MODE_BASIC_TYPE) {
			property.usage = 0;
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
		if (i > 0)
			bt += ",";

		bt += Variant::get_type_name(Variant::Type(i));
	}

	List<String> script_extensions;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (List<String>::Element *E = script_extensions.front(); E; E = E->next()) {
		if (script_ext_hint != String())
			script_ext_hint += ",";
		script_ext_hint += "*." + E->get();
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "function/call_mode", PROPERTY_HINT_ENUM, "Self,Node Path,Instance,Basic Type,Singleton"), "set_call_mode", "get_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function/base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function/base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function/singleton"), "set_singleton", "get_singleton");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "function/basic_type", PROPERTY_HINT_ENUM, bt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "function/node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "function/argument_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_argument_cache", "_get_argument_cache");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "function/function"), "set_function", "get_function"); //when set, if loaded properly, will override argument count.
	ADD_PROPERTY(PropertyInfo(Variant::INT, "function/use_default_args"), "set_use_default_args", "get_use_default_args");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "function/validate"), "set_validate", "get_validate");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rpc/call_mode", PROPERTY_HINT_ENUM, "Disabled,Reliable,Unreliable,ReliableToID,UnreliableToID"), "set_rpc_call_mode", "get_rpc_call_mode"); //when set, if loaded properly, will override argument count.

	BIND_CONSTANT(CALL_MODE_SELF);
	BIND_CONSTANT(CALL_MODE_NODE_PATH);
	BIND_CONSTANT(CALL_MODE_INSTANCE);
	BIND_CONSTANT(CALL_MODE_BASIC_TYPE);
}

class VisualScriptNodeInstanceFunctionCall : public VisualScriptNodeInstance {
public:
	VisualScriptFunctionCall::CallMode call_mode;
	NodePath node_path;
	int input_args;
	bool validate;
	bool returns;
	VisualScriptFunctionCall::RPCCallMode rpc_mode;
	StringName function;
	StringName singleton;

	VisualScriptFunctionCall *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	_FORCE_INLINE_ bool call_rpc(Object *p_base, const Variant **p_args, int p_argcount) {

		if (!p_base)
			return false;

		Node *node = p_base->cast_to<Node>();
		if (!node)
			return false;

		int to_id = 0;
		bool reliable = true;

		if (rpc_mode >= VisualScriptFunctionCall::RPC_RELIABLE_TO_ID) {
			to_id = *p_args[0];
			p_args += 1;
			p_argcount -= 1;
			if (rpc_mode == VisualScriptFunctionCall::RPC_UNRELIABLE_TO_ID) {
				reliable = false;
			}
		} else if (rpc_mode == VisualScriptFunctionCall::RPC_UNRELIABLE) {
			reliable = false;
		}

		node->rpcp(to_id, !reliable, function, p_args, p_argcount);

		return true;
	}

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

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

				Node *node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Base object is not a Node!";
					return 0;
				}

				Node *another = node->get_node(node_path);
				if (!node) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
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
						*p_outputs[1] = v.call(function, p_inputs + 1, input_args, r_error);
					} else {
						*p_outputs[0] = v.call(function, p_inputs + 1, input_args, r_error);
					}
				} else {
					v.call(function, p_inputs + 1, input_args, r_error);
				}

				if (call_mode == VisualScriptFunctionCall::CALL_MODE_INSTANCE) {
					*p_outputs[0] = *p_inputs[0];
				}

			} break;
			case VisualScriptFunctionCall::CALL_MODE_SINGLETON: {

				Object *object = GlobalConfig::get_singleton()->get_singleton_object(singleton);
				if (!object) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
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
			r_error.error = Variant::CallError::CALL_OK;
			r_error_str = String();
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptFunctionCall::instance(VisualScriptInstance *p_instance) {

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
	node.instance();
	node->set_call_mode(cmode);
	return node;
}

//////////////////////////////////////////
////////////////SET//////////////////////
//////////////////////////////////////////

static const char *event_type_names[InputEvent::TYPE_MAX] = {
	"None",
	"Key",
	"MouseMotion",
	"MouseButton",
	"JoypadMotion",
	"JoypadButton",
	"ScreenTouch",
	"ScreenDrag",
	"Action"
};

int VisualScriptPropertySet::get_output_sequence_port_count() const {

	return call_mode != CALL_MODE_BASIC_TYPE ? 1 : 0;
}

bool VisualScriptPropertySet::has_input_sequence_port() const {

	return call_mode != CALL_MODE_BASIC_TYPE ? true : false;
}

Node *VisualScriptPropertySet::_get_base_node() const {

#ifdef TOOLS_ENABLED
	Ref<Script> script = get_visual_script();
	if (!script.is_valid())
		return NULL;

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}

StringName VisualScriptPropertySet::_get_base_type() const {

	if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else if (call_mode == CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path)
			return path->get_class();
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

PropertyInfo VisualScriptPropertySet::get_input_value_port_info(int p_idx) const {

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

	PropertyInfo pinfo = type_cache;
	pinfo.name = "value";
	return pinfo;
}

PropertyInfo VisualScriptPropertySet::get_output_value_port_info(int p_idx) const {
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		return PropertyInfo(basic_type, "out");
	} else if (call_mode == CALL_MODE_INSTANCE) {
		return PropertyInfo(Variant::OBJECT, "pass");
	} else {
		return PropertyInfo();
	}
}

String VisualScriptPropertySet::get_caption() const {

	static const char *cname[4] = {
		"SelfSet",
		"NodeSet",
		"InstanceSet",
		"BasicSet"
	};

	return cname[call_mode];
}

String VisualScriptPropertySet::get_text() const {

	String prop;

	if (call_mode == CALL_MODE_BASIC_TYPE)
		prop = Variant::get_type_name(basic_type) + "." + property;
	else if (call_mode == CALL_MODE_NODE_PATH)
		prop = String(base_path) + ":" + property;
	else if (call_mode == CALL_MODE_SELF)
		prop = property;
	else if (call_mode == CALL_MODE_INSTANCE)
		prop = String(base_type) + ":" + property;

	return prop;
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

	if (basic_type == p_type)
		return;
	basic_type = p_type;

	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

Variant::Type VisualScriptPropertySet::get_basic_type() const {

	return basic_type;
}

void VisualScriptPropertySet::set_event_type(InputEvent::Type p_type) {

	if (event_type == p_type)
		return;
	event_type = p_type;
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		_update_cache();
	}
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

InputEvent::Type VisualScriptPropertySet::get_event_type() const {

	return event_type;
}

void VisualScriptPropertySet::set_base_type(const StringName &p_type) {

	if (base_type == p_type)
		return;

	base_type = p_type;
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptPropertySet::get_base_type() const {

	return base_type;
}

void VisualScriptPropertySet::set_base_script(const String &p_path) {

	if (base_script == p_path)
		return;

	base_script = p_path;
	_change_notify();
	ports_changed_notify();
}

String VisualScriptPropertySet::get_base_script() const {

	return base_script;
}

void VisualScriptPropertySet::_update_cache() {

	if (!OS::get_singleton()->get_main_loop())
		return;
	if (!OS::get_singleton()->get_main_loop()->cast_to<SceneTree>())
		return;

	if (!OS::get_singleton()->get_main_loop()->cast_to<SceneTree>()->is_editor_hint()) //only update cache if editor exists, it's pointless otherwise
		return;

	if (call_mode == CALL_MODE_BASIC_TYPE) {

		//not super efficient..

		Variant v;
		if (basic_type == Variant::INPUT_EVENT) {
			InputEvent ev;
			ev.type = event_type;
			v = ev;
		} else {
			Variant::CallError ce;
			v = Variant::construct(basic_type, NULL, 0, ce);
		}

		List<PropertyInfo> pinfo;
		v.get_property_list(&pinfo);

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

			if (E->get().name == property) {

				type_cache = E->get();
			}
		}

	} else {

		StringName type;
		Ref<Script> script;
		Node *node = NULL;

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
			if (base_script != String()) {

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

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

			if (E->get().name == property) {
				type_cache = E->get();
				return;
			}
		}
	}
}

void VisualScriptPropertySet::set_property(const StringName &p_type) {

	if (property == p_type)
		return;

	property = p_type;
	_update_cache();
	_change_notify();
	ports_changed_notify();
}
StringName VisualScriptPropertySet::get_property() const {

	return property;
}

void VisualScriptPropertySet::set_base_path(const NodePath &p_type) {

	if (base_path == p_type)
		return;

	base_path = p_type;
	_update_base_type();
	_change_notify();
	ports_changed_notify();
}

NodePath VisualScriptPropertySet::get_base_path() const {

	return base_path;
}

void VisualScriptPropertySet::set_call_mode(CallMode p_mode) {

	if (call_mode == p_mode)
		return;

	call_mode = p_mode;
	_update_base_type();
	_change_notify();
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

void VisualScriptPropertySet::_validate_property(PropertyInfo &property) const {

	if (property.name == "property/base_type") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name == "property/base_script") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = 0;
		}
	}

	if (property.name == "property/basic_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE) {
			property.usage = 0;
		}
	}

	if (property.name == "property/event_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE || basic_type != Variant::INPUT_EVENT) {
			property.usage = 0;
		}
	}

	if (property.name == "property/node_path") {
		if (call_mode != CALL_MODE_NODE_PATH) {
			property.usage = 0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string = bnode->get_path(); //convert to loong string
			} else {
			}
		}
	}

	if (property.name == "property/property") {

		if (call_mode == CALL_MODE_BASIC_TYPE) {

			property.hint = PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE;
			property.hint_string = Variant::get_type_name(basic_type);

		} else if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
			property.hint_string = itos(get_visual_script()->get_instance_ID());
		} else if (call_mode == CALL_MODE_INSTANCE) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
			property.hint_string = base_type;

			if (base_script != String()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {

					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {

					Ref<Script> script = Ref<Resource>(ResourceCache::get(base_script));
					if (script.is_valid()) {

						property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
						property.hint_string = itos(script->get_instance_ID());
					}
				}
			}

		} else if (call_mode == CALL_MODE_NODE_PATH) {
			Node *node = _get_base_node();
			if (node) {
				property.hint = PROPERTY_HINT_PROPERTY_OF_INSTANCE;
				property.hint_string = itos(node->get_instance_ID());
			} else {
				property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
				property.hint_string = get_base_type();
			}
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

	ClassDB::bind_method(D_METHOD("set_event_type", "event_type"), &VisualScriptPropertySet::set_event_type);
	ClassDB::bind_method(D_METHOD("get_event_type"), &VisualScriptPropertySet::get_event_type);

	ClassDB::bind_method(D_METHOD("set_property", "property"), &VisualScriptPropertySet::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &VisualScriptPropertySet::get_property);

	ClassDB::bind_method(D_METHOD("set_call_mode", "mode"), &VisualScriptPropertySet::set_call_mode);
	ClassDB::bind_method(D_METHOD("get_call_mode"), &VisualScriptPropertySet::get_call_mode);

	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &VisualScriptPropertySet::set_base_path);
	ClassDB::bind_method(D_METHOD("get_base_path"), &VisualScriptPropertySet::get_base_path);

	String bt;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0)
			bt += ",";

		bt += Variant::get_type_name(Variant::Type(i));
	}

	String et;
	for (int i = 0; i < InputEvent::TYPE_MAX; i++) {
		if (i > 0)
			et += ",";

		et += event_type_names[i];
	}

	List<String> script_extensions;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (List<String>::Element *E = script_extensions.front(); E; E = E->next()) {
		if (script_ext_hint != String())
			script_ext_hint += ",";
		script_ext_hint += "*." + E->get();
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/set_mode", PROPERTY_HINT_ENUM, "Self,Node Path,Instance,Basic Type"), "set_call_mode", "get_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/type_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_type_cache", "_get_type_cache");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/basic_type", PROPERTY_HINT_ENUM, bt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/event_type", PROPERTY_HINT_ENUM, et), "set_event_type", "get_event_type");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "property/node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/property"), "set_property", "get_property");

	BIND_CONSTANT(CALL_MODE_SELF);
	BIND_CONSTANT(CALL_MODE_NODE_PATH);
	BIND_CONSTANT(CALL_MODE_INSTANCE);
}

class VisualScriptNodeInstancePropertySet : public VisualScriptNodeInstance {
public:
	VisualScriptPropertySet::CallMode call_mode;
	NodePath node_path;
	StringName property;

	VisualScriptPropertySet *node;
	VisualScriptInstance *instance;

	//virtual int get_working_memory_size() const { return 0; }
	//virtual bool is_output_port_unsequenced(int p_idx) const { return false; }
	//virtual bool get_output_port_unsequenced(int p_idx,Variant* r_value,Variant* p_working_mem,String &r_error) const { return true; }

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		switch (call_mode) {

			case VisualScriptPropertySet::CALL_MODE_SELF: {

				Object *object = instance->get_owner_ptr();

				bool valid;

				object->set(property, *p_inputs[0], &valid);

				if (!valid) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid set value '" + String(*p_inputs[0]) + "' on property '" + String(property) + "' of type " + object->get_class();
				}
			} break;
			case VisualScriptPropertySet::CALL_MODE_NODE_PATH: {

				Node *node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Base object is not a Node!";
					return 0;
				}

				Node *another = node->get_node(node_path);
				if (!node) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Path does not lead Node!";
					return 0;
				}

				bool valid;

				another->set(property, *p_inputs[0], &valid);

				if (!valid) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid set value '" + String(*p_inputs[0]) + "' on property '" + String(property) + "' of type " + another->get_class();
				}

			} break;
			case VisualScriptPropertySet::CALL_MODE_INSTANCE:
			case VisualScriptPropertySet::CALL_MODE_BASIC_TYPE: {

				Variant v = *p_inputs[0];

				bool valid;

				v.set(property, *p_inputs[1], &valid);

				if (!valid) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = "Invalid set value '" + String(*p_inputs[1]) + "' (" + Variant::get_type_name(p_inputs[1]->get_type()) + ") on property '" + String(property) + "' of type " + Variant::get_type_name(v.get_type());
				}

				*p_outputs[0] = v;

			} break;
		}
		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptPropertySet::instance(VisualScriptInstance *p_instance) {

	VisualScriptNodeInstancePropertySet *instance = memnew(VisualScriptNodeInstancePropertySet);
	instance->node = this;
	instance->instance = p_instance;
	instance->property = property;
	instance->call_mode = call_mode;
	instance->node_path = base_path;
	return instance;
}

VisualScriptPropertySet::TypeGuess VisualScriptPropertySet::guess_output_type(TypeGuess *p_inputs, int p_output) const {

	if (p_output == 0 && call_mode == CALL_MODE_INSTANCE) {
		return p_inputs[0];
	}

	return VisualScriptNode::guess_output_type(p_inputs, p_output);
}
VisualScriptPropertySet::VisualScriptPropertySet() {

	call_mode = CALL_MODE_SELF;
	base_type = "Object";
	basic_type = Variant::NIL;
	event_type = InputEvent::NONE;
}

template <VisualScriptPropertySet::CallMode cmode>
static Ref<VisualScriptNode> create_property_set_node(const String &p_name) {

	Ref<VisualScriptPropertySet> node;
	node.instance();
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
	if (!script.is_valid())
		return NULL;

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	if (!main_loop)
		return NULL;

	SceneTree *scene_tree = main_loop->cast_to<SceneTree>();

	if (!scene_tree)
		return NULL;

	Node *edited_scene = scene_tree->get_edited_scene_root();

	if (!edited_scene)
		return NULL;

	Node *script_node = _find_script_node(edited_scene, edited_scene, script);

	if (!script_node)
		return NULL;

	if (!script_node->has_node(base_path))
		return NULL;

	Node *path_to = script_node->get_node(base_path);

	return path_to;
#else

	return NULL;
#endif
}

StringName VisualScriptPropertyGet::_get_base_type() const {

	if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid())
		return get_visual_script()->get_instance_base_type();
	else if (call_mode == CALL_MODE_NODE_PATH && get_visual_script().is_valid()) {
		Node *path = _get_base_node();
		if (path)
			return path->get_class();
	}

	return base_type;
}

int VisualScriptPropertyGet::get_input_value_port_count() const {

	return (call_mode == CALL_MODE_BASIC_TYPE || call_mode == CALL_MODE_INSTANCE) ? 1 : 0;
}
int VisualScriptPropertyGet::get_output_value_port_count() const {

	return 1;
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
		} else {
			p_idx--;
		}
	}
	return PropertyInfo();
}

PropertyInfo VisualScriptPropertyGet::get_output_value_port_info(int p_idx) const {

	return PropertyInfo(type_cache, "value");
}

String VisualScriptPropertyGet::get_caption() const {

	static const char *cname[4] = {
		"SelfGet",
		"NodeGet",
		"InstanceGet",
		"BasicGet"
	};

	return cname[call_mode];
}

String VisualScriptPropertyGet::get_text() const {

	String prop;

	if (call_mode == CALL_MODE_BASIC_TYPE)
		prop = Variant::get_type_name(basic_type) + "." + property;
	else if (call_mode == CALL_MODE_NODE_PATH)
		prop = String(base_path) + ":" + property;
	else if (call_mode == CALL_MODE_SELF)
		prop = property;
	else if (call_mode == CALL_MODE_INSTANCE)
		prop = String(base_type) + ":" + property;

	return prop;
}

void VisualScriptPropertyGet::set_base_type(const StringName &p_type) {

	if (base_type == p_type)
		return;

	base_type = p_type;
	_change_notify();
	ports_changed_notify();
}

StringName VisualScriptPropertyGet::get_base_type() const {

	return base_type;
}

void VisualScriptPropertyGet::set_base_script(const String &p_path) {

	if (base_script == p_path)
		return;

	base_script = p_path;
	_change_notify();
	ports_changed_notify();
}

String VisualScriptPropertyGet::get_base_script() const {

	return base_script;
}

void VisualScriptPropertyGet::_update_cache() {

	if (call_mode == CALL_MODE_BASIC_TYPE) {

		//not super efficient..

		Variant v;
		if (basic_type == Variant::INPUT_EVENT) {
			InputEvent ev;
			ev.type = event_type;
			v = ev;
		} else {
			Variant::CallError ce;
			v = Variant::construct(basic_type, NULL, 0, ce);
		}

		List<PropertyInfo> pinfo;
		v.get_property_list(&pinfo);

		for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {

			if (E->get().name == property) {

				type_cache = E->get().type;
				return;
			}
		}

	} else {

		StringName type;
		Ref<Script> script;
		Node *node = NULL;

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
			if (base_script != String()) {

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

	if (property == p_type)
		return;

	property = p_type;

	_update_cache();
	_change_notify();
	ports_changed_notify();
}
StringName VisualScriptPropertyGet::get_property() const {

	return property;
}

void VisualScriptPropertyGet::set_base_path(const NodePath &p_type) {

	if (base_path == p_type)
		return;

	base_path = p_type;
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

NodePath VisualScriptPropertyGet::get_base_path() const {

	return base_path;
}

void VisualScriptPropertyGet::set_call_mode(CallMode p_mode) {

	if (call_mode == p_mode)
		return;

	call_mode = p_mode;
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}
VisualScriptPropertyGet::CallMode VisualScriptPropertyGet::get_call_mode() const {

	return call_mode;
}

void VisualScriptPropertyGet::set_basic_type(Variant::Type p_type) {

	if (basic_type == p_type)
		return;
	basic_type = p_type;

	_change_notify();
	ports_changed_notify();
}

Variant::Type VisualScriptPropertyGet::get_basic_type() const {

	return basic_type;
}

void VisualScriptPropertyGet::set_event_type(InputEvent::Type p_type) {

	if (event_type == p_type)
		return;
	event_type = p_type;
	if (call_mode == CALL_MODE_BASIC_TYPE) {
		_update_cache();
	}
	_change_notify();
	_update_base_type();
	ports_changed_notify();
}

InputEvent::Type VisualScriptPropertyGet::get_event_type() const {

	return event_type;
}

void VisualScriptPropertyGet::_set_type_cache(Variant::Type p_type) {
	type_cache = p_type;
}

Variant::Type VisualScriptPropertyGet::_get_type_cache() const {

	return type_cache;
}

void VisualScriptPropertyGet::_validate_property(PropertyInfo &property) const {

	if (property.name == "property/base_type") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}

	if (property.name == "property/base_script") {
		if (call_mode != CALL_MODE_INSTANCE) {
			property.usage = 0;
		}
	}

	if (property.name == "property/basic_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE) {
			property.usage = 0;
		}
	}
	if (property.name == "property/event_type") {
		if (call_mode != CALL_MODE_BASIC_TYPE || basic_type != Variant::INPUT_EVENT) {
			property.usage = 0;
		}
	}

	if (property.name == "property/node_path") {
		if (call_mode != CALL_MODE_NODE_PATH) {
			property.usage = 0;
		} else {

			Node *bnode = _get_base_node();
			if (bnode) {
				property.hint_string = bnode->get_path(); //convert to loong string
			} else {
			}
		}
	}

	if (property.name == "property/property") {

		if (call_mode == CALL_MODE_BASIC_TYPE) {

			property.hint = PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE;
			property.hint_string = Variant::get_type_name(basic_type);

		} else if (call_mode == CALL_MODE_SELF && get_visual_script().is_valid()) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
			property.hint_string = itos(get_visual_script()->get_instance_ID());
		} else if (call_mode == CALL_MODE_INSTANCE) {
			property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
			property.hint_string = base_type;

			if (base_script != String()) {
				if (!ResourceCache::has(base_script) && ScriptServer::edit_request_func) {

					ScriptServer::edit_request_func(base_script); //make sure it's loaded
				}

				if (ResourceCache::has(base_script)) {

					Ref<Script> script = Ref<Resource>(ResourceCache::get(base_script));
					if (script.is_valid()) {

						property.hint = PROPERTY_HINT_PROPERTY_OF_SCRIPT;
						property.hint_string = itos(script->get_instance_ID());
					}
				}
			}
		} else if (call_mode == CALL_MODE_NODE_PATH) {
			Node *node = _get_base_node();
			if (node) {
				property.hint = PROPERTY_HINT_PROPERTY_OF_INSTANCE;
				property.hint_string = itos(node->get_instance_ID());
			} else {
				property.hint = PROPERTY_HINT_PROPERTY_OF_BASE_TYPE;
				property.hint_string = get_base_type();
			}
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

	ClassDB::bind_method(D_METHOD("set_event_type", "event_type"), &VisualScriptPropertyGet::set_event_type);
	ClassDB::bind_method(D_METHOD("get_event_type"), &VisualScriptPropertyGet::get_event_type);

	ClassDB::bind_method(D_METHOD("set_property", "property"), &VisualScriptPropertyGet::set_property);
	ClassDB::bind_method(D_METHOD("get_property"), &VisualScriptPropertyGet::get_property);

	ClassDB::bind_method(D_METHOD("set_call_mode", "mode"), &VisualScriptPropertyGet::set_call_mode);
	ClassDB::bind_method(D_METHOD("get_call_mode"), &VisualScriptPropertyGet::get_call_mode);

	ClassDB::bind_method(D_METHOD("set_base_path", "base_path"), &VisualScriptPropertyGet::set_base_path);
	ClassDB::bind_method(D_METHOD("get_base_path"), &VisualScriptPropertyGet::get_base_path);

	String bt;
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		if (i > 0)
			bt += ",";

		bt += Variant::get_type_name(Variant::Type(i));
	}

	String et;
	for (int i = 0; i < InputEvent::TYPE_MAX; i++) {
		if (i > 0)
			et += ",";

		et += event_type_names[i];
	}

	List<String> script_extensions;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->get_recognized_extensions(&script_extensions);
	}

	String script_ext_hint;
	for (List<String>::Element *E = script_extensions.front(); E; E = E->next()) {
		if (script_ext_hint != String())
			script_ext_hint += ",";
		script_ext_hint += "." + E->get();
	}

	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/set_mode", PROPERTY_HINT_ENUM, "Self,Node Path,Instance,Basic Type"), "set_call_mode", "get_call_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/base_type", PROPERTY_HINT_TYPE_STRING, "Object"), "set_base_type", "get_base_type");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/base_script", PROPERTY_HINT_FILE, script_ext_hint), "set_base_script", "get_base_script");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/type_cache", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_type_cache", "_get_type_cache");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/basic_type", PROPERTY_HINT_ENUM, bt), "set_basic_type", "get_basic_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "property/event_type", PROPERTY_HINT_ENUM, et), "set_event_type", "get_event_type");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "property/node_path", PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE), "set_base_path", "get_base_path");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "property/property"), "set_property", "get_property");

	BIND_CONSTANT(CALL_MODE_SELF);
	BIND_CONSTANT(CALL_MODE_NODE_PATH);
	BIND_CONSTANT(CALL_MODE_INSTANCE);
}

class VisualScriptNodeInstancePropertyGet : public VisualScriptNodeInstance {
public:
	VisualScriptPropertyGet::CallMode call_mode;
	NodePath node_path;
	StringName property;

	VisualScriptPropertyGet *node;
	VisualScriptInstance *instance;

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		switch (call_mode) {

			case VisualScriptPropertyGet::CALL_MODE_SELF: {

				Object *object = instance->get_owner_ptr();

				bool valid;

				*p_outputs[0] = object->get(property, &valid);

				if (!valid) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Invalid index property name.");
					return 0;
				}
			} break;
			case VisualScriptPropertyGet::CALL_MODE_NODE_PATH: {

				Node *node = instance->get_owner_ptr()->cast_to<Node>();
				if (!node) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Base object is not a Node!");
					return 0;
				}

				Node *another = node->get_node(node_path);
				if (!node) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Path does not lead Node!");
					return 0;
				}

				bool valid;

				*p_outputs[0] = another->get(property, &valid);

				if (!valid) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = vformat(RTR("Invalid index property name '%s' in node %s."), String(property), another->get_name());
					return 0;
				}

			} break;
			default: {

				bool valid;
				Variant v = *p_inputs[0];

				*p_outputs[0] = v.get(property, &valid);

				if (!valid) {
					r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
					r_error_str = RTR("Invalid index property name.");
				}
			};
		}

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptPropertyGet::instance(VisualScriptInstance *p_instance) {

	VisualScriptNodeInstancePropertyGet *instance = memnew(VisualScriptNodeInstancePropertyGet);
	instance->node = this;
	instance->instance = p_instance;
	instance->property = property;
	instance->call_mode = call_mode;
	instance->node_path = base_path;

	return instance;
}

VisualScriptPropertyGet::VisualScriptPropertyGet() {

	call_mode = CALL_MODE_SELF;
	base_type = "Object";
	basic_type = Variant::NIL;
	event_type = InputEvent::NONE;
	type_cache = Variant::NIL;
}

template <VisualScriptPropertyGet::CallMode cmode>
static Ref<VisualScriptNode> create_property_get_node(const String &p_name) {

	Ref<VisualScriptPropertyGet> node;
	node.instance();
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

		if (!vs->has_custom_signal(name))
			return 0;

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

		if (!vs->has_custom_signal(name))
			return PropertyInfo();

		return PropertyInfo(vs->custom_signal_get_argument_type(name, p_idx), vs->custom_signal_get_argument_name(name, p_idx));
	}

	return PropertyInfo();
}

PropertyInfo VisualScriptEmitSignal::get_output_value_port_info(int p_idx) const {

	return PropertyInfo();
}

String VisualScriptEmitSignal::get_caption() const {

	return "EmitSignal";
}

String VisualScriptEmitSignal::get_text() const {

	return "emit " + String(name);
}

void VisualScriptEmitSignal::set_signal(const StringName &p_type) {

	if (name == p_type)
		return;

	name = p_type;

	_change_notify();
	ports_changed_notify();
}
StringName VisualScriptEmitSignal::get_signal() const {

	return name;
}

void VisualScriptEmitSignal::_validate_property(PropertyInfo &property) const {

	if (property.name == "signal/signal") {
		property.hint = PROPERTY_HINT_ENUM;

		List<StringName> sigs;

		Ref<VisualScript> vs = get_visual_script();
		if (vs.is_valid()) {

			vs->get_custom_signal_list(&sigs);
		}

		String ml;
		for (List<StringName>::Element *E = sigs.front(); E; E = E->next()) {

			if (ml != String())
				ml += ",";
			ml += E->get();
		}

		property.hint_string = ml;
	}
}

void VisualScriptEmitSignal::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_signal", "name"), &VisualScriptEmitSignal::set_signal);
	ClassDB::bind_method(D_METHOD("get_signal"), &VisualScriptEmitSignal::get_signal);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "signal/signal"), "set_signal", "get_signal");
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

	virtual int step(const Variant **p_inputs, Variant **p_outputs, StartMode p_start_mode, Variant *p_working_mem, Variant::CallError &r_error, String &r_error_str) {

		Object *obj = instance->get_owner_ptr();

		obj->emit_signal(name, p_inputs, argcount);

		return 0;
	}
};

VisualScriptNodeInstance *VisualScriptEmitSignal::instance(VisualScriptInstance *p_instance) {

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
	node.instance();

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
		Variant::CallError ce;
		Variant vt = Variant::construct(t, NULL, 0, ce);
		List<MethodInfo> ml;
		vt.get_method_list(&ml);

		for (List<MethodInfo>::Element *E = ml.front(); E; E = E->next()) {
			VisualScriptLanguage::singleton->add_register_func("functions/by_type/" + type_name + "/" + E->get().name, create_basic_type_call_node);
		}
	}
}
