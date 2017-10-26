/*************************************************************************/
/*  nativescript.cpp                                                     */
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
#include "nativescript.h"

#include "gdnative/gdnative.h"

#include "core/global_constants.h"
#include "core/project_settings.h"
#include "io/file_access_encrypted.h"
#include "os/file_access.h"
#include "os/os.h"

#include "scene/main/scene_tree.h"
#include "scene/resources/scene_format_text.h"

#ifndef NO_THREADS
#include "os/thread.h"
#endif

#if defined(TOOLS_ENABLED) && defined(DEBUG_METHODS_ENABLED)
#include "api_generator.h"
#endif

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

////// Script stuff

void NativeScript::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_class_name", "class_name"), &NativeScript::set_class_name);
	ClassDB::bind_method(D_METHOD("get_class_name"), &NativeScript::get_class_name);

	ClassDB::bind_method(D_METHOD("set_library", "library"), &NativeScript::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &NativeScript::get_library);

	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "class_name"), "set_class_name", "get_class_name");
	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "library", PROPERTY_HINT_RESOURCE_TYPE, "GDNativeLibrary"), "set_library", "get_library");

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "new", &NativeScript::_new, MethodInfo(Variant::OBJECT, "new"));
}

#define NSL NativeScriptLanguage::get_singleton()

#ifdef TOOLS_ENABLED

void NativeScript::_update_placeholder(PlaceHolderScriptInstance *p_placeholder) {
	NativeScriptDesc *script_data = get_script_desc();

	ERR_FAIL_COND(!script_data);

	List<PropertyInfo> info;
	get_script_property_list(&info);
	Map<StringName, Variant> values;
	for (List<PropertyInfo>::Element *E = info.front(); E; E = E->next()) {
		Variant value;
		get_property_default_value(E->get().name, value);
		values[E->get().name] = value;
	}

	p_placeholder->update(info, values);
}

void NativeScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	placeholders.erase(p_placeholder);
}

#endif

void NativeScript::set_class_name(String p_class_name) {
	class_name = p_class_name;
}

String NativeScript::get_class_name() const {
	return class_name;
}

void NativeScript::set_library(Ref<GDNativeLibrary> p_library) {
	if (!library.is_null()) {
		WARN_PRINT("library on NativeScript already set. Do nothing.");
		return;
	}
	library = p_library;
	lib_path = library->get_active_library_path();

#ifndef NO_THREADS
	if (Thread::get_caller_id() != Thread::get_main_id()) {
		NSL->defer_init_library(p_library, this);
	} else
#endif
	{
		NSL->init_library(p_library);
		NSL->register_script(this);
	}
}

Ref<GDNativeLibrary> NativeScript::get_library() const {
	return library;
}

bool NativeScript::can_instance() const {

	NativeScriptDesc *script_data = get_script_desc();

#ifdef TOOLS_ENABLED

	return script_data || (!is_tool() && !ScriptServer::is_scripting_enabled());
#else
	return script_data;
#endif
}

Ref<Script> NativeScript::get_base_script() const {
	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data)
		return Ref<Script>();

	Ref<NativeScript> ns = Ref<NativeScript>(NSL->create_script());
	ns->set_class_name(script_data->base);
	ns->set_library(get_library());
	return ns;
}

StringName NativeScript::get_instance_base_type() const {
	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data)
		return "";

	return script_data->base_native_type;
}

ScriptInstance *NativeScript::instance_create(Object *p_this) {

	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data) {
		return NULL;
	}

#ifdef TOOLS_ENABLED
	if (!ScriptServer::is_scripting_enabled() && !is_tool()) {
		// placeholder for nodes. For tools we want the rool thing.

		PlaceHolderScriptInstance *sins = memnew(PlaceHolderScriptInstance(NSL, Ref<Script>(this), p_this));
		placeholders.insert(sins);

		if (script_data->create_func.create_func) {
			script_data->create_func.create_func(
					(godot_object *)p_this,
					script_data->create_func.method_data);
		}

		_update_placeholder(sins);

		return sins;
	}
#endif

	NativeScriptInstance *nsi = memnew(NativeScriptInstance);

	nsi->owner = p_this;
	nsi->script = Ref<NativeScript>(this);

#ifndef TOOLS_ENABLED
	if (!ScriptServer::is_scripting_enabled()) {
		nsi->userdata = NULL;
	} else {
		nsi->userdata = script_data->create_func.create_func((godot_object *)p_this, script_data->create_func.method_data);
	}
#else
	nsi->userdata = script_data->create_func.create_func((godot_object *)p_this, script_data->create_func.method_data);
#endif

#ifndef NO_THREADS
	owners_lock->lock();
#endif

	instance_owners.insert(p_this);

#ifndef NO_THREADS
	owners_lock->unlock();
#endif

	return nsi;
}

bool NativeScript::instance_has(const Object *p_this) const {
	return instance_owners.has((Object *)p_this);
}

bool NativeScript::has_source_code() const {
	return false;
}

String NativeScript::get_source_code() const {
	return "";
}

void NativeScript::set_source_code(const String &p_code) {
}

Error NativeScript::reload(bool p_keep_state) {
	return FAILED;
}

bool NativeScript::has_method(const StringName &p_method) const {
	NativeScriptDesc *script_data = get_script_desc();

	while (script_data) {
		if (script_data->methods.has(p_method))
			return true;

		script_data = script_data->base_data;
	}
	return false;
}

MethodInfo NativeScript::get_method_info(const StringName &p_method) const {
	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data)
		return MethodInfo();

	while (script_data) {
		Map<StringName, NativeScriptDesc::Method>::Element *M = script_data->methods.find(p_method);

		if (M)
			return M->get().info;

		script_data = script_data->base_data;
	}
	return MethodInfo();
}

bool NativeScript::is_tool() const {
	NativeScriptDesc *script_data = get_script_desc();

	if (script_data)
		return script_data->is_tool;

	return false;
}

ScriptLanguage *NativeScript::get_language() const {
	return NativeScriptLanguage::get_singleton();
}

bool NativeScript::has_script_signal(const StringName &p_signal) const {
	NativeScriptDesc *script_data = get_script_desc();

	while (script_data) {
		if (script_data->signals_.has(p_signal))
			return true;
		script_data = script_data->base_data;
	}
	return false;
}

void NativeScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data)
		return;

	Set<MethodInfo> signals_;

	while (script_data) {

		for (Map<StringName, NativeScriptDesc::Signal>::Element *S = script_data->signals_.front(); S; S = S->next()) {
			signals_.insert(S->get().signal);
		}

		script_data = script_data->base_data;
	}

	for (Set<MethodInfo>::Element *E = signals_.front(); E; E = E->next()) {
		r_signals->push_back(E->get());
	}
}

bool NativeScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	NativeScriptDesc *script_data = get_script_desc();

	OrderedHashMap<StringName, NativeScriptDesc::Property>::Element P;
	while (!P && script_data) {
		P = script_data->properties.find(p_property);
		script_data = script_data->base_data;
	}
	if (!P)
		return false;

	r_value = P.get().default_value;
	return true;
}

void NativeScript::update_exports() {
}

void NativeScript::get_script_method_list(List<MethodInfo> *p_list) const {
	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data)
		return;

	Set<MethodInfo> methods;

	while (script_data) {

		for (Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.front(); E; E = E->next()) {
			methods.insert(E->get().info);
		}

		script_data = script_data->base_data;
	}

	for (Set<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
		p_list->push_back(E->get());
	}
}

void NativeScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	NativeScriptDesc *script_data = get_script_desc();

	Set<StringName> existing_properties;
	while (script_data) {
		List<PropertyInfo>::Element *insert_position = p_list->front();
		bool insert_before = true;

		for (OrderedHashMap<StringName, NativeScriptDesc::Property>::Element E = script_data->properties.front(); E; E = E.next()) {
			if (!existing_properties.has(E.key())) {
				insert_position = insert_before ? p_list->insert_before(insert_position, E.get().info) : p_list->insert_after(insert_position, E.get().info);
				insert_before = false;
				existing_properties.insert(E.key());
			}
		}
		script_data = script_data->base_data;
	}
}

Variant NativeScript::_new(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (lib_path.empty() || class_name.empty() || library.is_null()) {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}

	NativeScriptDesc *script_data = get_script_desc();

	if (!script_data) {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}

	r_error.error = Variant::CallError::CALL_OK;

	REF ref;
	Object *owner = NULL;

	if (!(script_data->base_native_type == "")) {
		owner = ClassDB::instance(script_data->base_native_type);
	} else {
		owner = memnew(Reference);
	}

	Reference *r = Object::cast_to<Reference>(owner);
	if (r) {
		ref = REF(r);
	}

	NativeScriptInstance *instance = (NativeScriptInstance *)instance_create(owner);

	owner->set_script_instance(instance);

	if (!instance) {
		if (ref.is_null()) {
			memdelete(owner); //no owner, sorry
		}
		return Variant();
	}

	if (ref.is_valid()) {
		return ref;
	} else {
		return owner;
	}
}

// TODO(karroffel): implement this
NativeScript::NativeScript() {
	library = Ref<GDNative>();
	lib_path = "";
	class_name = "";
#ifndef NO_THREADS
	owners_lock = Mutex::create();
#endif
}

// TODO(karroffel): implement this
NativeScript::~NativeScript() {
	NSL->unregister_script(this);

#ifndef NO_THREADS
	memdelete(owners_lock);
#endif
}

////// ScriptInstance stuff

#define GET_SCRIPT_DESC() script->get_script_desc()

void NativeScriptInstance::_ml_call_reversed(NativeScriptDesc *script_data, const StringName &p_method, const Variant **p_args, int p_argcount) {
	if (script_data->base_data) {
		_ml_call_reversed(script_data->base_data, p_method, p_args, p_argcount);
	}

	Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find(p_method);
	if (E) {
		godot_variant res = E->get().method.method((godot_object *)owner, E->get().method.method_data, userdata, p_argcount, (godot_variant **)p_args);
		godot_variant_destroy(&res);
	}
}

bool NativeScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {
		OrderedHashMap<StringName, NativeScriptDesc::Property>::Element P = script_data->properties.find(p_name);
		if (P) {
			P.get().setter.set_func((godot_object *)owner,
					P.get().setter.method_data,
					userdata,
					(godot_variant *)&p_value);
			return true;
		}

		Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find("_set");
		if (E) {
			Variant name = p_name;
			const Variant *args[2] = { &name, &p_value };

			E->get().method.method((godot_object *)owner,
					E->get().method.method_data,
					userdata,
					2,
					(godot_variant **)args);
			return true;
		}

		script_data = script_data->base_data;
	}
	return false;
}
bool NativeScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {
		OrderedHashMap<StringName, NativeScriptDesc::Property>::Element P = script_data->properties.find(p_name);
		if (P) {
			godot_variant value;
			value = P.get().getter.get_func((godot_object *)owner,
					P.get().getter.method_data,
					userdata);
			r_ret = *(Variant *)&value;
			godot_variant_destroy(&value);
			return true;
		}

		Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find("_get");
		if (E) {
			Variant name = p_name;
			const Variant *args[1] = { &name };

			godot_variant result;
			result = E->get().method.method((godot_object *)owner,
					E->get().method.method_data,
					userdata,
					1,
					(godot_variant **)args);
			r_ret = *(Variant *)&result;
			godot_variant_destroy(&result);
			if (r_ret.get_type() == Variant::NIL) {
				return false;
			}
			return true;
		}

		script_data = script_data->base_data;
	}
	return false;
}

void NativeScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	script->get_script_property_list(p_properties);

	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {

		Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find("_get_property_list");
		if (E) {

			godot_variant result;
			result = E->get().method.method((godot_object *)owner,
					E->get().method.method_data,
					userdata,
					0,
					NULL);
			Variant res = *(Variant *)&result;
			godot_variant_destroy(&result);

			if (res.get_type() != Variant::ARRAY) {
				ERR_EXPLAIN("_get_property_list must return an array of dictionaries");
				ERR_FAIL();
			}

			Array arr = res;
			for (int i = 0; i < arr.size(); i++) {
				Dictionary d = arr[i];

				ERR_CONTINUE(!d.has("name"));
				ERR_CONTINUE(!d.has("type"));

				PropertyInfo info;

				info.type = Variant::Type(d["type"].operator int64_t());
				ERR_CONTINUE(info.type < 0 || info.type >= Variant::VARIANT_MAX);

				info.name = d["name"];
				ERR_CONTINUE(info.name == "");

				if (d.has("hint")) {
					info.hint = PropertyHint(d["hint"].operator int64_t());
				}

				if (d.has("hint_string")) {
					info.hint_string = d["hint_string"];
				}

				if (d.has("usage")) {
					info.usage = d["usage"];
				}

				p_properties->push_back(info);
			}
		}

		script_data = script_data->base_data;
	}
	return;
}

Variant::Type NativeScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {

	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {

		OrderedHashMap<StringName, NativeScriptDesc::Property>::Element P = script_data->properties.find(p_name);
		if (P) {
			*r_is_valid = true;
			return P.get().info.type;
		}

		script_data = script_data->base_data;
	}
	return Variant::NIL;
}

void NativeScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	script->get_method_list(p_list);
}

bool NativeScriptInstance::has_method(const StringName &p_method) const {
	return script->has_method(p_method);
}

Variant NativeScriptInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {
		Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find(p_method);
		if (E) {
			godot_variant result;
			result = E->get().method.method((godot_object *)owner,
					E->get().method.method_data,
					userdata,
					p_argcount,
					(godot_variant **)p_args);
			Variant res = *(Variant *)&result;
			godot_variant_destroy(&result);
			r_error.error = Variant::CallError::CALL_OK;
			return res;
		}

		script_data = script_data->base_data;
	}

	r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

void NativeScriptInstance::notification(int p_notification) {
	Variant value = p_notification;
	const Variant *args[1] = { &value };
	call_multilevel("_notification", args, 1);
}

void NativeScriptInstance::refcount_incremented() {
	Variant::CallError err;
	call("_refcount_incremented", NULL, 0, err);
	if (err.error != Variant::CallError::CALL_OK && err.error != Variant::CallError::CALL_ERROR_INVALID_METHOD) {
		ERR_PRINT("Failed to invoke _refcount_incremented - should not happen");
	}
}

bool NativeScriptInstance::refcount_decremented() {
	Variant::CallError err;
	Variant ret = call("_refcount_decremented", NULL, 0, err);
	if (err.error != Variant::CallError::CALL_OK && err.error != Variant::CallError::CALL_ERROR_INVALID_METHOD) {
		ERR_PRINT("Failed to invoke _refcount_decremented - should not happen");
		return true; // assume we can destroy the object
	}
	if (err.error == Variant::CallError::CALL_ERROR_INVALID_METHOD) {
		// the method does not exist, default is true
		return true;
	}
	return ret;
}

Ref<Script> NativeScriptInstance::get_script() const {
	return script;
}

NativeScriptInstance::RPCMode NativeScriptInstance::get_rpc_mode(const StringName &p_method) const {

	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {

		Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find(p_method);
		if (E) {
			switch (E->get().rpc_mode) {
				case GODOT_METHOD_RPC_MODE_DISABLED:
					return RPC_MODE_DISABLED;
				case GODOT_METHOD_RPC_MODE_REMOTE:
					return RPC_MODE_REMOTE;
				case GODOT_METHOD_RPC_MODE_SYNC:
					return RPC_MODE_SYNC;
				case GODOT_METHOD_RPC_MODE_MASTER:
					return RPC_MODE_MASTER;
				case GODOT_METHOD_RPC_MODE_SLAVE:
					return RPC_MODE_SLAVE;
				default:
					return RPC_MODE_DISABLED;
			}
		}

		script_data = script_data->base_data;
	}

	return RPC_MODE_DISABLED;
}

// TODO(karroffel): implement this
NativeScriptInstance::RPCMode NativeScriptInstance::get_rset_mode(const StringName &p_variable) const {

	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {

		OrderedHashMap<StringName, NativeScriptDesc::Property>::Element E = script_data->properties.find(p_variable);
		if (E) {
			switch (E.get().rset_mode) {
				case GODOT_METHOD_RPC_MODE_DISABLED:
					return RPC_MODE_DISABLED;
				case GODOT_METHOD_RPC_MODE_REMOTE:
					return RPC_MODE_REMOTE;
				case GODOT_METHOD_RPC_MODE_SYNC:
					return RPC_MODE_SYNC;
				case GODOT_METHOD_RPC_MODE_MASTER:
					return RPC_MODE_MASTER;
				case GODOT_METHOD_RPC_MODE_SLAVE:
					return RPC_MODE_SLAVE;
				default:
					return RPC_MODE_DISABLED;
			}
		}

		script_data = script_data->base_data;
	}

	return RPC_MODE_DISABLED;
}

ScriptLanguage *NativeScriptInstance::get_language() {
	return NativeScriptLanguage::get_singleton();
}

void NativeScriptInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {
	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	while (script_data) {
		Map<StringName, NativeScriptDesc::Method>::Element *E = script_data->methods.find(p_method);
		if (E) {
			godot_variant res = E->get().method.method((godot_object *)owner,
					E->get().method.method_data,
					userdata,
					p_argcount,
					(godot_variant **)p_args);
			godot_variant_destroy(&res);
		}
		script_data = script_data->base_data;
	}
}

void NativeScriptInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {
	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	if (script_data) {
		_ml_call_reversed(script_data, p_method, p_args, p_argcount);
	}
}

NativeScriptInstance::~NativeScriptInstance() {

	NativeScriptDesc *script_data = GET_SCRIPT_DESC();

	if (!script_data)
		return;

	script_data->destroy_func.destroy_func((godot_object *)owner, script_data->destroy_func.method_data, userdata);

	if (owner) {

#ifndef NO_THREADS
		script->owners_lock->lock();
#endif

		script->instance_owners.erase(owner);

#ifndef NO_THREADS
		script->owners_lock->unlock();
#endif
	}
}

////// ScriptingLanguage stuff

NativeScriptLanguage *NativeScriptLanguage::singleton;

extern "C" void _native_script_hook();
void NativeScriptLanguage::_hacky_api_anchor() {
	_native_script_hook();
}

void NativeScriptLanguage::_unload_stuff() {
	for (Map<String, Map<StringName, NativeScriptDesc> >::Element *L = library_classes.front(); L; L = L->next()) {
		for (Map<StringName, NativeScriptDesc>::Element *C = L->get().front(); C; C = C->next()) {

			// free property stuff first
			for (OrderedHashMap<StringName, NativeScriptDesc::Property>::Element P = C->get().properties.front(); P; P = P.next()) {
				if (P.get().getter.free_func)
					P.get().getter.free_func(P.get().getter.method_data);

				if (P.get().setter.free_func)
					P.get().setter.free_func(P.get().setter.method_data);
			}

			// free method stuff
			for (Map<StringName, NativeScriptDesc::Method>::Element *M = C->get().methods.front(); M; M = M->next()) {
				if (M->get().method.free_func)
					M->get().method.free_func(M->get().method.method_data);
			}

			// free constructor/destructor
			if (C->get().create_func.free_func)
				C->get().create_func.free_func(C->get().create_func.method_data);

			if (C->get().destroy_func.free_func)
				C->get().destroy_func.free_func(C->get().destroy_func.method_data);
		}
	}
}

NativeScriptLanguage::NativeScriptLanguage() {
	NativeScriptLanguage::singleton = this;
#ifndef NO_THREADS
	mutex = Mutex::create();
#endif
}

// TODO(karroffel): implement this
NativeScriptLanguage::~NativeScriptLanguage() {
	// _unload_stuff(); // NOTE(karroffel): This gets called in ::finish()

	for (Map<String, Ref<GDNative> >::Element *L = NSL->library_gdnatives.front(); L; L = L->next()) {

		L->get()->terminate();
		NSL->library_classes.clear();
		NSL->library_gdnatives.clear();
		NSL->library_script_users.clear();
	}

#ifndef NO_THREADS
	memdelete(mutex);
#endif
}

String NativeScriptLanguage::get_name() const {
	return "NativeScript";
}

void _add_reload_node() {
#ifdef TOOLS_ENABLED
	NativeReloadNode *rn = memnew(NativeReloadNode);
	EditorNode::get_singleton()->add_child(rn);
#endif
}

// TODO(karroffel): implement this
void NativeScriptLanguage::init() {

#if defined(TOOLS_ENABLED) && defined(DEBUG_METHODS_ENABLED)

	List<String> args = OS::get_singleton()->get_cmdline_args();

	List<String>::Element *E = args.find("--gdnative-generate-json-api");

	if (E && E->next()) {
		if (generate_c_api(E->next()->get()) != OK) {
			ERR_PRINT("Failed to generate C API\n");
		}
	}
#endif

#ifdef TOOLS_ENABLED
	EditorNode::add_init_callback(&_add_reload_node);
#endif
}
String NativeScriptLanguage::get_type() const {
	return "NativeScript";
}
String NativeScriptLanguage::get_extension() const {
	return "gdns";
}
Error NativeScriptLanguage::execute_file(const String &p_path) {
	return OK; // Qué?
}
void NativeScriptLanguage::finish() {
	_unload_stuff();
}
void NativeScriptLanguage::get_reserved_words(List<String> *p_words) const {
}
void NativeScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {
}
void NativeScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {
}

// TODO(karroffel): implement this
Ref<Script> NativeScriptLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {
	NativeScript *s = memnew(NativeScript);
	s->set_class_name(p_class_name);
	// TODO(karroffel): use p_base_class_name
	return Ref<NativeScript>(s);
}
bool NativeScriptLanguage::validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions) const {
	return false;
}

Script *NativeScriptLanguage::create_script() const {
	NativeScript *script = memnew(NativeScript);
	return script;
}
bool NativeScriptLanguage::has_named_classes() const {
	return true;
}
bool NativeScriptLanguage::supports_builtin_mode() const {
	return true;
}
int NativeScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1;
}
String NativeScriptLanguage::make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const {
	return "";
}
void NativeScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
}
void NativeScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
}

// Debugging stuff here. Not used for now.
String NativeScriptLanguage::debug_get_error() const {
	return "";
}
int NativeScriptLanguage::debug_get_stack_level_count() const {
	return -1;
}
int NativeScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return -1;
}
String NativeScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return "";
}
String NativeScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return "";
}
void NativeScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
}
void NativeScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
}
void NativeScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
}
String NativeScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return "";
}
// Debugging stuff end.

void NativeScriptLanguage::reload_all_scripts() {
}

void NativeScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
}
void NativeScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdns");
}

void NativeScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
}

void NativeScriptLanguage::get_public_constants(List<Pair<String, Variant> > *p_constants) const {
}

void NativeScriptLanguage::profiling_start() {
}

void NativeScriptLanguage::profiling_stop() {
}

int NativeScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int NativeScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

#ifndef NO_THREADS
void NativeScriptLanguage::defer_init_library(Ref<GDNativeLibrary> lib, NativeScript *script) {
	MutexLock lock(mutex);
	libs_to_init.insert(lib);
	scripts_to_register.insert(script);
	has_objects_to_register = true;
}
#endif

void NativeScriptLanguage::init_library(const Ref<GDNativeLibrary> &lib) {
#ifndef NO_THREADS
	MutexLock lock(mutex);
#endif
	// See if this library was "registered" already.
	const String &lib_path = lib->get_active_library_path();
	ERR_EXPLAIN(lib->get_name() + " does not have a library for the current platform");
	ERR_FAIL_COND(lib_path.length() == 0);
	Map<String, Ref<GDNative> >::Element *E = library_gdnatives.find(lib_path);

	if (!E) {
		Ref<GDNative> gdn;
		gdn.instance();
		gdn->set_library(lib);

		// TODO(karroffel): check the return value?
		gdn->initialize();

		library_gdnatives.insert(lib_path, gdn);

		library_classes.insert(lib_path, Map<StringName, NativeScriptDesc>());

		if (!library_script_users.has(lib_path))
			library_script_users.insert(lib_path, Set<NativeScript *>());

		void *proc_ptr;

		Error err = gdn->get_symbol(_init_call_name, proc_ptr);

		if (err != OK) {
			ERR_PRINT(String("No " + _init_call_name + " in \"" + lib_path + "\" found").utf8().get_data());
		} else {
			((void (*)(godot_string *))proc_ptr)((godot_string *)&lib_path);
		}
	} else {
		// already initialized. Nice.
	}
}

void NativeScriptLanguage::register_script(NativeScript *script) {
#ifndef NO_THREADS
	MutexLock lock(mutex);
#endif
	library_script_users[script->lib_path].insert(script);
}

void NativeScriptLanguage::unregister_script(NativeScript *script) {
#ifndef NO_THREADS
	MutexLock lock(mutex);
#endif
	Map<String, Set<NativeScript *> >::Element *S = library_script_users.find(script->lib_path);
	if (S) {
		S->get().erase(script);
		if (S->get().size() == 0) {
			library_script_users.erase(S);
		}
	}
#ifndef NO_THREADS
	scripts_to_register.erase(script);
#endif
}

void NativeScriptLanguage::call_libraries_cb(const StringName &name) {
	// library_gdnatives is modified only from the main thread, so it's safe not to use mutex here
	for (Map<String, Ref<GDNative> >::Element *L = library_gdnatives.front(); L; L = L->next()) {
		if (L->get()->is_initialized()) {

			void *proc_ptr;
			Error err = L->get()->get_symbol(name, proc_ptr);

			if (!err) {
				((void (*)())proc_ptr)();
			}
		}
	}
}

void NativeScriptLanguage::frame() {
#ifndef NO_THREADS
	if (has_objects_to_register) {
		MutexLock lock(mutex);
		for (Set<Ref<GDNativeLibrary> >::Element *L = libs_to_init.front(); L; L = L->next()) {
			init_library(L->get());
		}
		libs_to_init.clear();
		for (Set<NativeScript *>::Element *S = scripts_to_register.front(); S; S = S->next()) {
			register_script(S->get());
		}
		scripts_to_register.clear();
		has_objects_to_register = false;
	}
#endif
	call_libraries_cb(_frame_call_name);
}

#ifndef NO_THREADS

void NativeScriptLanguage::thread_enter() {
	call_libraries_cb(_thread_enter_call_name);
}

void NativeScriptLanguage::thread_exit() {
	call_libraries_cb(_thread_exit_call_name);
}

#endif // NO_THREADS

void NativeReloadNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_notification"), &NativeReloadNode::_notification);
}

void NativeReloadNode::_notification(int p_what) {
#ifdef TOOLS_ENABLED

	switch (p_what) {
		case MainLoop::NOTIFICATION_WM_FOCUS_OUT: {

			if (unloaded)
				break;
#ifndef NO_THREADS
			MutexLock lock(NSL->mutex);
#endif
			NSL->_unload_stuff();
			for (Map<String, Ref<GDNative> >::Element *L = NSL->library_gdnatives.front(); L; L = L->next()) {

				L->get()->terminate();
				NSL->library_classes.erase(L->key());
			}

			unloaded = true;

		} break;

		case MainLoop::NOTIFICATION_WM_FOCUS_IN: {

			if (!unloaded)
				break;
#ifndef NO_THREADS
			MutexLock lock(NSL->mutex);
#endif
			Set<StringName> libs_to_remove;
			for (Map<String, Ref<GDNative> >::Element *L = NSL->library_gdnatives.front(); L; L = L->next()) {

				if (!L->get()->initialize()) {
					libs_to_remove.insert(L->key());
					continue;
				}

				NSL->library_classes.insert(L->key(), Map<StringName, NativeScriptDesc>());

				void *args[1] = {
					(void *)&L->key()
				};

				// here the library registers all the classes and stuff.

				void *proc_ptr;
				Error err = L->get()->get_symbol("godot_nativescript_init", proc_ptr);
				if (err != OK) {
					ERR_PRINT(String("No godot_nativescript_init in \"" + L->key() + "\" found").utf8().get_data());
				} else {
					((void (*)(void *))proc_ptr)((void *)&L->key());
				}

				for (Map<String, Set<NativeScript *> >::Element *U = NSL->library_script_users.front(); U; U = U->next()) {
					for (Set<NativeScript *>::Element *S = U->get().front(); S; S = S->next()) {
						NativeScript *script = S->get();

						if (script->placeholders.size() == 0)
							continue;

						for (Set<PlaceHolderScriptInstance *>::Element *P = script->placeholders.front(); P; P = P->next()) {
							script->_update_placeholder(P->get());
						}
					}
				}
			}

			unloaded = false;

			for (Set<StringName>::Element *R = libs_to_remove.front(); R; R = R->next()) {
				NSL->library_gdnatives.erase(R->get());
			}

		} break;
		default: {
		};
	}
#endif
}

RES ResourceFormatLoaderNativeScript::load(const String &p_path, const String &p_original_path, Error *r_error) {
	ResourceFormatLoaderText rsflt;
	return rsflt.load(p_path, p_original_path, r_error);
}

void ResourceFormatLoaderNativeScript::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdns");
}

bool ResourceFormatLoaderNativeScript::handles_type(const String &p_type) const {
	return (p_type == "Script" || p_type == "NativeScript");
}

String ResourceFormatLoaderNativeScript::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "gdns")
		return "NativeScript";
	return "";
}

Error ResourceFormatSaverNativeScript::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	ResourceFormatSaverText rfst;
	return rfst.save(p_path, p_resource, p_flags);
}

bool ResourceFormatSaverNativeScript::recognize(const RES &p_resource) const {
	return Object::cast_to<NativeScript>(*p_resource) != NULL;
}

void ResourceFormatSaverNativeScript::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<NativeScript>(*p_resource)) {
		p_extensions->push_back("gdns");
	}
}
