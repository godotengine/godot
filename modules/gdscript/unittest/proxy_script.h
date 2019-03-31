/*************************************************************************/
/*  proxy_script.h                                                       */
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

#ifndef PROXY_SCRIPT_H
#define PROXY_SCRIPT_H

#include "method_watcher.h"

#include "core/script_language.h"

class ProxyScript : public Script {
	GDCLASS(ProxyScript, Script);

private:
	RefPtr m_script;

protected:
	virtual bool editor_can_reload_from_file(); // this is handled by editor better

	friend class PlaceHolderScriptInstance;
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder);

public:
	ProxyScript(RefPtr script);
	virtual bool can_instance() const;

	virtual Ref<Script> get_base_script() const; //for script inheritance

	virtual StringName get_instance_base_type() const; // this may not work in all scripts, will return empty if so
	virtual ScriptInstance *instance_create(Object *p_this);
	virtual PlaceHolderScriptInstance *placeholder_instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String &p_code);
	virtual Error reload(bool p_keep_state = false);

	virtual bool has_method(const StringName &p_method) const;
	virtual MethodInfo get_method_info(const StringName &p_method) const;

	virtual bool is_tool() const;
	virtual bool is_valid() const;

	virtual ScriptLanguage *get_language() const;

	virtual bool has_script_signal(const StringName &p_signal) const;
	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const;

	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const;

	virtual void update_exports(); //editor tool
	virtual void get_script_method_list(List<MethodInfo> *p_list) const;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const;

	virtual int get_member_line(const StringName &p_member) const;

	virtual void get_constants(Map<StringName, Variant> *p_constants);
	virtual void get_members(Set<StringName> *p_constants);

	virtual bool is_placeholder_fallback_enabled() const;
};

class ProxyScriptInstance : public ScriptInstance {
private:
	Ref<ProxyScript> m_script;
	ScriptInstance *m_script_instance;
	mutable MethodWatcher m_method_watcher;

	Variant _bind_method(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _add_property(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	void bind_method(const String &p_name, const Variant &p_return);
	void add_property(const String &p_name, const StringName p_setter, const StringName p_getter);

public:
	ProxyScriptInstance(Ref<ProxyScript> script, ScriptInstance *script_instance);
	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid = NULL) const;

	virtual Object *get_owner();
	virtual void get_property_state(List<Pair<StringName, Variant> > &state);

	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName &p_method) const;
	virtual Variant call(const StringName &p_method, VARIANT_ARG_LIST);
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void call_multilevel(const StringName &p_method, VARIANT_ARG_LIST);
	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void notification(int p_notification);

	virtual void refcount_incremented();
	virtual bool refcount_decremented(); //return true if it can die

	virtual Ref<Script> get_script() const;

	virtual bool is_placeholder() const;

	virtual void property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid);
	virtual Variant property_get_fallback(const StringName &p_name, bool *r_valid);

	virtual MultiplayerAPI::RPCMode get_rpc_mode(const StringName &p_method) const;
	virtual MultiplayerAPI::RPCMode get_rset_mode(const StringName &p_variable) const;

	virtual ScriptLanguage *get_language();
	virtual ~ProxyScriptInstance();

	const Vector<MethodWatcher::Args> get_calls(const String &p_name) const;
};

#endif // PROXY_SCRIPT_H
