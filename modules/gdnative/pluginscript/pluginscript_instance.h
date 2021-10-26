/*************************************************************************/
/*  pluginscript_instance.h                                              */
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

#ifndef PLUGINSCRIPT_INSTANCE_H
#define PLUGINSCRIPT_INSTANCE_H

// Godot imports
#include "core/object/script_language.h"

// PluginScript imports
#include <pluginscript/godot_pluginscript.h>

class PluginScript;

class PluginScriptInstance : public ScriptInstance {
	friend class PluginScript;

private:
	Ref<PluginScript> _script;
	Object *_owner = nullptr;
	Variant _owner_variant;
	godot_pluginscript_instance_data *_data = nullptr;
	const godot_pluginscript_instance_desc *_desc = nullptr;

public:
	_FORCE_INLINE_ Object *get_owner() { return _owner; }

	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid = nullptr) const;

	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName &p_method) const;

	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	virtual void notification(int p_notification);
	virtual String to_string(bool *r_valid);

	virtual Ref<Script> get_script() const;

	virtual ScriptLanguage *get_language();

	virtual const Vector<Multiplayer::RPCConfig> get_rpc_methods() const;

	virtual void refcount_incremented();
	virtual bool refcount_decremented();

	PluginScriptInstance();
	bool init(PluginScript *p_script, Object *p_owner);
	virtual ~PluginScriptInstance();
};

#endif // PLUGINSCRIPT_INSTANCE_H
