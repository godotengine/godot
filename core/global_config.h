/*************************************************************************/
/*  globals.h                                                            */
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
#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

#include "object.h"
#include "os/thread_safe.h"
#include "set.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class GlobalConfig : public Object {

	GDCLASS(GlobalConfig, Object);
	_THREAD_SAFE_CLASS_

public:
	typedef Map<String, Variant> CustomMap;

	struct Singleton {
		StringName name;
		Object *ptr;
		Singleton(const StringName &p_name = StringName(), Object *p_ptr = NULL) {
			name = p_name;
			ptr = p_ptr;
		}
	};

protected:
	enum {
		NO_ORDER_BASE = 1 << 18
	};

	struct VariantContainer {
		int order;
		bool persist;
		Variant variant;
		Variant initial;
		bool hide_from_editor;
		bool overrided;
		VariantContainer() {
			order = 0;
			hide_from_editor = false;
			persist = false;
			overrided = false;
		}
		VariantContainer(const Variant &p_variant, int p_order, bool p_persist = false) {
			variant = p_variant;
			order = p_order;
			hide_from_editor = false;
			persist = p_persist;
			overrided = false;
		}
	};

	bool registering_order;
	int last_order;
	Map<StringName, VariantContainer> props;
	String resource_path;
	Map<StringName, PropertyInfo> custom_prop_info;
	bool disable_platform_override;
	bool using_datapack;
	List<String> input_presets;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static GlobalConfig *singleton;

	Error _load_settings(const String p_path);
	Error _load_settings_binary(const String p_path);

	Error _save_settings_text(const String &p_file, const Map<String, List<String> > &props, const CustomMap &p_custom = CustomMap());
	Error _save_settings_binary(const String &p_file, const Map<String, List<String> > &props, const CustomMap &p_custom = CustomMap());

	List<Singleton> singletons;
	Map<StringName, Object *> singleton_ptrs;

	Error _save_custom_bnd(const String &p_file);

	bool _load_resource_pack(const String &p_pack);

	void _add_property_info_bind(const Dictionary &p_info);

protected:
	static void _bind_methods();

public:
	bool has(String p_var) const;
	String localize_path(const String &p_path) const;
	String globalize_path(const String &p_path) const;

	void set_initial_value(const String &p_name, const Variant &p_value);
	bool property_can_revert(const String &p_name);
	Variant property_get_revert(const String &p_name);

	String get_resource_path() const;

	static GlobalConfig *get_singleton();

	void clear(const String &p_name);
	int get_order(const String &p_name) const;
	void set_order(const String &p_name, int p_order);

	Error setup(const String &p_path, const String &p_main_pack);

	Error save_custom(const String &p_path = "", const CustomMap &p_custom = CustomMap(), const Set<String> &p_ignore_masks = Set<String>());
	Error save();
	void set_custom_property_info(const String &p_prop, const PropertyInfo &p_info);

	void add_singleton(const Singleton &p_singleton);
	void get_singletons(List<Singleton> *p_singletons);

	bool has_singleton(const String &p_name) const;

	Vector<String> get_optimizer_presets() const;

	List<String> get_input_presets() const { return input_presets; }

	void set_disable_platform_override(bool p_disable);
	Object *get_singleton_object(const String &p_name) const;

	void register_global_defaults();

	bool is_using_datapack() const;

	void set_registering_order(bool p_registering);

	GlobalConfig();
	~GlobalConfig();
};

//not a macro any longer
Variant _GLOBAL_DEF(const String &p_var, const Variant &p_default);
#define GLOBAL_DEF(m_var, m_value) _GLOBAL_DEF(m_var, m_value)
#define GLOBAL_GET(m_var) GlobalConfig::get_singleton()->get(m_var)

#endif
