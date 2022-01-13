/*************************************************************************/
/*  gd_mono_class.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GD_MONO_CLASS_H
#define GD_MONO_CLASS_H

#include "core/map.h"
#include "core/ustring.h"

#include "gd_mono_field.h"
#include "gd_mono_header.h"
#include "gd_mono_method.h"
#include "gd_mono_property.h"
#include "gd_mono_utils.h"

class GDMonoClass {
	struct MethodKey {
		struct Hasher {
			static _FORCE_INLINE_ uint32_t hash(const MethodKey &p_key) {
				uint32_t hash = 0;

				GDMonoUtils::hash_combine(hash, p_key.name.hash());
				GDMonoUtils::hash_combine(hash, HashMapHasherDefault::hash(p_key.params_count));

				return hash;
			}
		};

		_FORCE_INLINE_ bool operator==(const MethodKey &p_a) const {
			return p_a.params_count == params_count && p_a.name == name;
		}

		MethodKey() {}

		MethodKey(const StringName &p_name, int p_params_count) {
			name = p_name;
			params_count = p_params_count;
		}

		StringName name;
		int params_count;
	};

	StringName namespace_name;
	StringName class_name;

	MonoClass *mono_class;
	GDMonoAssembly *assembly;

	bool attrs_fetched;
	MonoCustomAttrInfo *attributes;

	// This contains both the original method names and remapped method names from the native Godot identifiers to the C# functions.
	// Most method-related functions refer to this and it's possible this is unintuitive for outside users; this may be a prime location for refactoring or renaming.
	bool methods_fetched;
	HashMap<MethodKey, GDMonoMethod *, MethodKey::Hasher> methods;

	bool method_list_fetched;
	Vector<GDMonoMethod *> method_list;

	bool fields_fetched;
	Map<StringName, GDMonoField *> fields;
	Vector<GDMonoField *> fields_list;

	bool properties_fetched;
	Map<StringName, GDMonoProperty *> properties;
	Vector<GDMonoProperty *> properties_list;

	bool delegates_fetched;
	Map<StringName, GDMonoClass *> delegates;
	Vector<GDMonoClass *> delegates_list;

	friend class GDMonoAssembly;
	GDMonoClass(const StringName &p_namespace, const StringName &p_name, MonoClass *p_class, GDMonoAssembly *p_assembly);

public:
	static String get_full_name(MonoClass *p_mono_class);
	static MonoType *get_mono_type(MonoClass *p_mono_class);

	String get_full_name() const;
	String get_type_desc() const;
	MonoType *get_mono_type() const;

	uint32_t get_flags() const;
	bool is_static() const;

	bool is_assignable_from(GDMonoClass *p_from) const;

	StringName get_namespace() const;
	_FORCE_INLINE_ StringName get_name() const { return class_name; }
	String get_name_for_lookup() const;

	_FORCE_INLINE_ MonoClass *get_mono_ptr() const { return mono_class; }
	_FORCE_INLINE_ const GDMonoAssembly *get_assembly() const { return assembly; }

	GDMonoClass *get_parent_class() const;
	GDMonoClass *get_nesting_class() const;

#ifdef TOOLS_ENABLED
	Vector<MonoClassField *> get_enum_fields();
#endif

	GDMonoMethod *get_fetched_method_unknown_params(const StringName &p_name);
	bool has_fetched_method_unknown_params(const StringName &p_name);

	bool has_attribute(GDMonoClass *p_attr_class);
	MonoObject *get_attribute(GDMonoClass *p_attr_class);

	void fetch_attributes();
	void fetch_methods_with_godot_api_checks(GDMonoClass *p_native_base);

	bool implements_interface(GDMonoClass *p_interface);
	bool has_public_parameterless_ctor();

	GDMonoMethod *get_method(const StringName &p_name, int p_params_count = 0);
	GDMonoMethod *get_method(MonoMethod *p_raw_method);
	GDMonoMethod *get_method(MonoMethod *p_raw_method, const StringName &p_name);
	GDMonoMethod *get_method(MonoMethod *p_raw_method, const StringName &p_name, int p_params_count);
	GDMonoMethod *get_method_with_desc(const String &p_description, bool p_include_namespace);

	GDMonoField *get_field(const StringName &p_name);
	const Vector<GDMonoField *> &get_all_fields();

	GDMonoProperty *get_property(const StringName &p_name);
	const Vector<GDMonoProperty *> &get_all_properties();

	const Vector<GDMonoClass *> &get_all_delegates();

	const Vector<GDMonoMethod *> &get_all_methods();

	~GDMonoClass();
};

#endif // GD_MONO_CLASS_H
