/*************************************************************************/
/*  gd_mono_class.h                                                      */
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
#ifndef GD_MONO_CLASS_H
#define GD_MONO_CLASS_H

#include <mono/metadata/debug-helpers.h>

#include "map.h"
#include "ustring.h"

#include "gd_mono_field.h"
#include "gd_mono_header.h"
#include "gd_mono_method.h"
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

	bool methods_fetched;
	HashMap<MethodKey, GDMonoMethod *, MethodKey::Hasher> methods;

	bool fields_fetched;
	Map<StringName, GDMonoField *> fields;
	Vector<GDMonoField *> fields_list;

	friend class GDMonoAssembly;
	GDMonoClass(const StringName &p_namespace, const StringName &p_name, MonoClass *p_class, GDMonoAssembly *p_assembly);

public:
	static MonoType *get_raw_type(GDMonoClass *p_class);

	bool is_assignable_from(GDMonoClass *p_from) const;

	_FORCE_INLINE_ StringName get_namespace() const { return namespace_name; }
	_FORCE_INLINE_ StringName get_name() const { return class_name; }

	_FORCE_INLINE_ MonoClass *get_raw() const { return mono_class; }
	_FORCE_INLINE_ const GDMonoAssembly *get_assembly() const { return assembly; }

	String get_full_name() const;

	GDMonoClass *get_parent_class();

#ifdef TOOLS_ENABLED
	Vector<MonoClassField *> get_enum_fields();
#endif

	bool has_method(const StringName &p_name);

	bool has_attribute(GDMonoClass *p_attr_class);
	MonoObject *get_attribute(GDMonoClass *p_attr_class);

	void fetch_attributes();
	void fetch_methods_with_godot_api_checks(GDMonoClass *p_native_base);

	GDMonoMethod *get_method(const StringName &p_name);
	GDMonoMethod *get_method(const StringName &p_name, int p_params_count);
	GDMonoMethod *get_method(MonoMethod *p_raw_method);
	GDMonoMethod *get_method(MonoMethod *p_raw_method, const StringName &p_name);
	GDMonoMethod *get_method(MonoMethod *p_raw_method, const StringName &p_name, int p_params_count);
	GDMonoMethod *get_method_with_desc(const String &p_description, bool p_includes_namespace);

	GDMonoField *get_field(const StringName &p_name);
	const Vector<GDMonoField *> &get_all_fields();

	~GDMonoClass();
};

#endif // GD_MONO_CLASS_H
