/*************************************************************************/
/*  type_db.h                                                            */
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

#ifndef TYPE_DB_H
#define TYPE_DB_H

#include "core/script_language.h"
#include "core/ustring.h"

class TypeDB : public Reference {
	GDCLASS(TypeDB, Reference);

	enum {
		SOURCE_NONE,
		SOURCE_ENGINE,
		SOURCE_SCRIPT_CLASS,
		SOURCE_CUSTOM_TYPE
	};

	struct TypeInfo {
		StringName name;
		StringName base;
		StringName native;
		Ref<Script> script;
		RES icon;
		int source = SOURCE_NONE;
	};

	typedef HashMap<StringName, TypeInfo> TypeMap;
	typedef HashMap<String, StringName> PathMap;

	TypeMap type_map;
	PathMap path_map;

protected:
	static void _bind_methods();

public:
	void refresh();

	bool path_exists(const String &p_path) const;
	StringName get_class_by_path(const String &p_path) const;
	String get_path_by_class(const StringName &p_type) const;
	StringName get_native(const StringName &p_type) const;

	RES get_icon(const StringName &p_type) const;

	Object *instance(const StringName &p_type) const;
	bool can_instance(const StringName &p_type) const;

	bool class_exists(const StringName &p_type) const;
	void get_class_list(List<StringName> *p_class_list) const;
	bool is_parent_class(const String &p_type, const String &p_inherits) const;
	String get_parent_class(const String &p_type) const;
	void get_inheritors_from_class(const String &p_type, List<StringName> *p_classes) const;

	TypeDB();
};

#endif // TYPE_DB_H