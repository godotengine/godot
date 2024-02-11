/**************************************************************************/
/*  property_list_helper.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef PROPERTY_LIST_HELPER_H
#define PROPERTY_LIST_HELPER_H

#include "core/object/object.h"

class PropertyListHelper {
	struct Property {
		PropertyInfo info;
		Variant default_value;
		StringName setter_name;
		StringName getter_name;
		Callable setter;
		Callable getter;
	};

	String prefix;
	HashMap<String, Property> property_list;

	const Property *_get_property(const String &p_property, int *r_index) const;
	void _bind_property(const Property &p_property, const Object *p_object);

public:
	void set_prefix(const String &p_prefix);
	void register_property(const PropertyInfo &p_info, const Variant &p_default, const StringName &p_setter, const StringName &p_getter);
	void setup_for_instance(const PropertyListHelper &p_base, const Object *p_object);

	void get_property_list(List<PropertyInfo> *p_list, int p_count) const;
	bool property_get_value(const String &p_property, Variant &r_ret) const;
	bool property_set_value(const String &p_property, const Variant &p_value) const;
	bool property_can_revert(const String &p_property) const;
	bool property_get_revert(const String &p_property, Variant &r_value) const;
};

#endif // PROPERTY_LIST_HELPER_H
