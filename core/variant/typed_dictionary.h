/**************************************************************************/
/*  typed_dictionary.h                                                    */
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

#pragma once

#include "core/variant/type_info.h"

template <typename K, typename V>
class TypedDictionary : public Dictionary {
public:
	_FORCE_INLINE_ void operator=(const Dictionary &p_dictionary) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_dictionary), "Cannot assign a dictionary with different element types.");
		Dictionary::operator=(p_dictionary);
	}

	_FORCE_INLINE_ TypedDictionary(const Dictionary &p_dictionary) {
		set_typed(GodotTypeInfo::Internal::get_variant_type<K>(), GodotTypeInfo::Internal::get_object_class_name_or_empty<K>(), Variant(),
				GodotTypeInfo::Internal::get_variant_type<V>(), GodotTypeInfo::Internal::get_object_class_name_or_empty<V>(), Variant());
		if (is_same_typed(p_dictionary)) {
			Dictionary::operator=(p_dictionary);
		} else {
			assign(p_dictionary);
		}
	}

	_FORCE_INLINE_ TypedDictionary(std::initializer_list<KeyValue<Variant, Variant>> p_init) :
			TypedDictionary(Dictionary(p_init)) {}

	_FORCE_INLINE_ TypedDictionary() {
		set_typed(GodotTypeInfo::Internal::get_variant_type<K>(), GodotTypeInfo::Internal::get_object_class_name_or_empty<K>(), Variant(),
				GodotTypeInfo::Internal::get_variant_type<V>(), GodotTypeInfo::Internal::get_object_class_name_or_empty<V>(), Variant());
	}
};
