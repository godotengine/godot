/*************************************************************************/
/*  collections_glue.cpp                                                 */
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

#ifdef MONO_GLUE_ENABLED

#include <mono/metadata/exception.h>

#include "core/variant/array.h"

#include "../mono_gd/gd_mono_cache.h"
#include "../mono_gd/gd_mono_class.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../mono_gd/gd_mono_utils.h"

Array *godot_icall_Array_Ctor() {
	return memnew(Array);
}

void godot_icall_Array_Dtor(Array *ptr) {
	memdelete(ptr);
}

MonoObject *godot_icall_Array_At(Array *ptr, int32_t index) {
	if (index < 0 || index >= ptr->size()) {
		GDMonoUtils::set_pending_exception(mono_get_exception_index_out_of_range());
		return nullptr;
	}
	return GDMonoMarshal::variant_to_mono_object(ptr->operator[](index));
}

MonoObject *godot_icall_Array_At_Generic(Array *ptr, int32_t index, uint32_t type_encoding, GDMonoClass *type_class) {
	if (index < 0 || index >= ptr->size()) {
		GDMonoUtils::set_pending_exception(mono_get_exception_index_out_of_range());
		return nullptr;
	}
	return GDMonoMarshal::variant_to_mono_object(ptr->operator[](index), ManagedType(type_encoding, type_class));
}

void godot_icall_Array_SetAt(Array *ptr, int32_t index, MonoObject *value) {
	if (index < 0 || index >= ptr->size()) {
		GDMonoUtils::set_pending_exception(mono_get_exception_index_out_of_range());
		return;
	}
	ptr->operator[](index) = GDMonoMarshal::mono_object_to_variant(value);
}

int32_t godot_icall_Array_Count(Array *ptr) {
	return ptr->size();
}

int32_t godot_icall_Array_Add(Array *ptr, MonoObject *item) {
	ptr->append(GDMonoMarshal::mono_object_to_variant(item));
	return ptr->size();
}

void godot_icall_Array_Clear(Array *ptr) {
	ptr->clear();
}

MonoBoolean godot_icall_Array_Contains(Array *ptr, MonoObject *item) {
	return ptr->find(GDMonoMarshal::mono_object_to_variant(item)) != -1;
}

void godot_icall_Array_CopyTo(Array *ptr, MonoArray *array, int32_t array_index) {
	unsigned int count = ptr->size();

	if (mono_array_length(array) < (array_index + count)) {
		MonoException *exc = mono_get_exception_argument("", "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");
		GDMonoUtils::set_pending_exception(exc);
		return;
	}

	for (unsigned int i = 0; i < count; i++) {
		MonoObject *boxed = GDMonoMarshal::variant_to_mono_object(ptr->operator[](i));
		mono_array_setref(array, array_index, boxed);
		array_index++;
	}
}

Array *godot_icall_Array_Ctor_MonoArray(MonoArray *mono_array) {
	Array *godot_array = memnew(Array);
	unsigned int count = mono_array_length(mono_array);
	godot_array->resize(count);
	for (unsigned int i = 0; i < count; i++) {
		MonoObject *item = mono_array_get(mono_array, MonoObject *, i);
		godot_icall_Array_SetAt(godot_array, i, item);
	}
	return godot_array;
}

Array *godot_icall_Array_Duplicate(Array *ptr, MonoBoolean deep) {
	return memnew(Array(ptr->duplicate(deep)));
}

Array *godot_icall_Array_Concatenate(Array *left, Array *right) {
	int count = left->size() + right->size();
	Array *new_array = memnew(Array(left->duplicate(false)));
	new_array->resize(count);
	for (unsigned int i = 0; i < (unsigned int)right->size(); i++) {
		new_array->operator[](i + left->size()) = right->operator[](i);
	}
	return new_array;
}

int32_t godot_icall_Array_IndexOf(Array *ptr, MonoObject *item) {
	return ptr->find(GDMonoMarshal::mono_object_to_variant(item));
}

void godot_icall_Array_Insert(Array *ptr, int32_t index, MonoObject *item) {
	if (index < 0 || index > ptr->size()) {
		GDMonoUtils::set_pending_exception(mono_get_exception_index_out_of_range());
		return;
	}
	ptr->insert(index, GDMonoMarshal::mono_object_to_variant(item));
}

MonoBoolean godot_icall_Array_Remove(Array *ptr, MonoObject *item) {
	int idx = ptr->find(GDMonoMarshal::mono_object_to_variant(item));
	if (idx >= 0) {
		ptr->remove_at(idx);
		return true;
	}
	return false;
}

void godot_icall_Array_RemoveAt(Array *ptr, int32_t index) {
	if (index < 0 || index >= ptr->size()) {
		GDMonoUtils::set_pending_exception(mono_get_exception_index_out_of_range());
		return;
	}
	ptr->remove_at(index);
}

int32_t godot_icall_Array_Resize(Array *ptr, int32_t new_size) {
	return (int32_t)ptr->resize(new_size);
}

void godot_icall_Array_Shuffle(Array *ptr) {
	ptr->shuffle();
}

void godot_icall_Array_Generic_GetElementTypeInfo(MonoReflectionType *refltype, uint32_t *type_encoding, GDMonoClass **type_class) {
	MonoType *elem_type = mono_reflection_type_get_type(refltype);

	*type_encoding = mono_type_get_type(elem_type);
	MonoClass *type_class_raw = mono_class_from_mono_type(elem_type);
	*type_class = GDMono::get_singleton()->get_class(type_class_raw);
}

MonoString *godot_icall_Array_ToString(Array *ptr) {
	return GDMonoMarshal::mono_string_from_godot(Variant(*ptr).operator String());
}

Dictionary *godot_icall_Dictionary_Ctor() {
	return memnew(Dictionary);
}

void godot_icall_Dictionary_Dtor(Dictionary *ptr) {
	memdelete(ptr);
}

MonoObject *godot_icall_Dictionary_GetValue(Dictionary *ptr, MonoObject *key) {
	Variant *ret = ptr->getptr(GDMonoMarshal::mono_object_to_variant(key));
	if (ret == nullptr) {
		MonoObject *exc = mono_object_new(mono_domain_get(), CACHED_CLASS(KeyNotFoundException)->get_mono_ptr());
#ifdef DEBUG_ENABLED
		CRASH_COND(!exc);
#endif
		GDMonoUtils::runtime_object_init(exc, CACHED_CLASS(KeyNotFoundException));
		GDMonoUtils::set_pending_exception((MonoException *)exc);
		return nullptr;
	}
	return GDMonoMarshal::variant_to_mono_object(ret);
}

MonoObject *godot_icall_Dictionary_GetValue_Generic(Dictionary *ptr, MonoObject *key, uint32_t type_encoding, GDMonoClass *type_class) {
	Variant *ret = ptr->getptr(GDMonoMarshal::mono_object_to_variant(key));
	if (ret == nullptr) {
		MonoObject *exc = mono_object_new(mono_domain_get(), CACHED_CLASS(KeyNotFoundException)->get_mono_ptr());
#ifdef DEBUG_ENABLED
		CRASH_COND(!exc);
#endif
		GDMonoUtils::runtime_object_init(exc, CACHED_CLASS(KeyNotFoundException));
		GDMonoUtils::set_pending_exception((MonoException *)exc);
		return nullptr;
	}
	return GDMonoMarshal::variant_to_mono_object(ret, ManagedType(type_encoding, type_class));
}

void godot_icall_Dictionary_SetValue(Dictionary *ptr, MonoObject *key, MonoObject *value) {
	ptr->operator[](GDMonoMarshal::mono_object_to_variant(key)) = GDMonoMarshal::mono_object_to_variant(value);
}

Array *godot_icall_Dictionary_Keys(Dictionary *ptr) {
	return memnew(Array(ptr->keys()));
}

Array *godot_icall_Dictionary_Values(Dictionary *ptr) {
	return memnew(Array(ptr->values()));
}

int32_t godot_icall_Dictionary_Count(Dictionary *ptr) {
	return ptr->size();
}

int32_t godot_icall_Dictionary_KeyValuePairs(Dictionary *ptr, Array **keys, Array **values) {
	*keys = godot_icall_Dictionary_Keys(ptr);
	*values = godot_icall_Dictionary_Values(ptr);
	return godot_icall_Dictionary_Count(ptr);
}

void godot_icall_Dictionary_KeyValuePairAt(Dictionary *ptr, int index, MonoObject **key, MonoObject **value) {
	*key = GDMonoMarshal::variant_to_mono_object(ptr->get_key_at_index(index));
	*value = GDMonoMarshal::variant_to_mono_object(ptr->get_value_at_index(index));
}

void godot_icall_Dictionary_Add(Dictionary *ptr, MonoObject *key, MonoObject *value) {
	Variant varKey = GDMonoMarshal::mono_object_to_variant(key);
	Variant *ret = ptr->getptr(varKey);
	if (ret != nullptr) {
		GDMonoUtils::set_pending_exception(mono_get_exception_argument("key", "An element with the same key already exists"));
		return;
	}
	ptr->operator[](varKey) = GDMonoMarshal::mono_object_to_variant(value);
}

void godot_icall_Dictionary_Clear(Dictionary *ptr) {
	ptr->clear();
}

MonoBoolean godot_icall_Dictionary_Contains(Dictionary *ptr, MonoObject *key, MonoObject *value) {
	// no dupes
	Variant *ret = ptr->getptr(GDMonoMarshal::mono_object_to_variant(key));
	return ret != nullptr && *ret == GDMonoMarshal::mono_object_to_variant(value);
}

MonoBoolean godot_icall_Dictionary_ContainsKey(Dictionary *ptr, MonoObject *key) {
	return ptr->has(GDMonoMarshal::mono_object_to_variant(key));
}

Dictionary *godot_icall_Dictionary_Duplicate(Dictionary *ptr, MonoBoolean deep) {
	return memnew(Dictionary(ptr->duplicate(deep)));
}

MonoBoolean godot_icall_Dictionary_RemoveKey(Dictionary *ptr, MonoObject *key) {
	return ptr->erase(GDMonoMarshal::mono_object_to_variant(key));
}

MonoBoolean godot_icall_Dictionary_Remove(Dictionary *ptr, MonoObject *key, MonoObject *value) {
	Variant varKey = GDMonoMarshal::mono_object_to_variant(key);

	// no dupes
	Variant *ret = ptr->getptr(varKey);
	if (ret != nullptr && *ret == GDMonoMarshal::mono_object_to_variant(value)) {
		ptr->erase(varKey);
		return true;
	}

	return false;
}

MonoBoolean godot_icall_Dictionary_TryGetValue(Dictionary *ptr, MonoObject *key, MonoObject **value) {
	Variant *ret = ptr->getptr(GDMonoMarshal::mono_object_to_variant(key));
	if (ret == nullptr) {
		*value = nullptr;
		return false;
	}
	*value = GDMonoMarshal::variant_to_mono_object(ret);
	return true;
}

MonoBoolean godot_icall_Dictionary_TryGetValue_Generic(Dictionary *ptr, MonoObject *key, MonoObject **value, uint32_t type_encoding, GDMonoClass *type_class) {
	Variant *ret = ptr->getptr(GDMonoMarshal::mono_object_to_variant(key));
	if (ret == nullptr) {
		*value = nullptr;
		return false;
	}
	*value = GDMonoMarshal::variant_to_mono_object(ret, ManagedType(type_encoding, type_class));
	return true;
}

void godot_icall_Dictionary_Generic_GetValueTypeInfo(MonoReflectionType *refltype, uint32_t *type_encoding, GDMonoClass **type_class) {
	MonoType *value_type = mono_reflection_type_get_type(refltype);

	*type_encoding = mono_type_get_type(value_type);
	MonoClass *type_class_raw = mono_class_from_mono_type(value_type);
	*type_class = GDMono::get_singleton()->get_class(type_class_raw);
}

MonoString *godot_icall_Dictionary_ToString(Dictionary *ptr) {
	return GDMonoMarshal::mono_string_from_godot(Variant(*ptr).operator String());
}

void godot_register_collections_icalls() {
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Ctor", godot_icall_Array_Ctor);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Ctor_MonoArray", godot_icall_Array_Ctor_MonoArray);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Dtor", godot_icall_Array_Dtor);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_At", godot_icall_Array_At);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_At_Generic", godot_icall_Array_At_Generic);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_SetAt", godot_icall_Array_SetAt);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Count", godot_icall_Array_Count);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Add", godot_icall_Array_Add);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Clear", godot_icall_Array_Clear);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Concatenate", godot_icall_Array_Concatenate);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Contains", godot_icall_Array_Contains);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_CopyTo", godot_icall_Array_CopyTo);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Duplicate", godot_icall_Array_Duplicate);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_IndexOf", godot_icall_Array_IndexOf);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Insert", godot_icall_Array_Insert);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Remove", godot_icall_Array_Remove);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_RemoveAt", godot_icall_Array_RemoveAt);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Resize", godot_icall_Array_Resize);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Shuffle", godot_icall_Array_Shuffle);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_Generic_GetElementTypeInfo", godot_icall_Array_Generic_GetElementTypeInfo);
	GDMonoUtils::add_internal_call("Godot.Collections.Array::godot_icall_Array_ToString", godot_icall_Array_ToString);

	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Ctor", godot_icall_Dictionary_Ctor);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Dtor", godot_icall_Dictionary_Dtor);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_GetValue", godot_icall_Dictionary_GetValue);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_GetValue_Generic", godot_icall_Dictionary_GetValue_Generic);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_SetValue", godot_icall_Dictionary_SetValue);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Keys", godot_icall_Dictionary_Keys);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Values", godot_icall_Dictionary_Values);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Count", godot_icall_Dictionary_Count);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_KeyValuePairs", godot_icall_Dictionary_KeyValuePairs);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_KeyValuePairAt", godot_icall_Dictionary_KeyValuePairAt);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Add", godot_icall_Dictionary_Add);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Clear", godot_icall_Dictionary_Clear);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Contains", godot_icall_Dictionary_Contains);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_ContainsKey", godot_icall_Dictionary_ContainsKey);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Duplicate", godot_icall_Dictionary_Duplicate);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_RemoveKey", godot_icall_Dictionary_RemoveKey);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Remove", godot_icall_Dictionary_Remove);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_TryGetValue", godot_icall_Dictionary_TryGetValue);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_TryGetValue_Generic", godot_icall_Dictionary_TryGetValue_Generic);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_Generic_GetValueTypeInfo", godot_icall_Dictionary_Generic_GetValueTypeInfo);
	GDMonoUtils::add_internal_call("Godot.Collections.Dictionary::godot_icall_Dictionary_ToString", godot_icall_Dictionary_ToString);
}

#endif // MONO_GLUE_ENABLED
