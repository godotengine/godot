/*************************************************************************/
/*  dictionary.h                                                         */
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

#ifndef GODOT_DICTIONARY_H
#define GODOT_DICTIONARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_DICTIONARY_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_DICTIONARY_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_DICTIONARY_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_DICTIONARY_SIZE];
} godot_dictionary;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/array.h>
#include <gdnative/gdnative.h>
#include <gdnative/variant.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_dictionary_new(godot_dictionary *r_dest);
void GDAPI godot_dictionary_new_copy(godot_dictionary *r_dest, const godot_dictionary *p_src);
void GDAPI godot_dictionary_destroy(godot_dictionary *p_self);

godot_dictionary GDAPI godot_dictionary_duplicate(const godot_dictionary *p_self, const godot_bool p_deep);

godot_int GDAPI godot_dictionary_size(const godot_dictionary *p_self);

godot_bool GDAPI godot_dictionary_empty(const godot_dictionary *p_self);

void GDAPI godot_dictionary_clear(godot_dictionary *p_self);

godot_bool GDAPI godot_dictionary_has(const godot_dictionary *p_self, const godot_variant *p_key);

godot_bool GDAPI godot_dictionary_has_all(const godot_dictionary *p_self, const godot_array *p_keys);

void GDAPI godot_dictionary_erase(godot_dictionary *p_self, const godot_variant *p_key);

godot_int GDAPI godot_dictionary_hash(const godot_dictionary *p_self);

godot_array GDAPI godot_dictionary_keys(const godot_dictionary *p_self);

godot_array GDAPI godot_dictionary_values(const godot_dictionary *p_self);

godot_variant GDAPI godot_dictionary_get(const godot_dictionary *p_self, const godot_variant *p_key);
void GDAPI godot_dictionary_set(godot_dictionary *p_self, const godot_variant *p_key, const godot_variant *p_value);

godot_variant GDAPI *godot_dictionary_operator_index(godot_dictionary *p_self, const godot_variant *p_key);

const godot_variant GDAPI *godot_dictionary_operator_index_const(const godot_dictionary *p_self, const godot_variant *p_key);

godot_variant GDAPI *godot_dictionary_next(const godot_dictionary *p_self, const godot_variant *p_key);

godot_bool GDAPI godot_dictionary_operator_equal(const godot_dictionary *p_self, const godot_dictionary *p_b);

godot_string GDAPI godot_dictionary_to_json(const godot_dictionary *p_self);

// GDNative core 1.1

godot_bool GDAPI godot_dictionary_erase_with_return(godot_dictionary *p_self, const godot_variant *p_key);

godot_variant GDAPI godot_dictionary_get_with_default(const godot_dictionary *p_self, const godot_variant *p_key, const godot_variant *p_default);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DICTIONARY_H
