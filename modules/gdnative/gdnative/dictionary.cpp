/*************************************************************************/
/*  dictionary.cpp                                                       */
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

#include "gdnative/dictionary.h"

#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

static_assert(sizeof(godot_dictionary) == sizeof(Dictionary), "Dictionary size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_dictionary_new(godot_dictionary *p_self) {
	memnew_placement(p_self, Dictionary);
}

void GDAPI godot_dictionary_new_copy(godot_dictionary *r_dest, const godot_dictionary *p_src) {
	memnew_placement(r_dest, Dictionary(*(Dictionary *)p_src));
}

void GDAPI godot_dictionary_destroy(godot_dictionary *p_self) {
	Dictionary *self = (Dictionary *)p_self;
	self->~Dictionary();
}

godot_variant GDAPI *godot_dictionary_operator_index(godot_dictionary *p_self, const godot_variant *p_key) {
	Dictionary *self = (Dictionary *)p_self;
	return (godot_variant *)&self->operator[](*((const Variant *)p_key));
}

const godot_variant GDAPI *godot_dictionary_operator_index_const(const godot_dictionary *p_self, const godot_variant *p_key) {
	const Dictionary *self = (const Dictionary *)p_self;
	return (const godot_variant *)&self->operator[](*((const Variant *)p_key));
}

#ifdef __cplusplus
}
#endif
