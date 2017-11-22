/*************************************************************************/
/*  array.cpp                                                            */
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
#include "gdnative/array.h"

#include "core/array.h"
#include "core/os/memory.h"

#include "core/color.h"
#include "core/dvector.h"

#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_array_new(godot_array *r_dest) {
	Array *dest = (Array *)r_dest;
	memnew_placement(dest, Array);
}

void GDAPI godot_array_new_copy(godot_array *r_dest, const godot_array *p_src) {
	Array *dest = (Array *)r_dest;
	const Array *src = (const Array *)p_src;
	memnew_placement(dest, Array(*src));
}

void GDAPI godot_array_new_pool_color_array(godot_array *r_dest, const godot_pool_color_array *p_pca) {
	Array *dest = (Array *)r_dest;
	PoolVector<Color> *pca = (PoolVector<Color> *)p_pca;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_vector3_array(godot_array *r_dest, const godot_pool_vector3_array *p_pv3a) {
	Array *dest = (Array *)r_dest;
	PoolVector<Vector3> *pca = (PoolVector<Vector3> *)p_pv3a;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_vector2_array(godot_array *r_dest, const godot_pool_vector2_array *p_pv2a) {
	Array *dest = (Array *)r_dest;
	PoolVector<Vector2> *pca = (PoolVector<Vector2> *)p_pv2a;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_string_array(godot_array *r_dest, const godot_pool_string_array *p_psa) {
	Array *dest = (Array *)r_dest;
	PoolVector<String> *pca = (PoolVector<String> *)p_psa;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_real_array(godot_array *r_dest, const godot_pool_real_array *p_pra) {
	Array *dest = (Array *)r_dest;
	PoolVector<godot_real> *pca = (PoolVector<godot_real> *)p_pra;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_int_array(godot_array *r_dest, const godot_pool_int_array *p_pia) {
	Array *dest = (Array *)r_dest;
	PoolVector<godot_int> *pca = (PoolVector<godot_int> *)p_pia;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_byte_array(godot_array *r_dest, const godot_pool_byte_array *p_pba) {
	Array *dest = (Array *)r_dest;
	PoolVector<uint8_t> *pca = (PoolVector<uint8_t> *)p_pba;
	memnew_placement(dest, Array);
	dest->resize(pca->size());

	for (int i = 0; i < dest->size(); i++) {
		Variant v = pca->operator[](i);
		dest->operator[](i) = v;
	}
}

void GDAPI godot_array_set(godot_array *p_self, const godot_int p_idx, const godot_variant *p_value) {
	Array *self = (Array *)p_self;
	Variant *val = (Variant *)p_value;
	self->operator[](p_idx) = *val;
}

godot_variant GDAPI godot_array_get(const godot_array *p_self, const godot_int p_idx) {
	godot_variant raw_dest;
	Variant *dest = (Variant *)&raw_dest;
	const Array *self = (const Array *)p_self;
	memnew_placement(dest, Variant(self->operator[](p_idx)));
	return raw_dest;
}

godot_variant GDAPI *godot_array_operator_index(godot_array *p_self, const godot_int p_idx) {
	Array *self = (Array *)p_self;
	return (godot_variant *)&self->operator[](p_idx);
}

const godot_variant GDAPI *godot_array_operator_index_const(const godot_array *p_self, const godot_int p_idx) {
	const Array *self = (const Array *)p_self;
	return (const godot_variant *)&self->operator[](p_idx);
}

void GDAPI godot_array_append(godot_array *p_self, const godot_variant *p_value) {
	Array *self = (Array *)p_self;
	Variant *val = (Variant *)p_value;
	self->append(*val);
}

void GDAPI godot_array_clear(godot_array *p_self) {
	Array *self = (Array *)p_self;
	self->clear();
}

godot_int GDAPI godot_array_count(const godot_array *p_self, const godot_variant *p_value) {
	const Array *self = (const Array *)p_self;
	const Variant *val = (const Variant *)p_value;
	return self->count(*val);
}

godot_bool GDAPI godot_array_empty(const godot_array *p_self) {
	const Array *self = (const Array *)p_self;
	return self->empty();
}

void GDAPI godot_array_erase(godot_array *p_self, const godot_variant *p_value) {
	Array *self = (Array *)p_self;
	const Variant *val = (const Variant *)p_value;
	self->erase(*val);
}

godot_variant GDAPI godot_array_front(const godot_array *p_self) {
	const Array *self = (const Array *)p_self;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = self->front();
	return v;
}

godot_variant GDAPI godot_array_back(const godot_array *p_self) {
	const Array *self = (const Array *)p_self;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = self->back();
	return v;
}

godot_int GDAPI godot_array_find(const godot_array *p_self, const godot_variant *p_what, const godot_int p_from) {
	const Array *self = (const Array *)p_self;
	const Variant *val = (const Variant *)p_what;
	return self->find(*val, p_from);
}

godot_int GDAPI godot_array_find_last(const godot_array *p_self, const godot_variant *p_what) {
	const Array *self = (const Array *)p_self;
	const Variant *val = (const Variant *)p_what;
	return self->find_last(*val);
}

godot_bool GDAPI godot_array_has(const godot_array *p_self, const godot_variant *p_value) {
	const Array *self = (const Array *)p_self;
	const Variant *val = (const Variant *)p_value;
	return self->has(*val);
}

godot_int GDAPI godot_array_hash(const godot_array *p_self) {
	const Array *self = (const Array *)p_self;
	return self->hash();
}

void GDAPI godot_array_insert(godot_array *p_self, const godot_int p_pos, const godot_variant *p_value) {
	Array *self = (Array *)p_self;
	const Variant *val = (const Variant *)p_value;
	self->insert(p_pos, *val);
}

void GDAPI godot_array_invert(godot_array *p_self) {
	Array *self = (Array *)p_self;
	self->invert();
}

godot_variant GDAPI godot_array_pop_back(godot_array *p_self) {
	Array *self = (Array *)p_self;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = self->pop_back();
	return v;
}

godot_variant GDAPI godot_array_pop_front(godot_array *p_self) {
	Array *self = (Array *)p_self;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = self->pop_front();
	return v;
}

void GDAPI godot_array_push_back(godot_array *p_self, const godot_variant *p_value) {
	Array *self = (Array *)p_self;
	const Variant *val = (const Variant *)p_value;
	self->push_back(*val);
}

void GDAPI godot_array_push_front(godot_array *p_self, const godot_variant *p_value) {
	Array *self = (Array *)p_self;
	const Variant *val = (const Variant *)p_value;
	self->push_front(*val);
}

void GDAPI godot_array_remove(godot_array *p_self, const godot_int p_idx) {
	Array *self = (Array *)p_self;
	self->remove(p_idx);
}

void GDAPI godot_array_resize(godot_array *p_self, const godot_int p_size) {
	Array *self = (Array *)p_self;
	self->resize(p_size);
}

godot_int GDAPI godot_array_rfind(const godot_array *p_self, const godot_variant *p_what, const godot_int p_from) {
	const Array *self = (const Array *)p_self;
	const Variant *val = (const Variant *)p_what;
	return self->rfind(*val, p_from);
}

godot_int GDAPI godot_array_size(const godot_array *p_self) {
	const Array *self = (const Array *)p_self;
	return self->size();
}

void GDAPI godot_array_sort(godot_array *p_self) {
	Array *self = (Array *)p_self;
	self->sort();
}

void GDAPI godot_array_sort_custom(godot_array *p_self, godot_object *p_obj, const godot_string *p_func) {
	Array *self = (Array *)p_self;
	const String *func = (const String *)p_func;
	self->sort_custom((Object *)p_obj, *func);
}

godot_int GDAPI godot_array_bsearch(godot_array *p_self, const godot_variant *p_value, const godot_bool p_before) {
	Array *self = (Array *)p_self;
	return self->bsearch((const Variant *)p_value, p_before);
}

godot_int GDAPI godot_array_bsearch_custom(godot_array *p_self, const godot_variant *p_value, godot_object *p_obj, const godot_string *p_func, const godot_bool p_before) {
	Array *self = (Array *)p_self;
	const String *func = (const String *)p_func;
	return self->bsearch_custom((const Variant *)p_value, (Object *)p_obj, *func, p_before);
}

void GDAPI godot_array_destroy(godot_array *p_self) {
	((Array *)p_self)->~Array();
}

#ifdef __cplusplus
}
#endif
