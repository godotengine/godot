#include "godot_array.h"

#include "core/array.h"
#include "core/os/memory.h"

#include "core/color.h"
#include "core/dvector.h"

#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void _array_api_anchor() {
}

void GDAPI godot_array_new(godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	memnew_placement(a, Array);
}

void GDAPI godot_array_new_pool_color_array(godot_array *p_arr, const godot_pool_color_array *p_pca) {
	Array *a = (Array *)p_arr;
	PoolVector<Color> *pca = (PoolVector<Color> *)p_pca;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_vector3_array(godot_array *p_arr, const godot_pool_vector3_array *p_pv3a) {
	Array *a = (Array *)p_arr;
	PoolVector<Vector3> *pca = (PoolVector<Vector3> *)p_pv3a;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_vector2_array(godot_array *p_arr, const godot_pool_vector2_array *p_pv2a) {
	Array *a = (Array *)p_arr;
	PoolVector<Vector2> *pca = (PoolVector<Vector2> *)p_pv2a;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_string_array(godot_array *p_arr, const godot_pool_string_array *p_psa) {
	Array *a = (Array *)p_arr;
	PoolVector<String> *pca = (PoolVector<String> *)p_psa;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_real_array(godot_array *p_arr, const godot_pool_real_array *p_pra) {
	Array *a = (Array *)p_arr;
	PoolVector<godot_real> *pca = (PoolVector<godot_real> *)p_pra;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_int_array(godot_array *p_arr, const godot_pool_int_array *p_pia) {
	Array *a = (Array *)p_arr;
	PoolVector<godot_int> *pca = (PoolVector<godot_int> *)p_pia;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_new_pool_byte_array(godot_array *p_arr, const godot_pool_byte_array *p_pba) {
	Array *a = (Array *)p_arr;
	PoolVector<uint8_t> *pca = (PoolVector<uint8_t> *)p_pba;
	memnew_placement(a, Array);
	a->resize(pca->size());

	for (size_t i = 0; i < a->size(); i++) {
		Variant v = pca->operator[](i);
		a->operator[](i) = v;
	}
}

void GDAPI godot_array_set(godot_array *p_arr, const godot_int p_idx, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	a->operator[](p_idx) = *val;
}

godot_variant GDAPI *godot_array_get(godot_array *p_arr, const godot_int p_idx) {
	Array *a = (Array *)p_arr;
	return (godot_variant *)&a->operator[](p_idx);
}

void GDAPI godot_array_append(godot_array *p_arr, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	a->append(*val);
}

void GDAPI godot_array_clear(godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	a->clear();
}

godot_int GDAPI godot_array_count(godot_array *p_arr, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	return a->count(*val);
}

godot_bool GDAPI godot_array_empty(const godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	return a->empty();
}

void GDAPI godot_array_erase(godot_array *p_arr, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	a->erase(*val);
}

godot_variant GDAPI godot_array_front(const godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = a->front();
	return v;
}

godot_variant GDAPI godot_array_back(const godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = a->back();
	return v;
}

godot_int GDAPI godot_array_find(const godot_array *p_arr, const godot_variant *p_what, const godot_int p_from) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_what;
	return a->find(*val, p_from);
}

godot_int GDAPI godot_array_find_last(const godot_array *p_arr, const godot_variant *p_what) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_what;
	return a->find_last(*val);
}

godot_bool GDAPI godot_array_has(const godot_array *p_arr, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	return a->has(*val);
}

uint32_t GDAPI godot_array_hash(const godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	return a->hash();
}

void GDAPI godot_array_insert(godot_array *p_arr, const godot_int p_pos, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	a->insert(p_pos, *val);
}

void GDAPI godot_array_invert(godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	a->invert();
}

godot_bool GDAPI godot_array_is_shared(const godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	return false; // @Todo how do I do it?
}

godot_variant GDAPI godot_array_pop_back(godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = a->pop_back();
	return v;
}

godot_variant GDAPI godot_array_pop_front(godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	godot_variant v;
	Variant *val = (Variant *)&v;
	memnew_placement(val, Variant);
	*val = a->pop_front();
	return v;
}

void GDAPI godot_array_push_back(godot_array *p_arr, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	a->push_back(*val);
}

void GDAPI godot_array_push_front(godot_array *p_arr, const godot_variant *p_value) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_value;
	a->push_front(*val);
}

void GDAPI godot_array_remove(godot_array *p_arr, const godot_int p_idx) {
	Array *a = (Array *)p_arr;
	a->remove(p_idx);
}

void GDAPI godot_array_resize(godot_array *p_arr, const godot_int p_size) {
	Array *a = (Array *)p_arr;
	a->resize(p_size);
}

godot_int GDAPI godot_array_rfind(const godot_array *p_arr, const godot_variant *p_what, const godot_int p_from) {
	Array *a = (Array *)p_arr;
	Variant *val = (Variant *)p_what;
	return a->rfind(*val, p_from);
}

godot_int GDAPI godot_array_size(const godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	return a->size();
}

void GDAPI godot_array_sort(godot_array *p_arr) {
	Array *a = (Array *)p_arr;
	a->sort();
}

void GDAPI godot_array_sort_custom(godot_array *p_arr, godot_object *p_obj, const godot_string *p_func) {
	Array *a = (Array *)p_arr;
	String *func = (String *)p_func;
	a->sort_custom((Object *)p_obj, *func);
}

void GDAPI godot_array_destroy(godot_array *p_arr) {
	((Array *)p_arr)->~Array();
}

#ifdef __cplusplus
}
#endif
