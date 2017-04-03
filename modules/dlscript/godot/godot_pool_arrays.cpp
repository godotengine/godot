#include "godot_pool_arrays.h"

#include "array.h"
#include "dvector.h"
#include "variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void _pool_arrays_api_anchor() {
}

#define memnew_placement_custom(m_placement, m_class, m_constr) _post_initialize(new (m_placement, sizeof(m_class), "") m_constr)

// byte

void GDAPI godot_pool_byte_array_new(godot_pool_byte_array *p_pba) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	memnew_placement(pba, PoolVector<uint8_t>);
}

void GDAPI godot_pool_byte_array_new_with_array(godot_pool_byte_array *p_pba, const godot_array *p_a) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<uint8_t>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_byte_array_append(godot_pool_byte_array *p_pba, const uint8_t p_data) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	pba->append(p_data);
}

void GDAPI godot_pool_byte_array_append_array(godot_pool_byte_array *p_pba, const godot_pool_byte_array *p_array) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	PoolVector<uint8_t> *array = (PoolVector<uint8_t> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_byte_array_insert(godot_pool_byte_array *p_pba, const godot_int p_idx, const uint8_t p_data) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	return pba->insert(p_idx, p_data);
}

void GDAPI godot_pool_byte_array_invert(godot_pool_byte_array *p_pba) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_byte_array_push_back(godot_pool_byte_array *p_pba, const uint8_t p_data) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	pba->push_back(p_data);
}

void GDAPI godot_pool_byte_array_remove(godot_pool_byte_array *p_pba, const godot_int p_idx) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_byte_array_resize(godot_pool_byte_array *p_pba, const godot_int p_size) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_byte_array_set(godot_pool_byte_array *p_pba, const godot_int p_idx, const uint8_t p_data) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	pba->set(p_idx, p_data);
}

uint8_t GDAPI godot_pool_byte_array_get(godot_pool_byte_array *p_pba, const godot_int p_idx) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	return pba->get(p_idx);
}

godot_int GDAPI godot_pool_byte_array_size(godot_pool_byte_array *p_pba) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_byte_array_destroy(godot_pool_byte_array *p_pba) {
	((PoolVector<uint8_t> *)p_pba)->~PoolVector();
}

// int

void GDAPI godot_pool_int_array_new(godot_pool_int_array *p_pba) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	memnew_placement(pba, PoolVector<uint8_t>);
}

void GDAPI godot_pool_int_array_new_with_array(godot_pool_int_array *p_pba, const godot_array *p_a) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<uint8_t>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_int_array_append(godot_pool_int_array *p_pba, const godot_int p_data) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->append(p_data);
}

void GDAPI godot_pool_int_array_append_array(godot_pool_int_array *p_pba, const godot_pool_int_array *p_array) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	PoolVector<godot_int> *array = (PoolVector<godot_int> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_int_array_insert(godot_pool_int_array *p_pba, const godot_int p_idx, const godot_int p_data) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	return pba->insert(p_idx, p_data);
}

void GDAPI godot_pool_int_array_invert(godot_pool_int_array *p_pba) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_int_array_push_back(godot_pool_int_array *p_pba, const godot_int p_data) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->push_back(p_data);
}

void GDAPI godot_pool_int_array_remove(godot_pool_int_array *p_pba, const godot_int p_idx) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_int_array_resize(godot_pool_int_array *p_pba, const godot_int p_size) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_int_array_set(godot_pool_int_array *p_pba, const godot_int p_idx, const godot_int p_data) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->set(p_idx, p_data);
}

godot_int GDAPI godot_pool_int_array_get(godot_pool_int_array *p_pba, const godot_int p_idx) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	return pba->get(p_idx);
}

godot_int GDAPI godot_pool_int_array_size(godot_pool_int_array *p_pba) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_int_array_destroy(godot_pool_int_array *p_pba) {
	((PoolVector<godot_int> *)p_pba)->~PoolVector();
}

// real

void GDAPI godot_pool_real_array_new(godot_pool_real_array *p_pba) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	memnew_placement(pba, PoolVector<uint8_t>);
}

void GDAPI godot_pool_real_array_new_with_array(godot_pool_real_array *p_pba, const godot_array *p_a) {
	PoolVector<uint8_t> *pba = (PoolVector<uint8_t> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<uint8_t>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_real_array_append(godot_pool_real_array *p_pba, const godot_real p_data) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	pba->append(p_data);
}

void GDAPI godot_pool_real_array_append_array(godot_pool_real_array *p_pba, const godot_pool_real_array *p_array) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	PoolVector<godot_real> *array = (PoolVector<godot_real> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_real_array_insert(godot_pool_real_array *p_pba, const godot_int p_idx, const godot_real p_data) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	return pba->insert(p_idx, p_data);
}

void GDAPI godot_pool_real_array_invert(godot_pool_real_array *p_pba) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_real_array_push_back(godot_pool_real_array *p_pba, const godot_real p_data) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	pba->push_back(p_data);
}

void GDAPI godot_pool_real_array_remove(godot_pool_real_array *p_pba, const godot_int p_idx) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_real_array_resize(godot_pool_real_array *p_pba, const godot_int p_size) {
	PoolVector<godot_int> *pba = (PoolVector<godot_int> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_real_array_set(godot_pool_real_array *p_pba, const godot_int p_idx, const godot_real p_data) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	pba->set(p_idx, p_data);
}

godot_real GDAPI godot_pool_real_array_get(godot_pool_real_array *p_pba, const godot_int p_idx) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	return pba->get(p_idx);
}

godot_int GDAPI godot_pool_real_array_size(godot_pool_real_array *p_pba) {
	PoolVector<godot_real> *pba = (PoolVector<godot_real> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_real_array_destroy(godot_pool_real_array *p_pba) {
	((PoolVector<godot_real> *)p_pba)->~PoolVector();
}

// string

void GDAPI godot_pool_string_array_new(godot_pool_string_array *p_pba) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	memnew_placement(pba, PoolVector<String>);
}

void GDAPI godot_pool_string_array_new_with_array(godot_pool_string_array *p_pba, const godot_array *p_a) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<String>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_string_array_append(godot_pool_string_array *p_pba, const godot_string *p_data) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	String &s = *(String *)p_data;
	pba->append(s);
}

void GDAPI godot_pool_string_array_append_array(godot_pool_string_array *p_pba, const godot_pool_string_array *p_array) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	PoolVector<String> *array = (PoolVector<String> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_string_array_insert(godot_pool_string_array *p_pba, const godot_int p_idx, const godot_string *p_data) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	String &s = *(String *)p_data;
	return pba->insert(p_idx, s);
}

void GDAPI godot_pool_string_array_invert(godot_pool_string_array *p_pba) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_string_array_push_back(godot_pool_string_array *p_pba, const godot_string *p_data) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	String &s = *(String *)p_data;
	pba->push_back(s);
}

void GDAPI godot_pool_string_array_remove(godot_pool_string_array *p_pba, const godot_int p_idx) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_string_array_resize(godot_pool_string_array *p_pba, const godot_int p_size) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_string_array_set(godot_pool_string_array *p_pba, const godot_int p_idx, const godot_string *p_data) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	String &s = *(String *)p_data;
	pba->set(p_idx, s);
}

godot_string GDAPI godot_pool_string_array_get(godot_pool_string_array *p_pba, const godot_int p_idx) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	godot_string str;
	String *s = (String *)&str;
	memnew_placement(s, String);
	*s = pba->get(p_idx);
	return str;
}

godot_int GDAPI godot_pool_string_array_size(godot_pool_string_array *p_pba) {
	PoolVector<String> *pba = (PoolVector<String> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_string_array_destroy(godot_pool_string_array *p_pba) {
	((PoolVector<String> *)p_pba)->~PoolVector();
}

// vector2

void GDAPI godot_pool_vector2_array_new(godot_pool_vector2_array *p_pba) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	memnew_placement(pba, PoolVector<Vector2>);
}

void GDAPI godot_pool_vector2_array_new_with_array(godot_pool_vector2_array *p_pba, const godot_array *p_a) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<Vector2>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_vector2_array_append(godot_pool_vector2_array *p_pba, const godot_vector2 *p_data) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	Vector2 &s = *(Vector2 *)p_data;
	pba->append(s);
}

void GDAPI godot_pool_vector2_array_append_array(godot_pool_vector2_array *p_pba, const godot_pool_vector2_array *p_array) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	PoolVector<Vector2> *array = (PoolVector<Vector2> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_vector2_array_insert(godot_pool_vector2_array *p_pba, const godot_int p_idx, const godot_vector2 *p_data) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	Vector2 &s = *(Vector2 *)p_data;
	return pba->insert(p_idx, s);
}

void GDAPI godot_pool_vector2_array_invert(godot_pool_vector2_array *p_pba) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_vector2_array_push_back(godot_pool_vector2_array *p_pba, const godot_vector2 *p_data) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	Vector2 &s = *(Vector2 *)p_data;
	pba->push_back(s);
}

void GDAPI godot_pool_vector2_array_remove(godot_pool_vector2_array *p_pba, const godot_int p_idx) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_vector2_array_resize(godot_pool_vector2_array *p_pba, const godot_int p_size) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_vector2_array_set(godot_pool_vector2_array *p_pba, const godot_int p_idx, const godot_vector2 *p_data) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	Vector2 &s = *(Vector2 *)p_data;
	pba->set(p_idx, s);
}

godot_vector2 GDAPI godot_pool_vector2_array_get(godot_pool_vector2_array *p_pba, const godot_int p_idx) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	godot_vector2 v;
	Vector2 *s = (Vector2 *)&v;
	*s = pba->get(p_idx);
	return v;
}

godot_int GDAPI godot_pool_vector2_array_size(godot_pool_vector2_array *p_pba) {
	PoolVector<Vector2> *pba = (PoolVector<Vector2> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_vector2_array_destroy(godot_pool_vector2_array *p_pba) {
	((PoolVector<Vector2> *)p_pba)->~PoolVector();
}

// vector3

void GDAPI godot_pool_vector3_array_new(godot_pool_vector3_array *p_pba) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	memnew_placement(pba, PoolVector<Vector3>);
}

void GDAPI godot_pool_vector3_array_new_with_array(godot_pool_vector3_array *p_pba, const godot_array *p_a) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<Vector3>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_vector3_array_append(godot_pool_vector3_array *p_pba, const godot_vector3 *p_data) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	Vector3 &s = *(Vector3 *)p_data;
	pba->append(s);
}

void GDAPI godot_pool_vector3_array_append_array(godot_pool_vector3_array *p_pba, const godot_pool_vector3_array *p_array) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	PoolVector<Vector3> *array = (PoolVector<Vector3> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_vector3_array_insert(godot_pool_vector3_array *p_pba, const godot_int p_idx, const godot_vector3 *p_data) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	Vector3 &s = *(Vector3 *)p_data;
	return pba->insert(p_idx, s);
}

void GDAPI godot_pool_vector3_array_invert(godot_pool_vector3_array *p_pba) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_vector3_array_push_back(godot_pool_vector3_array *p_pba, const godot_vector3 *p_data) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	Vector3 &s = *(Vector3 *)p_data;
	pba->push_back(s);
}

void GDAPI godot_pool_vector3_array_remove(godot_pool_vector3_array *p_pba, const godot_int p_idx) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_vector3_array_resize(godot_pool_vector3_array *p_pba, const godot_int p_size) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_vector3_array_set(godot_pool_vector3_array *p_pba, const godot_int p_idx, const godot_vector3 *p_data) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	Vector3 &s = *(Vector3 *)p_data;
	pba->set(p_idx, s);
}

godot_vector3 GDAPI godot_pool_vector3_array_get(godot_pool_vector3_array *p_pba, const godot_int p_idx) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	godot_vector3 v;
	Vector3 *s = (Vector3 *)&v;
	*s = pba->get(p_idx);
	return v;
}

godot_int GDAPI godot_pool_vector3_array_size(godot_pool_vector3_array *p_pba) {
	PoolVector<Vector3> *pba = (PoolVector<Vector3> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_vector3_array_destroy(godot_pool_vector3_array *p_pba) {
	((PoolVector<Vector3> *)p_pba)->~PoolVector();
}

// color

void GDAPI godot_pool_color_array_new(godot_pool_color_array *p_pba) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	memnew_placement(pba, PoolVector<Color>);
}

void GDAPI godot_pool_color_array_new_with_array(godot_pool_color_array *p_pba, const godot_array *p_a) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	Array *a = (Array *)p_a;
	memnew_placement(pba, PoolVector<Color>);

	pba->resize(a->size());
	for (size_t i = 0; i < a->size(); i++) {
		pba->set(i, (*a)[i]);
	}
}

void GDAPI godot_pool_color_array_append(godot_pool_color_array *p_pba, const godot_color *p_data) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	Color &s = *(Color *)p_data;
	pba->append(s);
}

void GDAPI godot_pool_color_array_append_array(godot_pool_color_array *p_pba, const godot_pool_color_array *p_array) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	PoolVector<Color> *array = (PoolVector<Color> *)p_array;
	pba->append_array(*array);
}

int GDAPI godot_pool_color_array_insert(godot_pool_color_array *p_pba, const godot_int p_idx, const godot_color *p_data) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	Color &s = *(Color *)p_data;
	return pba->insert(p_idx, s);
}

void GDAPI godot_pool_color_array_invert(godot_pool_color_array *p_pba) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	pba->invert();
}

void GDAPI godot_pool_color_array_push_back(godot_pool_color_array *p_pba, const godot_color *p_data) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	Color &s = *(Color *)p_data;
	pba->push_back(s);
}

void GDAPI godot_pool_color_array_remove(godot_pool_color_array *p_pba, const godot_int p_idx) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	pba->remove(p_idx);
}

void GDAPI godot_pool_color_array_resize(godot_pool_color_array *p_pba, const godot_int p_size) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	pba->resize(p_size);
}

void GDAPI godot_pool_color_array_set(godot_pool_color_array *p_pba, const godot_int p_idx, const godot_color *p_data) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	Color &s = *(Color *)p_data;
	pba->set(p_idx, s);
}

godot_color GDAPI godot_pool_color_array_get(godot_pool_color_array *p_pba, const godot_int p_idx) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	godot_color v;
	Color *s = (Color *)&v;
	*s = pba->get(p_idx);
	return v;
}

godot_int GDAPI godot_pool_color_array_size(godot_pool_color_array *p_pba) {
	PoolVector<Color> *pba = (PoolVector<Color> *)p_pba;
	return pba->size();
}

void GDAPI godot_pool_color_array_destroy(godot_pool_color_array *p_pba) {
	((PoolVector<Color> *)p_pba)->~PoolVector();
}

#ifdef __cplusplus
}
#endif
