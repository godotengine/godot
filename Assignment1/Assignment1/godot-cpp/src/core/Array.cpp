#include "Array.hpp"
#include "GodotGlobal.hpp"
#include "Variant.hpp"

#include <cstdlib>

namespace godot {

class Object;

Array::Array() {
	godot::api->godot_array_new(&_godot_array);
}

Array::Array(const Array &other) {
	godot::api->godot_array_new_copy(&_godot_array, &other._godot_array);
}

Array &Array::operator=(const Array &other) {
	godot::api->godot_array_destroy(&_godot_array);
	godot::api->godot_array_new_copy(&_godot_array, &other._godot_array);
	return *this;
}

Array::Array(const PoolByteArray &a) {
	godot::api->godot_array_new_pool_byte_array(&_godot_array, (godot_pool_byte_array *)&a);
}

Array::Array(const PoolIntArray &a) {
	godot::api->godot_array_new_pool_int_array(&_godot_array, (godot_pool_int_array *)&a);
}

Array::Array(const PoolRealArray &a) {
	godot::api->godot_array_new_pool_real_array(&_godot_array, (godot_pool_real_array *)&a);
}

Array::Array(const PoolStringArray &a) {
	godot::api->godot_array_new_pool_string_array(&_godot_array, (godot_pool_string_array *)&a);
}

Array::Array(const PoolVector2Array &a) {
	godot::api->godot_array_new_pool_vector2_array(&_godot_array, (godot_pool_vector2_array *)&a);
}

Array::Array(const PoolVector3Array &a) {
	godot::api->godot_array_new_pool_vector3_array(&_godot_array, (godot_pool_vector3_array *)&a);
}

Array::Array(const PoolColorArray &a) {
	godot::api->godot_array_new_pool_color_array(&_godot_array, (godot_pool_color_array *)&a);
}

Variant &Array::operator[](const int idx) {
	godot_variant *v = godot::api->godot_array_operator_index(&_godot_array, idx);
	return *(Variant *)v;
}

Variant Array::operator[](const int idx) const {
	// Yes, I'm casting away the const... you can hate me now.
	// since the result is
	godot_variant *v = godot::api->godot_array_operator_index((godot_array *)&_godot_array, idx);
	return *(Variant *)v;
}

void Array::append(const Variant &v) {
	godot::api->godot_array_append(&_godot_array, (godot_variant *)&v);
}

void Array::clear() {
	godot::api->godot_array_clear(&_godot_array);
}

int Array::count(const Variant &v) {
	return godot::api->godot_array_count(&_godot_array, (godot_variant *)&v);
}

bool Array::empty() const {
	return godot::api->godot_array_empty(&_godot_array);
}

void Array::erase(const Variant &v) {
	godot::api->godot_array_erase(&_godot_array, (godot_variant *)&v);
}

Variant Array::front() const {
	godot_variant v = godot::api->godot_array_front(&_godot_array);
	return *(Variant *)&v;
}

Variant Array::back() const {
	godot_variant v = godot::api->godot_array_back(&_godot_array);
	return *(Variant *)&v;
}

int Array::find(const Variant &what, const int from) {
	return godot::api->godot_array_find(&_godot_array, (godot_variant *)&what, from);
}

int Array::find_last(const Variant &what) {
	return godot::api->godot_array_find_last(&_godot_array, (godot_variant *)&what);
}

bool Array::has(const Variant &what) const {
	return godot::api->godot_array_has(&_godot_array, (godot_variant *)&what);
}

uint32_t Array::hash() const {
	return godot::api->godot_array_hash(&_godot_array);
}

void Array::insert(const int pos, const Variant &value) {
	godot::api->godot_array_insert(&_godot_array, pos, (godot_variant *)&value);
}

void Array::invert() {
	godot::api->godot_array_invert(&_godot_array);
}

Variant Array::pop_back() {
	godot_variant v = godot::api->godot_array_pop_back(&_godot_array);
	return *(Variant *)&v;
}

Variant Array::pop_front() {
	godot_variant v = godot::api->godot_array_pop_front(&_godot_array);
	return *(Variant *)&v;
}

void Array::push_back(const Variant &v) {
	godot::api->godot_array_push_back(&_godot_array, (godot_variant *)&v);
}

void Array::push_front(const Variant &v) {
	godot::api->godot_array_push_front(&_godot_array, (godot_variant *)&v);
}

void Array::remove(const int idx) {
	godot::api->godot_array_remove(&_godot_array, idx);
}

int Array::size() const {
	return godot::api->godot_array_size(&_godot_array);
}

void Array::resize(const int size) {
	godot::api->godot_array_resize(&_godot_array, size);
}

int Array::rfind(const Variant &what, const int from) {
	return godot::api->godot_array_rfind(&_godot_array, (godot_variant *)&what, from);
}

void Array::sort() {
	godot::api->godot_array_sort(&_godot_array);
}

void Array::sort_custom(Object *obj, const String &func) {
	godot::api->godot_array_sort_custom(&_godot_array, (godot_object *)obj, (godot_string *)&func);
}

int Array::bsearch(const Variant &value, const bool before) {
	return godot::api->godot_array_bsearch(&_godot_array, (godot_variant *)&value, before);
}

int Array::bsearch_custom(const Variant &value, const Object *obj,
		const String &func, const bool before) {
	return godot::api->godot_array_bsearch_custom(&_godot_array, (godot_variant *)&value,
			(godot_object *)obj, (godot_string *)&func, before);
}

Array Array::duplicate(const bool deep) const {
	godot_array arr = godot::core_1_1_api->godot_array_duplicate(&_godot_array, deep);
	return *(Array *)&arr;
}

Variant Array::max() const {
	godot_variant v = godot::core_1_1_api->godot_array_max(&_godot_array);
	return *(Variant *)&v;
}

Variant Array::min() const {
	godot_variant v = godot::core_1_1_api->godot_array_min(&_godot_array);
	return *(Variant *)&v;
}

void Array::shuffle() {
	godot::core_1_1_api->godot_array_shuffle(&_godot_array);
}

Array::~Array() {
	godot::api->godot_array_destroy(&_godot_array);
}

} // namespace godot
