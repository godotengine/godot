#include "Dictionary.hpp"
#include "Array.hpp"
#include "GodotGlobal.hpp"
#include "Variant.hpp"

namespace godot {

Dictionary::Dictionary() {
	godot::api->godot_dictionary_new(&_godot_dictionary);
}

Dictionary::Dictionary(const Dictionary &other) {
	godot::api->godot_dictionary_new_copy(&_godot_dictionary, &other._godot_dictionary);
}

Dictionary &Dictionary::operator=(const Dictionary &other) {
	godot::api->godot_dictionary_destroy(&_godot_dictionary);
	godot::api->godot_dictionary_new_copy(&_godot_dictionary, &other._godot_dictionary);
	return *this;
}

void Dictionary::clear() {
	godot::api->godot_dictionary_clear(&_godot_dictionary);
}

bool Dictionary::empty() const {
	return godot::api->godot_dictionary_empty(&_godot_dictionary);
}

void Dictionary::erase(const Variant &key) {
	godot::api->godot_dictionary_erase(&_godot_dictionary, (godot_variant *)&key);
}

bool Dictionary::has(const Variant &key) const {
	return godot::api->godot_dictionary_has(&_godot_dictionary, (godot_variant *)&key);
}

bool Dictionary::has_all(const Array &keys) const {
	return godot::api->godot_dictionary_has_all(&_godot_dictionary, (godot_array *)&keys);
}

uint32_t Dictionary::hash() const {
	return godot::api->godot_dictionary_hash(&_godot_dictionary);
}

Array Dictionary::keys() const {
	godot_array a = godot::api->godot_dictionary_keys(&_godot_dictionary);
	return *(Array *)&a;
}

Variant &Dictionary::operator[](const Variant &key) {
	return *(Variant *)godot::api->godot_dictionary_operator_index(&_godot_dictionary, (godot_variant *)&key);
}

const Variant &Dictionary::operator[](const Variant &key) const {
	// oops I did it again
	return *(Variant *)godot::api->godot_dictionary_operator_index((godot_dictionary *)&_godot_dictionary, (godot_variant *)&key);
}

int Dictionary::size() const {
	return godot::api->godot_dictionary_size(&_godot_dictionary);
}

String Dictionary::to_json() const {
	godot_string s = godot::api->godot_dictionary_to_json(&_godot_dictionary);
	return *(String *)&s;
}

Array Dictionary::values() const {
	godot_array a = godot::api->godot_dictionary_values(&_godot_dictionary);
	return *(Array *)&a;
}

Dictionary::~Dictionary() {
	godot::api->godot_dictionary_destroy(&_godot_dictionary);
}

} // namespace godot
