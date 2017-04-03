#include "godot_dictionary.h"

#include "core/dictionary.h"

#include "core/os/memory.h"

#include "core/io/json.h"

#ifdef __cplusplus
extern "C" {
#endif

void _dictionary_api_anchor() {
}

void GDAPI godot_dictionary_new(godot_dictionary *p_dict) {
	Dictionary *dict = (Dictionary *)p_dict;
	memnew_placement(dict, Dictionary);
}

void GDAPI godot_dictionary_clear(godot_dictionary *p_dict) {
	Dictionary *dict = (Dictionary *)p_dict;
	dict->clear();
}

godot_bool GDAPI godot_dictionary_empty(const godot_dictionary *p_dict) {
	const Dictionary *dict = (const Dictionary *)p_dict;
	return dict->empty();
}

void GDAPI godot_dictionary_erase(godot_dictionary *p_dict, const godot_variant *p_key) {
	Dictionary *dict = (Dictionary *)p_dict;
	Variant *key = (Variant *)p_key;
	dict->erase(*key);
}

godot_bool GDAPI godot_dictionary_has(const godot_dictionary *p_dict, const godot_variant *p_key) {
	const Dictionary *dict = (const Dictionary *)p_dict;
	const Variant *key = (const Variant *)p_key;
	return dict->has(*key);
}

godot_bool GDAPI godot_dictionary_has_all(const godot_dictionary *p_dict, const godot_array *p_keys) {
	const Dictionary *dict = (const Dictionary *)p_dict;
	const Array *keys = (const Array *)p_keys;
	return dict->has_all(*keys);
}

uint32_t GDAPI godot_dictionary_hash(const godot_dictionary *p_dict) {
	const Dictionary *dict = (const Dictionary *)p_dict;
	return dict->hash();
}

godot_array GDAPI godot_dictionary_keys(const godot_dictionary *p_dict) {
	godot_array a;
	godot_array_new(&a);
	const Dictionary *dict = (const Dictionary *)p_dict;
	Array *array = (Array *)&a;
	*array = dict->keys();
	return a;
}

godot_int GDAPI godot_dictionary_parse_json(godot_dictionary *p_dict, const godot_string *p_json) {
	Dictionary *dict = (Dictionary *)p_dict;
	const String *json = (const String *)p_json;
	Variant ret;
	int err_line;
	String err_str;
	int err = (int)JSON::parse(*json, ret, err_str, err_line);
	*dict = ret;
	return err;
}

godot_variant GDAPI *godot_dictionary_operator_index(godot_dictionary *p_dict, const godot_variant *p_key) {
	Dictionary *dict = (Dictionary *)p_dict;
	Variant *key = (Variant *)p_key;
	return (godot_variant *)&dict->operator[](*key);
}

godot_int GDAPI godot_dictionary_size(const godot_dictionary *p_dict) {
	const Dictionary *dict = (const Dictionary *)p_dict;
	return dict->size();
}

godot_string GDAPI godot_dictionary_to_json(const godot_dictionary *p_dict) {
	const Dictionary *dict = (const Dictionary *)p_dict;
	godot_string str;
	godot_string_new(&str);
	String *s = (String *)&str;
	*s = JSON::print(Variant(*dict));
	return str;
}

godot_array GDAPI godot_dictionary_values(const godot_dictionary *p_dict) {
	godot_array a;
	godot_array_new(&a);
	const Dictionary *dict = (const Dictionary *)p_dict;
	Array *array = (Array *)&a;
	*array = dict->values();
	return a;
}

void GDAPI godot_dictionary_destroy(godot_dictionary *p_dict) {
	((Dictionary *)p_dict)->~Dictionary();
}

#ifdef __cplusplus
}
#endif
