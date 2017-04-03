#ifndef GODOT_DLSCRIPT_DICTIONARY_H
#define GODOT_DLSCRIPT_DICTIONARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_DICITIONARY_TYPE_DEFINED
typedef struct godot_dictionary {
	uint8_t _dont_touch_that[8];
} godot_dictionary;
#endif

#include "godot_array.h"
#include "godot_variant.h"

void GDAPI godot_dictionary_new(godot_dictionary *p_dict);

void GDAPI godot_dictionary_clear(godot_dictionary *p_dict);

godot_bool GDAPI godot_dictionary_empty(const godot_dictionary *p_dict);

void GDAPI godot_dictionary_erase(godot_dictionary *p_dict, const godot_variant *p_key);

godot_bool GDAPI godot_dictionary_has(const godot_dictionary *p_dict, const godot_variant *p_key);

godot_bool GDAPI godot_dictionary_has_all(const godot_dictionary *p_dict, const godot_array *p_keys);

uint32_t GDAPI godot_dictionary_hash(const godot_dictionary *p_dict);

godot_array GDAPI godot_dictionary_keys(const godot_dictionary *p_dict);

godot_int GDAPI godot_dictionary_parse_json(godot_dictionary *p_dict, const godot_string *p_json);

godot_variant GDAPI *godot_dictionary_operator_index(godot_dictionary *p_dict, const godot_variant *p_key);

godot_int GDAPI godot_dictionary_size(const godot_dictionary *p_dict);

godot_string GDAPI godot_dictionary_to_json(const godot_dictionary *p_dict);

godot_array GDAPI godot_dictionary_values(const godot_dictionary *p_dict);

void GDAPI godot_dictionary_destroy(godot_dictionary *p_dict);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_DICTIONARY_H
