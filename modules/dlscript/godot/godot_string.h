#ifndef GODOT_DLSCRIPT_STRING_H
#define GODOT_DLSCRIPT_STRING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef GODOT_CORE_API_GODOT_STRING_TYPE_DEFINED
typedef struct godot_string {
	uint8_t _dont_touch_that[8];
} godot_string;
#endif

#include "../godot.h"

void GDAPI godot_string_new(godot_string *p_str);
void GDAPI godot_string_new_data(godot_string *p_str, const char *p_contents, const int p_size);

void GDAPI godot_string_get_data(const godot_string *p_str, wchar_t *p_dest, int *p_size);

void GDAPI godot_string_copy_string(const godot_string *p_dest, const godot_string *p_src);

wchar_t GDAPI *godot_string_operator_index(godot_string *p_str, const godot_int p_idx);
const wchar_t GDAPI *godot_string_c_str(const godot_string *p_str);

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_a, const godot_string *p_b);
godot_bool GDAPI godot_string_operator_less(const godot_string *p_a, const godot_string *p_b);
void GDAPI godot_string_operator_plus(godot_string *p_dest, const godot_string *p_a, const godot_string *p_b);

// @Incomplete
// hmm, I guess exposing the whole API doesn't make much sense
// since the language used in the library has its own string funcs

void GDAPI godot_string_destroy(godot_string *p_str);

#ifdef __cplusplus
}
#endif

#endif // GODOT_DLSCRIPT_STRING_H
