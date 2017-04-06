#include "godot_string.h"

#include "string_db.h"
#include "ustring.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void _string_api_anchor() {
}

void GDAPI godot_string_new(godot_string *p_str) {
	String *p = (String *)p_str;
	memnew_placement(p, String);
	// *p = String(); // useless here
}

void GDAPI godot_string_new_data(godot_string *p_str, const char *p_contents, const int p_size) {
	String *p = (String *)p_str;
	memnew_placement(p, String);
	*p = String::utf8(p_contents, p_size);
}

void GDAPI godot_string_get_data(const godot_string *p_str, wchar_t *p_dest, int *p_size) {
	String *p = (String *)p_str;
	if (p_size != NULL) {
		*p_size = p->length();
	}
	if (p_dest != NULL) {
		memcpy(p_dest, p->ptr(), *p_size * sizeof(CharType));
	}
}

void GDAPI godot_string_copy_string(const godot_string *p_dest, const godot_string *p_src) {
	String *dest = (String *)p_dest;
	String *src = (String *)p_src;

	*dest = *src;
}

wchar_t GDAPI *godot_string_operator_index(godot_string *p_str, const godot_int p_idx) {
	String *s = (String *)p_str;
	return &(s->operator[](p_idx));
}

const wchar_t GDAPI *godot_string_c_str(const godot_string *p_str) {
	const String *s = (const String *)p_str;
	return s->c_str();
}

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_a, const godot_string *p_b) {
	String *a = (String *)p_a;
	String *b = (String *)p_b;
	return *a == *b;
}

godot_bool GDAPI godot_string_operator_less(const godot_string *p_a, const godot_string *p_b) {
	String *a = (String *)p_a;
	String *b = (String *)p_b;
	return *a < *b;
}

void GDAPI godot_string_operator_plus(godot_string *p_dest, const godot_string *p_a, const godot_string *p_b) {
	String *dest = (String *)p_dest;
	const String *a = (String *)p_a;
	const String *b = (String *)p_b;

	String tmp = *a + *b;
	godot_string_new(p_dest);
	*dest = tmp;
}

void GDAPI godot_string_destroy(godot_string *p_str) {
	String *p = (String *)p_str;
	p->~String();
}

#ifdef __cplusplus
}
#endif
