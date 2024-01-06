#ifndef UTIL_H
#define UTIL_H

#ifndef LAPI_GDEXTENSION
#include <core/string/print_string.h>
#else
#include <godot_cpp/variant/utility_functions.hpp>
using namespace godot;

inline void print_line(const Variant &v) {
	UtilityFunctions::print(v);
}
#endif
#endif
