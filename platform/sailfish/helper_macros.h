#ifndef SAILFISH_MACRO
#define SAILFISH_MACRO

// print verbose macros (more usefull than print_verbose funtion, from print_string.h)
#define mprint_verbose2(x, ...)                     \
	if (OS::get_singleton()->is_stdout_verbose()) { \
		OS::get_singleton()->print(x, __VA_ARGS__); \
	}

#define mprint_verbose(x)                           \
	if (OS::get_singleton()->is_stdout_verbose()) { \
		OS::get_singleton()->print(x);              \
	}
#endif