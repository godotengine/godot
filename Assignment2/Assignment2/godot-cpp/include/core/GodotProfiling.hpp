#ifndef GODOT_PROFILING_HPP
#define GODOT_PROFILING_HPP

#include "OS.hpp"


namespace godot {

class FunctionProfiling {
	char signature[1024];
	uint64_t ticks;

public:
	FunctionProfiling(const char *p_function, const int p_line) {
		snprintf(signature, 1024, "::%d::%s", p_line, p_function);
		ticks = OS::get_singleton()->get_ticks_usec();
	}
	~FunctionProfiling() {
		uint64_t t = OS::get_singleton()->get_ticks_usec() - ticks;
		if (t > 0) {
			Godot::gdnative_profiling_add_data(signature, t);
		}
	}
};

}

#ifdef DEBUG_ENABLED
#define GODOT_PROFILING_FUNCTION FunctionProfiling __function_profiling(__FUNCTION__, __LINE__);
#else
#define GODOT_PROFILING_FUNCTION
#endif

#endif
