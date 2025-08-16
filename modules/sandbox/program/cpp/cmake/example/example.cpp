#include <fmt/format.h>
#include <api.hpp>

extern "C" Variant test_function() {
	fmt::print("Hello, World!\n");
	return 42;
}
