#include "api.hpp"

static Variant my_function(Vector4 v) {
	print("Arg: ", v);
	return 123;
}

static Variant my_function2(String s, Array a) {
	print("Args: ", s, a);
	return s;
}

static Variant _ready() {
	print("_ready called!");
	return Nil;
}

static Variant _process() {
	static int counter = 0;
	if (++counter % 100 == 0) {
		print("Process called " + std::to_string(counter) + " times");
	}
	return Nil;
}

static Vector4 my_vector4(1.0f, 2.0f, 3.0f, 4.0f);
static String my_string("Hello, World!");
int main() {
	ADD_PROPERTY(my_vector4, Variant::VECTOR4);
	ADD_PROPERTY(my_string, Variant::STRING);

	ADD_API_FUNCTION(my_function, "int", "Vector4 v");
	ADD_API_FUNCTION(my_function2, "Dictionary", "String s, Array a");

	ADD_API_FUNCTION(_ready, "void");
	ADD_API_FUNCTION(_process, "void");
}
