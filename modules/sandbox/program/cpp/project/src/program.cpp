#include <api.hpp>

static String last_string = "Hello, world!";
static Variant last_printed_string() {
	return last_string;
}
static Variant print_string(String str) {
	printf("String: %s\n", str.utf8().c_str());
	fflush(stdout);
	last_string = str;
	return Nil;
}

static long fib(long n, long acc, long prev) {
	if (n == 0)
		return acc;
	else
		return fib(n - 1, prev + acc, acc);
}

static Variant fibonacci(int n) {
	return fib(n, 0, 1);
}

int main() {
	print("Hello, world!");

	// The entire Godot API is available
	Sandbox sandbox = get_node<Sandbox>();
	print(sandbox.is_binary_translated()
		? "The current program is accelerated by binary translation."
		: "The current program is running in interpreter mode.");

	// Add public API
	ADD_API_FUNCTION(last_printed_string, "String", "", "Returns the last printed string");
	ADD_API_FUNCTION(print_string, "void", "String str", "Prints a string to the console");
	ADD_API_FUNCTION(fibonacci, "long", "int n", "Calculates the nth Fibonacci number");

	// Add a sandboxed property
	static int meaning_of_life = 42;
	add_property("meaning_of_life", Variant::Type::INT, 42,
		[]() -> Variant { return meaning_of_life; },
		[](Variant value) -> Variant { meaning_of_life = value; print("Set to: ", meaning_of_life); return Nil; });

	halt();
}
