#include "example_node.h"

String ExampleNode::hello() const {
#if GDEXTENSION
	return "Hello, World! From GDExtension.";
#elif GODOT_MODULE
	return "Hello, World! From a module.";
#endif
}

void ExampleNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("hello"), &ExampleNode::hello);
}
