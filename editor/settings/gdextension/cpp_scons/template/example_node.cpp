#include "example_node.h"

void ExampleNode::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
#if GDEXTENSION
			print_line("The example GDExtension node has been added to the scene tree and is now ready!");
#elif GODOT_MODULE
			print_line("The example engine module node has been added to the scene tree and is now ready!");
#else
#error "Must build as Godot GDExtension or Godot module."
#endif
		} break;
	}
}

void ExampleNode::print_hello() const {
#if GDEXTENSION
	print_line("Hello, World! From GDExtension.");
#elif GODOT_MODULE
	print_line("Hello, World! From a module.");
#endif
}

String ExampleNode::return_hello() const {
#if GDEXTENSION
	return "Hello, World! From GDExtension.";
#elif GODOT_MODULE
	return "Hello, World! From a module.";
#endif
}

void ExampleNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("print_hello"), &ExampleNode::print_hello);
	ClassDB::bind_method(D_METHOD("return_hello"), &ExampleNode::return_hello);
}
