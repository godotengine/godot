#pragma once

#include "__LIBRARY_NAME___defines.h"

#if GDEXTENSION
#include <godot_cpp/classes/node.hpp>
#elif GODOT_MODULE
#include "scene/main/node.h"
#endif

class ExampleNode : public Node {
	GDCLASS(ExampleNode, Node);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void print_hello() const;
	String return_hello() const;
};
