#pragma once

#include "sandbox.h"
#include <godot_cpp/classes/node.hpp>

struct ScopedTreeBase {
	Sandbox *sandbox = nullptr;
	godot::Node *old_tree_base = nullptr;

	ScopedTreeBase(Sandbox *sandbox, godot::Node *old_tree_base) :
			sandbox(sandbox),
			old_tree_base(sandbox->get_tree_base()) {
		sandbox->set_tree_base(old_tree_base);
	}

	~ScopedTreeBase() {
		sandbox->set_tree_base(old_tree_base);
	}
};
