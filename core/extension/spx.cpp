/**************************************************************************/
/*  spx.cpp                                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "spx.h"
#include "gdextension_spx_ext.h"
#include "scene/main/node.h"
#include "scene/main/window.h"
#include "spx_engine.h"
#include "spx_input_proxy.h"
#include "spx_sprite.h"
#include "spx_ui.h"

#define SPX_ENGINE SpxEngine::get_singleton()
bool Spx::initialed = false;
bool Spx::debug_mode = false;

void Spx::register_types() {
	ClassDB::register_class<SpxSprite>();
	ClassDB::register_class<SpxInputProxy>();
}

void Spx::on_start(void *p_tree) {
	print_verbose("Spx::on_start");
	initialed = true;
	if (!SpxEngine::has_initialed()) {
		return;
	}
	auto tree = (SceneTree *)p_tree;
	if (tree == nullptr)
		return;
	Window *root = tree->get_root();
	if (root == nullptr) {
		return;
	}

	Node *new_node = memnew(Node);
	new_node->set_name("SpxEngineNode");
	root->add_child(new_node);
	SPX_ENGINE->set_root_node(tree, new_node);
	SPX_ENGINE->on_awake();
}

void Spx::on_fixed_update(double delta) {
	if (!initialed) {
		return;
	}
	if (!SpxEngine::has_initialed()) {
		return;
	}
	SPX_ENGINE->on_fixed_update(delta);
}

void Spx::on_update(double delta) {
	if (!initialed) {
		return;
	}
	if (!SpxEngine::has_initialed()) {
		return;
	}
	SPX_ENGINE->on_update(delta);
}

void Spx::on_destroy() {
	if (!initialed) {
		return;
	}
	if (!SpxEngine::has_initialed()) {
		return;
	}
	print_verbose("Spx::on_destroy");
	SPX_ENGINE->on_destroy();
	initialed = false;
}
