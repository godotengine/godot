/**************************************************************************/
/*  node.cpp                                                              */
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
#pragma once

#include "core/object/object.h"
#include "core/os/memory.h"
#include "node.h"
#include "fragment.h"

void Fragment::attach_to(Node *p_host, int p_child_index, bool p_include_internal) {
	set_host(p_host, p_child_index, p_include_internal);

	// _update_children_cache();

	// TODO: add children
	// int start_index = p_host;
	// if (!p_include_internal) {
	// 	return data.index;
	// } else {
	// 	switch (data.internal_mode) {
	// 		case INTERNAL_MODE_DISABLED: {
	// 			return data.parent->data.internal_children_front_count_cache + data.index;
	// 		} break;
	// 		case INTERNAL_MODE_FRONT: {
	// 			return data.index;
	// 		} break;
	// 		case INTERNAL_MODE_BACK: {
	// 			return data.parent->data.internal_children_front_count_cache + data.parent->data.external_children_count_cache + data.index;
	// 		} break;
	// 	}
	// 	return -1;
	// }
	// for (Node *child : data.children_cache) {
	// 	child->append_to(p_host);
	// 	child->move_to(p_child_index);
	// }
}

void Fragment::set_host(Node *p_node, int p_child_index, bool p_include_internal) {
	host = p_node;

	if (p_child_index < 0) {
		p_child_index += p_node->get_child_count(p_include_internal);
	}
	// anchor = memnew(Node::Anchor{p_child_index});
	// p_node->put_anchor(anchor);
}

Fragment::Fragment(){
	data.is_fragment = true;
}

Fragment::~Fragment(){

}
