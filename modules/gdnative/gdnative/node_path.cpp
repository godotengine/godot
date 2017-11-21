/*************************************************************************/
/*  node_path.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "gdnative/node_path.h"

#include "core/node_path.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_node_path_new(godot_node_path *r_dest, const godot_string *p_from) {
	NodePath *dest = (NodePath *)r_dest;
	const String *from = (const String *)p_from;
	memnew_placement(dest, NodePath(*from));
}

void GDAPI godot_node_path_new_copy(godot_node_path *r_dest, const godot_node_path *p_src) {
	NodePath *dest = (NodePath *)r_dest;
	const NodePath *src = (const NodePath *)p_src;
	memnew_placement(dest, NodePath(*src));
}

void GDAPI godot_node_path_destroy(godot_node_path *p_self) {
	NodePath *self = (NodePath *)p_self;
	self->~NodePath();
}

godot_string GDAPI godot_node_path_as_string(const godot_node_path *p_self) {
	godot_string ret;
	const NodePath *self = (const NodePath *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_bool GDAPI godot_node_path_is_absolute(const godot_node_path *p_self) {
	const NodePath *self = (const NodePath *)p_self;
	return self->is_absolute();
}

godot_int GDAPI godot_node_path_get_name_count(const godot_node_path *p_self) {
	const NodePath *self = (const NodePath *)p_self;
	return self->get_name_count();
}

godot_string GDAPI godot_node_path_get_name(const godot_node_path *p_self, const godot_int p_idx) {
	godot_string dest;
	const NodePath *self = (const NodePath *)p_self;

	memnew_placement(&dest, String(self->get_name(p_idx)));
	return dest;
}

godot_int GDAPI godot_node_path_get_subname_count(const godot_node_path *p_self) {
	const NodePath *self = (const NodePath *)p_self;
	return self->get_subname_count();
}

godot_string GDAPI godot_node_path_get_subname(const godot_node_path *p_self, const godot_int p_idx) {
	godot_string dest;
	const NodePath *self = (const NodePath *)p_self;

	memnew_placement(&dest, String(self->get_subname(p_idx)));
	return dest;
}

godot_string GDAPI godot_node_path_get_concatenated_subnames(const godot_node_path *p_self) {
	godot_string dest;
	const NodePath *self = (const NodePath *)p_self;
	memnew_placement(&dest, String(self->get_concatenated_subnames()));
	return dest;
}

godot_bool GDAPI godot_node_path_is_empty(const godot_node_path *p_self) {
	const NodePath *self = (const NodePath *)p_self;
	return self->is_empty();
}

godot_bool GDAPI godot_node_path_operator_equal(const godot_node_path *p_self, const godot_node_path *p_b) {
	const NodePath *self = (const NodePath *)p_self;
	const NodePath *b = (const NodePath *)p_b;
	return *self == *b;
}

#ifdef __cplusplus
}
#endif
