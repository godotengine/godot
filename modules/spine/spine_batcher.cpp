/*************************************************************************/
/*  register_types.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifdef MODULE_SPINE_ENABLED
#include "spine_batcher.h"

#define BATCH_CAPACITY 1024

void SpineBatcher::add(Ref<Texture> p_texture,
	const float* p_vertices, const float* p_uvs, int p_vertices_count,
	const int* p_indies, int p_indies_count,
	Color *p_color, bool flip_x, bool flip_y) {

	if (p_texture != texture
		|| vertices_count + (p_vertices_count >> 1) > BATCH_CAPACITY
		|| indies_count + p_indies_count > BATCH_CAPACITY * 3) {
	
		flush();
		texture = p_texture;
	}

	for (int i = 0; i < p_indies_count; ++i, ++indies_count)
		elements.indies[indies_count] = p_indies[i] + vertices_count;

	for (int i = 0; i < p_vertices_count; i += 2, ++vertices_count) {

		elements.vertices[vertices_count].x = flip_x ? -p_vertices[i] : p_vertices[i];
		elements.vertices[vertices_count].y = flip_y ? p_vertices[i + 1] : -p_vertices[i + 1];
		elements.colors[vertices_count] = *p_color;
		elements.uvs[vertices_count].x = p_uvs[i];
		elements.uvs[vertices_count].y = p_uvs[i + 1];
	}
}

void SpineBatcher::flush() {

	if (!vertices_count) return;

	RID ci = owner->get_canvas_item();
	VisualServer::get_singleton()->canvas_item_add_triangle_array_ptr(ci,
		indies_count / 3,
		elements.indies,
		elements.vertices,
		elements.colors,
		elements.uvs,
		texture->get_rid()
	);
	push_elements();

	vertices_count = 0;
	indies_count = 0;
}

void SpineBatcher::push_elements() {

	element_list.push_back(elements);

	elements.vertices = memnew_arr(Vector2, BATCH_CAPACITY);
	elements.colors = memnew_arr(Color, BATCH_CAPACITY);
	elements.uvs = memnew_arr(Vector2, BATCH_CAPACITY);
	elements.indies = memnew_arr(int, BATCH_CAPACITY * 3);
}

void SpineBatcher::reset() {

	for (List<Elements>::Element *E = element_list.front(); E; E = E->next()) {

		Elements& e = E->get();
		memdelete_arr(e.vertices);
		memdelete_arr(e.colors);
		memdelete_arr(e.uvs);
		memdelete_arr(e.indies);
	}
	element_list.clear();
}

SpineBatcher::SpineBatcher(Node2D *owner) : owner(owner) {

	vertices_count = 0;
	indies_count = 0;
	elements.vertices = memnew_arr(Vector2, BATCH_CAPACITY);
	elements.colors = memnew_arr(Color, BATCH_CAPACITY);
	elements.uvs = memnew_arr(Vector2, BATCH_CAPACITY);
	elements.indies = memnew_arr(int, BATCH_CAPACITY * 3);
}

SpineBatcher::~SpineBatcher() {

	reset();

	memdelete_arr(elements.vertices);
	memdelete_arr(elements.colors);
	memdelete_arr(elements.uvs);
	memdelete_arr(elements.indies);
}

#endif // MODULE_SPINE_ENABLED

