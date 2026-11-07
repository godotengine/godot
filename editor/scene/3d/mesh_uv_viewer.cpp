/**************************************************************************/
/*  mesh_uv_viewer.cpp                                                    */
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

#include "mesh_uv_viewer.h"

#include "core/object/callable_mp.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/tab_container.h"

bool MeshUvViewer::setup_mesh(const Ref<Mesh> &p_mesh) {
	_clear_all();
	if (p_mesh.is_null()) {
		return false;
	}

	bool has_uv1 = false;
	bool has_uv2 = false;
	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		Array a = p_mesh->surface_get_arrays(i);
		Vector<Vector2> uv1 = a[Mesh::ARRAY_TEX_UV];
		Vector<Vector2> uv2 = a[Mesh::ARRAY_TEX_UV2];
		has_uv1 = has_uv1 || !uv1.is_empty();
		has_uv2 = has_uv2 || !uv2.is_empty();
		if (has_uv1 && has_uv2) {
			break;
		}
	}

	const bool has_uvs = has_uv1 || has_uv2;
	if (!has_uvs) {
		return false;
	}

	tabs->set_tab_hidden(0, !has_uv1);
	tabs->set_tab_hidden(1, !has_uv2);
	tabs->set_current_tab(has_uv1 ? 0 : 1);

	if (has_uv1) {
		_create_uv_lines(0, p_mesh);
	}
	if (has_uv2) {
		_create_uv_lines(1, p_mesh);
	}

	return true;
}

void MeshUvViewer::set_view_min_size(float p_min_size) {
	for (int i = 0; i < 2; i++) {
		uv_control[i]->set_custom_minimum_size(Size2(p_min_size, p_min_size) * EDSCALE);
	}
}

void MeshUvViewer::_clear_all() {
	uv_lines[0].clear();
	uv_lines[1].clear();
	tabs->set_tab_hidden(0, true);
	tabs->set_tab_hidden(1, true);
}

void MeshUvViewer::_create_uv_lines(int p_layer, const Ref<Mesh> &p_mesh) {
	ERR_FAIL_COND(p_mesh.is_null());

	struct MeshEdgeSort {
		Vector2 a;
		Vector2 b;

		static uint32_t hash(const MeshEdgeSort &p_edge) {
			uint32_t h = hash_murmur3_one_32(HashMapHasherDefault::hash(p_edge.a));
			return hash_fmix32(hash_murmur3_one_32(HashMapHasherDefault::hash(p_edge.b), h));
		}

		bool operator==(const MeshEdgeSort &p_b) const {
			return a == p_b.a && b == p_b.b;
		}

		MeshEdgeSort() {}
		MeshEdgeSort(const Vector2 &p_a, const Vector2 &p_b) {
			if (p_a < p_b) {
				a = p_a;
				b = p_b;
			} else {
				b = p_a;
				a = p_b;
			}
		}
	};

	HashSet<MeshEdgeSort, MeshEdgeSort> edges;
	uv_lines[p_layer].clear();
	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}
		Array a = p_mesh->surface_get_arrays(i);

		Vector<Vector2> uv = a[p_layer == 0 ? Mesh::ARRAY_TEX_UV : Mesh::ARRAY_TEX_UV2];
		if (uv.is_empty()) {
			continue;
		}

		const Vector2 *r = uv.ptr();

		Vector<int> indices = a[Mesh::ARRAY_INDEX];
		const int *ri = nullptr;

		int ic;

		if (indices.size()) {
			ic = indices.size();
			ri = indices.ptr();
		} else {
			ic = uv.size();
		}

		for (int j = 0; j < ic; j += 3) {
			for (int k = 0; k < 3; k++) {
				MeshEdgeSort edge;
				if (ri) {
					edge.a = r[ri[j + k]];
					edge.b = r[ri[j + ((k + 1) % 3)]];
				} else {
					edge.a = r[j + k];
					edge.b = r[j + ((k + 1) % 3)];
				}

				if (edges.has(edge)) {
					continue;
				}

				uv_lines[p_layer].push_back(edge.a);
				uv_lines[p_layer].push_back(edge.b);
				edges.insert(edge);
			}
		}
	}
}

void MeshUvViewer::_uv_draw(int p_layer) {
	if (uv_lines[p_layer].is_empty()) {
		return;
	}

	uv_control[p_layer]->set_clip_contents(true);
	uv_control[p_layer]->draw_rect(
			Rect2(Vector2(), uv_control[p_layer]->get_size()),
			get_theme_color(SNAME("dark_color_3"), EditorStringName(Editor)));

	// Draw an outline to represent the UV2's beginning and end area (useful on Black OLED theme).
	// Top-left coordinate needs to be `(1, 1)` to prevent `clip_contents` from clipping the top and left lines.
	uv_control[p_layer]->draw_rect(
			Rect2(Vector2(1, 1), uv_control[p_layer]->get_size() - Vector2(1, 1)),
			get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.125),
			false,
			Math::round(EDSCALE));

	for (int x = 1; x <= 7; x++) {
		uv_control[p_layer]->draw_line(
				Vector2(uv_control[p_layer]->get_size().x * 0.125 * x, 0),
				Vector2(uv_control[p_layer]->get_size().x * 0.125 * x, uv_control[p_layer]->get_size().y),
				get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.125),
				Math::round(EDSCALE));
	}

	for (int y = 1; y <= 7; y++) {
		uv_control[p_layer]->draw_line(
				Vector2(0, uv_control[p_layer]->get_size().y * 0.125 * y),
				Vector2(uv_control[p_layer]->get_size().x, uv_control[p_layer]->get_size().y * 0.125 * y),
				get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.125),
				Math::round(EDSCALE));
	}

	uv_control[p_layer]->draw_set_transform(Vector2(), 0, uv_control[p_layer]->get_size());

	// Use a translucent color to allow overlapping triangles to be visible.
	// Divide line width by the drawing scale set above, so that line width is consistent regardless of dialog size.
	// Aspect ratio is preserved by the parent AspectRatioContainer, so we only need to check the X size which is always equal to Y.
	uv_control[p_layer]->draw_multiline(
			uv_lines[p_layer],
			get_theme_color(SNAME("mono_color"), EditorStringName(Editor)) * Color(1, 1, 1, 0.5),
			Math::round(EDSCALE) / uv_control[p_layer]->get_size().x);
}

MeshUvViewer::MeshUvViewer() {
	tabs = memnew(TabContainer);
	add_child(tabs);
	for (int i = 0; i < 2; i++) {
		AspectRatioContainer *arc = memnew(AspectRatioContainer);
		arc->set_name(i == 0 ? "UV" : "UV2");
		tabs->add_child(arc);

		uv_control[i] = memnew(Control);
		uv_control[i]->set_custom_minimum_size(Size2(300, 300) * EDSCALE);
		uv_control[i]->connect(SceneStringName(draw), callable_mp(this, &MeshUvViewer::_uv_draw).bind(i));
		arc->add_child(uv_control[i]);
	}
}
