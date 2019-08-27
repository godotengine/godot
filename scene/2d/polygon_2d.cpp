/*************************************************************************/
/*  polygon_2d.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "polygon_2d.h"

#include "core/math/geometry.h"
#include "skeleton_2d.h"

Dictionary Polygon2D::_edit_get_state() const {
	Dictionary state = Node2D::_edit_get_state();
	state["offset"] = offset;
	return state;
}

void Polygon2D::_edit_set_state(const Dictionary &p_state) {
	Node2D::_edit_set_state(p_state);
	set_offset(p_state["offset"]);
}

void Polygon2D::_edit_set_pivot(const Point2 &p_pivot) {
	set_position(get_transform().xform(p_pivot));
	set_offset(get_offset() - p_pivot);
}

Point2 Polygon2D::_edit_get_pivot() const {
	return Vector2();
}

bool Polygon2D::_edit_use_pivot() const {
	return true;
}

Rect2 Polygon2D::_edit_get_rect() const {
	if (rect_cache_dirty) {
		int l = polygon.size();
		PoolVector<Vector2>::Read r = polygon.read();
		item_rect = Rect2();
		for (int i = 0; i < l; i++) {
			Vector2 pos = r[i] + offset;
			if (i == 0)
				item_rect.position = pos;
			else
				item_rect.expand_to(pos);
		}
		rect_cache_dirty = false;
	}

	return item_rect;
}

bool Polygon2D::_edit_use_rect() const {
	return polygon.size() > 0;
}

bool Polygon2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {

	Vector<Vector2> polygon2d = Variant(polygon);
	if (internal_vertices > 0) {
		polygon2d.resize(polygon2d.size() - internal_vertices);
	}
	return Geometry::is_point_in_polygon(p_point - get_offset(), polygon2d);
}

void Polygon2D::_skeleton_bone_setup_changed() {
	update();
}

void Polygon2D::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_DRAW: {

			if (polygon.size() < 3)
				return;

			Skeleton2D *skeleton_node = NULL;
			if (has_node(skeleton)) {
				skeleton_node = Object::cast_to<Skeleton2D>(get_node(skeleton));
			}

			ObjectID new_skeleton_id = 0;

			if (skeleton_node) {
				VS::get_singleton()->canvas_item_attach_skeleton(get_canvas_item(), skeleton_node->get_skeleton());
				new_skeleton_id = skeleton_node->get_instance_id();
			} else {
				VS::get_singleton()->canvas_item_attach_skeleton(get_canvas_item(), RID());
			}

			if (new_skeleton_id != current_skeleton_id) {
				Object *old_skeleton = ObjectDB::get_instance(current_skeleton_id);
				if (old_skeleton) {
					old_skeleton->disconnect("bone_setup_changed", this, "_skeleton_bone_setup_changed");
				}

				if (skeleton_node) {
					skeleton_node->connect("bone_setup_changed", this, "_skeleton_bone_setup_changed");
				}

				current_skeleton_id = new_skeleton_id;
			}

			Vector<Vector2> points;
			Vector<Vector2> uvs;
			Vector<int> bones;
			Vector<float> weights;

			int len = polygon.size();
			if ((invert || polygons.size() == 0) && internal_vertices > 0) {
				//if no polygons are around, internal vertices must not be drawn, else let them be
				len -= internal_vertices;
			}

			if (len <= 0) {
				return;
			}
			points.resize(len);

			{

				PoolVector<Vector2>::Read polyr = polygon.read();
				for (int i = 0; i < len; i++) {
					points.write[i] = polyr[i] + offset;
				}
			}

			if (invert) {

				Rect2 bounds;
				int highest_idx = -1;
				float highest_y = -1e20;
				float sum = 0;

				for (int i = 0; i < len; i++) {
					if (i == 0)
						bounds.position = points[i];
					else
						bounds.expand_to(points[i]);
					if (points[i].y > highest_y) {
						highest_idx = i;
						highest_y = points[i].y;
					}
					int ni = (i + 1) % len;
					sum += (points[ni].x - points[i].x) * (points[ni].y + points[i].y);
				}

				bounds = bounds.grow(invert_border);

				Vector2 ep[7] = {
					Vector2(points[highest_idx].x, points[highest_idx].y + invert_border),
					Vector2(bounds.position + bounds.size),
					Vector2(bounds.position + Vector2(bounds.size.x, 0)),
					Vector2(bounds.position),
					Vector2(bounds.position + Vector2(0, bounds.size.y)),
					Vector2(points[highest_idx].x - CMP_EPSILON, points[highest_idx].y + invert_border),
					Vector2(points[highest_idx].x - CMP_EPSILON, points[highest_idx].y),
				};

				if (sum > 0) {
					SWAP(ep[1], ep[4]);
					SWAP(ep[2], ep[3]);
					SWAP(ep[5], ep[0]);
					SWAP(ep[6], points.write[highest_idx]);
				}

				points.resize(points.size() + 7);
				for (int i = points.size() - 1; i >= highest_idx + 7; i--) {

					points.write[i] = points[i - 7];
				}

				for (int i = 0; i < 7; i++) {

					points.write[highest_idx + i + 1] = ep[i];
				}

				len = points.size();
			}

			if (texture.is_valid()) {

				Transform2D texmat(tex_rot, tex_ofs);
				texmat.scale(tex_scale);
				Size2 tex_size = texture->get_size();

				uvs.resize(len);

				if (points.size() == uv.size()) {

					PoolVector<Vector2>::Read uvr = uv.read();

					for (int i = 0; i < len; i++) {
						uvs.write[i] = texmat.xform(uvr[i]) / tex_size;
					}

				} else {
					for (int i = 0; i < len; i++) {
						uvs.write[i] = texmat.xform(points[i]) / tex_size;
					}
				}
			}

			if (skeleton_node && !invert && bone_weights.size()) {
				//a skeleton is set! fill indices and weights
				int vc = len;
				bones.resize(vc * 4);
				weights.resize(vc * 4);

				int *bonesw = bones.ptrw();
				float *weightsw = weights.ptrw();

				for (int i = 0; i < vc * 4; i++) {
					bonesw[i] = 0;
					weightsw[i] = 0;
				}

				for (int i = 0; i < bone_weights.size(); i++) {
					if (bone_weights[i].weights.size() != points.size()) {
						continue; //different number of vertices, sorry not using.
					}
					if (!skeleton_node->has_node(bone_weights[i].path)) {
						continue; //node does not exist
					}
					Bone2D *bone = Object::cast_to<Bone2D>(skeleton_node->get_node(bone_weights[i].path));
					if (!bone) {
						continue;
					}

					int bone_index = bone->get_index_in_skeleton();
					PoolVector<float>::Read r = bone_weights[i].weights.read();
					for (int j = 0; j < vc; j++) {
						if (r[j] == 0.0)
							continue; //weight is unpainted, skip
						//find an index with a weight
						for (int k = 0; k < 4; k++) {
							if (weightsw[j * 4 + k] < r[j]) {
								//this is less than this weight, insert weight!
								for (int l = 3; l > k; l--) {
									weightsw[j * 4 + l] = weightsw[j * 4 + l - 1];
									bonesw[j * 4 + l] = bonesw[j * 4 + l - 1];
								}
								weightsw[j * 4 + k] = r[j];
								bonesw[j * 4 + k] = bone_index;
								break;
							}
						}
					}
				}

				//normalize the weights
				for (int i = 0; i < vc; i++) {
					float tw = 0;
					for (int j = 0; j < 4; j++) {
						tw += weightsw[i * 4 + j];
					}
					if (tw == 0)
						continue; //unpainted, do nothing

					//normalize
					for (int j = 0; j < 4; j++) {
						weightsw[i * 4 + j] /= tw;
					}
				}
			}

			Vector<Color> colors;
			if (vertex_colors.size() == points.size()) {
				colors.resize(len);
				PoolVector<Color>::Read color_r = vertex_colors.read();
				for (int i = 0; i < len; i++) {
					colors.write[i] = color_r[i];
				}
			} else {
				colors.push_back(color);
			}

			//			Vector<int> indices = Geometry::triangulate_polygon(points);
			//			VS::get_singleton()->canvas_item_add_triangle_array(get_canvas_item(), indices, points, colors, uvs, texture.is_valid() ? texture->get_rid() : RID());

			if (invert || polygons.size() == 0) {
				Vector<int> indices = Geometry::triangulate_polygon(points);
				if (indices.size()) {
					VS::get_singleton()->canvas_item_add_triangle_array(get_canvas_item(), indices, points, colors, uvs, bones, weights, texture.is_valid() ? texture->get_rid() : RID());
				}
			} else {
				//draw individual polygons
				Vector<int> total_indices;
				for (int i = 0; i < polygons.size(); i++) {
					PoolVector<int> src_indices = polygons[i];
					int ic = src_indices.size();
					if (ic < 3)
						continue;
					PoolVector<int>::Read r = src_indices.read();

					Vector<Vector2> tmp_points;
					tmp_points.resize(ic);

					for (int j = 0; j < ic; j++) {
						int idx = r[j];
						ERR_CONTINUE(idx < 0 || idx >= points.size());
						tmp_points.write[j] = points[r[j]];
					}
					Vector<int> indices = Geometry::triangulate_polygon(tmp_points);
					int ic2 = indices.size();
					const int *r2 = indices.ptr();

					int bic = total_indices.size();
					total_indices.resize(bic + ic2);
					int *w2 = total_indices.ptrw();

					for (int j = 0; j < ic2; j++) {
						w2[j + bic] = r[r2[j]];
					}
				}

				if (total_indices.size()) {
					VS::get_singleton()->canvas_item_add_triangle_array(get_canvas_item(), total_indices, points, colors, uvs, bones, weights, texture.is_valid() ? texture->get_rid() : RID());
				}

#if 0
				//use splits
				Vector<int> loop;
				int sc = splits.size();
				PoolVector<int>::Read r = splits.read();


				print_line("has splits, amount " + itos(splits.size()));
				Vector<Vector<int> > loops;

				// find a point that can be used to begin, must not be in a split, and have to the left and right the same one
				// like this one -> x---o
				//                   \ / \ .
				//                    o---o
				int base_point = -1;
				{
					int current_point = -1;
					int base_point_prev_split = -1;


					for (int i = 0; i < points.size(); i++) {

						//find if this point is in a split
						int split_index = -1;
						bool has_prev_split = false;
						int min_dist_to_end = 0x7FFFFFFF;

						for (int j = 0; j < sc; j += 2) {

							int split_pos = -1;
							int split_end = -1;

							if (r[j + 0] == i) { //found split in first point
								split_pos = r[j + 0];
								split_end = r[j + 1];
							} else if (r[j + 1] == i) { //found split in second point
								split_pos = r[j + 1];
								split_end = r[j + 0];
							}

							if (split_pos == split_end) {
								continue; //either nothing found or begin == end, this not a split in either case
							}

							if (j == base_point_prev_split) {
								has_prev_split = true;
							}

							//compute distance from split pos to split end in current traversal direction
							int dist_to_end = split_end > split_pos ? split_end - split_pos : (last - split_pos + split_end);

							if (dist_to_end < min_dist_to_end) {
								//always keep the valid split with the least distance to the loop
								min_dist_to_end = dist_to_end;
								split_index = j;
							}
						}

						if (split_index == -1) {
							current_point = i; //no split here, we are testing this point
						} else if (has_prev_split) {
							base_point = current_point; // there is a split and it contains the previous visited split, success
							break;
						} else {
							//invalidate current point and keep split
							current_point = -1;
							base_point_prev_split = split_index;
						}
					}
				}

				print_line("found base point: " + itos(base_point));

				if (base_point != -1) {

					int point = base_point;
					int last = base_point;
					//go through all the points, find splits
					do {

						int split;
						int last_dist_to_end = -1; //maximum valid distance to end

						do {

							loop.push_back(point); //push current point

							split = -1;
							int end = -1;

							int max_dist_to_end = 0;

							//find if this point is in a split
							for (int j = 0; j < sc; j += 2) {

								int split_pos = -1;
								int split_end = -1;

								if (r[j + 0] == point) { //match first split index
									split_pos = r[j + 0];
									split_end = r[j + 1];
								} else if (r[j + 1] == point) { //match second split index
									split_pos = r[j + 1];
									split_end = r[j + 0];
								}

								if (split_pos == split_end) {
									continue; //either nothing found or begin == end, this not a split in either case
								}

								//compute distance from split pos to split end
								int dist_to_end = split_end > split_pos ? split_end - split_pos : (points.size() - split_pos + split_end);

								if (last_dist_to_end != -1 && dist_to_end >= last_dist_to_end) {
									//distance must be shorter than in last iteration, means we've tested this before so ignore
									continue;
								} else if (dist_to_end > max_dist_to_end) {
									//always keep the valid point with the most distance (as long as it's valid)
									max_dist_to_end = dist_to_end;
									split = split_pos;
									end = split_end;
								}
							}

							if (split != -1) {
								//found a split!
								int from = end;

								//add points until last is reached
								while (true) {
									//find if point is in a split
									loop.push_back(from);

									if (from == last) {
										break;
									}

									from++;
									if (from >= points.size()) { //wrap if reached end
										from = 0;
									}

									if (from == loop[0]) {
										break; //end because we reached split source
									}
								}

								loops.push_back(loop); //done with this loop
								loop.clear();

								last_dist_to_end = max_dist_to_end;
								last = end; //algorithm can safely finish in this split point
							}

						} while (split != -1);

					} while (point != last);
				}

				if (loop.size() >=2 ) { //points remained
					//points remain
					loop.push_back(last); //no splits found, use last
					loops.push_back(loop);
				}

				print_line("total loops: " + itos(loops.size()));

				if (loops.size()) { //loops found
					Vector<int> indices;

					for (int i = 0; i < loops.size(); i++) {
						Vector<int> loop = loops[i];
						Vector<Vector2> vertices;
						vertices.resize(loop.size());
						for (int j = 0; j < vertices.size(); j++) {
							vertices.write[j] = points[loop[j]];
						}
						Vector<int> sub_indices = Geometry::triangulate_polygon(vertices);
						int from = indices.size();
						indices.resize(from + sub_indices.size());
						for (int j = 0; j < sub_indices.size(); j++) {
							indices.write[from + j] = loop[sub_indices[j]];
						}
					}

					VS::get_singleton()->canvas_item_add_triangle_array(get_canvas_item(), indices, points, colors, uvs, bones, weights, texture.is_valid() ? texture->get_rid() : RID());
				}
#endif
			}

		} break;
	}
}

void Polygon2D::set_polygon(const PoolVector<Vector2> &p_polygon) {
	polygon = p_polygon;
	rect_cache_dirty = true;
	update();
}

PoolVector<Vector2> Polygon2D::get_polygon() const {

	return polygon;
}

void Polygon2D::set_internal_vertex_count(int p_count) {

	internal_vertices = p_count;
}

int Polygon2D::get_internal_vertex_count() const {
	return internal_vertices;
}

void Polygon2D::set_uv(const PoolVector<Vector2> &p_uv) {

	uv = p_uv;
	update();
}

PoolVector<Vector2> Polygon2D::get_uv() const {

	return uv;
}

void Polygon2D::set_polygons(const Array &p_polygons) {

	polygons = p_polygons;
	update();
}

Array Polygon2D::get_polygons() const {

	return polygons;
}

void Polygon2D::set_color(const Color &p_color) {

	color = p_color;
	update();
}
Color Polygon2D::get_color() const {

	return color;
}

void Polygon2D::set_vertex_colors(const PoolVector<Color> &p_colors) {

	vertex_colors = p_colors;
	update();
}
PoolVector<Color> Polygon2D::get_vertex_colors() const {

	return vertex_colors;
}

void Polygon2D::set_texture(const Ref<Texture> &p_texture) {

	texture = p_texture;

	/*if (texture.is_valid()) {
		uint32_t flags=texture->get_flags();
		flags&=~Texture::FLAG_REPEAT;
		if (tex_tile)
			flags|=Texture::FLAG_REPEAT;

		texture->set_flags(flags);
	}*/
	update();
}
Ref<Texture> Polygon2D::get_texture() const {

	return texture;
}

void Polygon2D::set_texture_offset(const Vector2 &p_offset) {

	tex_ofs = p_offset;
	update();
}
Vector2 Polygon2D::get_texture_offset() const {

	return tex_ofs;
}

void Polygon2D::set_texture_rotation(float p_rot) {

	tex_rot = p_rot;
	update();
}
float Polygon2D::get_texture_rotation() const {

	return tex_rot;
}

void Polygon2D::set_texture_rotation_degrees(float p_rot) {

	set_texture_rotation(Math::deg2rad(p_rot));
}
float Polygon2D::get_texture_rotation_degrees() const {

	return Math::rad2deg(get_texture_rotation());
}

void Polygon2D::set_texture_scale(const Size2 &p_scale) {

	tex_scale = p_scale;
	update();
}
Size2 Polygon2D::get_texture_scale() const {

	return tex_scale;
}

void Polygon2D::set_invert(bool p_invert) {

	invert = p_invert;
	update();
}
bool Polygon2D::get_invert() const {

	return invert;
}

void Polygon2D::set_antialiased(bool p_antialiased) {

	antialiased = p_antialiased;
	update();
}
bool Polygon2D::get_antialiased() const {

	return antialiased;
}

void Polygon2D::set_invert_border(float p_invert_border) {

	invert_border = p_invert_border;
	update();
}
float Polygon2D::get_invert_border() const {

	return invert_border;
}

void Polygon2D::set_offset(const Vector2 &p_offset) {

	offset = p_offset;
	rect_cache_dirty = true;
	update();
	_change_notify("offset");
}

Vector2 Polygon2D::get_offset() const {

	return offset;
}

void Polygon2D::add_bone(const NodePath &p_path, const PoolVector<float> &p_weights) {

	Bone bone;
	bone.path = p_path;
	bone.weights = p_weights;
	bone_weights.push_back(bone);
}
int Polygon2D::get_bone_count() const {
	return bone_weights.size();
}
NodePath Polygon2D::get_bone_path(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, bone_weights.size(), NodePath());
	return bone_weights[p_index].path;
}
PoolVector<float> Polygon2D::get_bone_weights(int p_index) const {

	ERR_FAIL_INDEX_V(p_index, bone_weights.size(), PoolVector<float>());
	return bone_weights[p_index].weights;
}
void Polygon2D::erase_bone(int p_idx) {

	ERR_FAIL_INDEX(p_idx, bone_weights.size());
	bone_weights.remove(p_idx);
}

void Polygon2D::clear_bones() {
	bone_weights.clear();
}

void Polygon2D::set_bone_weights(int p_index, const PoolVector<float> &p_weights) {
	ERR_FAIL_INDEX(p_index, bone_weights.size());
	bone_weights.write[p_index].weights = p_weights;
	update();
}
void Polygon2D::set_bone_path(int p_index, const NodePath &p_path) {
	ERR_FAIL_INDEX(p_index, bone_weights.size());
	bone_weights.write[p_index].path = p_path;
	update();
}

Array Polygon2D::_get_bones() const {
	Array bones;
	for (int i = 0; i < get_bone_count(); i++) {
		bones.push_back(get_bone_path(i));
		bones.push_back(get_bone_weights(i));
	}
	return bones;
}
void Polygon2D::_set_bones(const Array &p_bones) {

	ERR_FAIL_COND(p_bones.size() & 1);
	clear_bones();
	for (int i = 0; i < p_bones.size(); i += 2) {
		add_bone(p_bones[i], p_bones[i + 1]);
	}
}

void Polygon2D::set_skeleton(const NodePath &p_skeleton) {
	if (skeleton == p_skeleton)
		return;
	skeleton = p_skeleton;
	update();
}

NodePath Polygon2D::get_skeleton() const {
	return skeleton;
}

void Polygon2D::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &Polygon2D::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &Polygon2D::get_polygon);

	ClassDB::bind_method(D_METHOD("set_uv", "uv"), &Polygon2D::set_uv);
	ClassDB::bind_method(D_METHOD("get_uv"), &Polygon2D::get_uv);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &Polygon2D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &Polygon2D::get_color);

	ClassDB::bind_method(D_METHOD("set_polygons", "polygons"), &Polygon2D::set_polygons);
	ClassDB::bind_method(D_METHOD("get_polygons"), &Polygon2D::get_polygons);

	ClassDB::bind_method(D_METHOD("set_vertex_colors", "vertex_colors"), &Polygon2D::set_vertex_colors);
	ClassDB::bind_method(D_METHOD("get_vertex_colors"), &Polygon2D::get_vertex_colors);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &Polygon2D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &Polygon2D::get_texture);

	ClassDB::bind_method(D_METHOD("set_texture_offset", "texture_offset"), &Polygon2D::set_texture_offset);
	ClassDB::bind_method(D_METHOD("get_texture_offset"), &Polygon2D::get_texture_offset);

	ClassDB::bind_method(D_METHOD("set_texture_rotation", "texture_rotation"), &Polygon2D::set_texture_rotation);
	ClassDB::bind_method(D_METHOD("get_texture_rotation"), &Polygon2D::get_texture_rotation);

	ClassDB::bind_method(D_METHOD("set_texture_rotation_degrees", "texture_rotation"), &Polygon2D::set_texture_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_texture_rotation_degrees"), &Polygon2D::get_texture_rotation_degrees);

	ClassDB::bind_method(D_METHOD("set_texture_scale", "texture_scale"), &Polygon2D::set_texture_scale);
	ClassDB::bind_method(D_METHOD("get_texture_scale"), &Polygon2D::get_texture_scale);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &Polygon2D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &Polygon2D::get_invert);

	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &Polygon2D::set_antialiased);
	ClassDB::bind_method(D_METHOD("get_antialiased"), &Polygon2D::get_antialiased);

	ClassDB::bind_method(D_METHOD("set_invert_border", "invert_border"), &Polygon2D::set_invert_border);
	ClassDB::bind_method(D_METHOD("get_invert_border"), &Polygon2D::get_invert_border);

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &Polygon2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &Polygon2D::get_offset);

	ClassDB::bind_method(D_METHOD("add_bone", "path", "weights"), &Polygon2D::add_bone);
	ClassDB::bind_method(D_METHOD("get_bone_count"), &Polygon2D::get_bone_count);
	ClassDB::bind_method(D_METHOD("get_bone_path", "index"), &Polygon2D::get_bone_path);
	ClassDB::bind_method(D_METHOD("get_bone_weights", "index"), &Polygon2D::get_bone_weights);
	ClassDB::bind_method(D_METHOD("erase_bone", "index"), &Polygon2D::erase_bone);
	ClassDB::bind_method(D_METHOD("clear_bones"), &Polygon2D::clear_bones);
	ClassDB::bind_method(D_METHOD("set_bone_path", "index", "path"), &Polygon2D::set_bone_path);
	ClassDB::bind_method(D_METHOD("set_bone_weights", "index", "weights"), &Polygon2D::set_bone_weights);

	ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &Polygon2D::set_skeleton);
	ClassDB::bind_method(D_METHOD("get_skeleton"), &Polygon2D::get_skeleton);

	ClassDB::bind_method(D_METHOD("set_internal_vertex_count", "internal_vertex_count"), &Polygon2D::set_internal_vertex_count);
	ClassDB::bind_method(D_METHOD("get_internal_vertex_count"), &Polygon2D::get_internal_vertex_count);

	ClassDB::bind_method(D_METHOD("_set_bones", "bones"), &Polygon2D::_set_bones);
	ClassDB::bind_method(D_METHOD("_get_bones"), &Polygon2D::_get_bones);

	ClassDB::bind_method(D_METHOD("_skeleton_bone_setup_changed"), &Polygon2D::_skeleton_bone_setup_changed);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased"), "set_antialiased", "get_antialiased");
	ADD_GROUP("Texture", "");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_GROUP("Texture", "texture_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "texture_offset"), "set_texture_offset", "get_texture_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "texture_scale"), "set_texture_scale", "get_texture_scale");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "texture_rotation_degrees", PROPERTY_HINT_RANGE, "-1080,1080,0.1,or_lesser,or_greater"), "set_texture_rotation_degrees", "get_texture_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "texture_rotation", PROPERTY_HINT_NONE, "", 0), "set_texture_rotation", "get_texture_rotation");
	ADD_GROUP("Skeleton", "");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton2D"), "set_skeleton", "get_skeleton");

	ADD_GROUP("Invert", "invert_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert_enable"), "set_invert", "get_invert");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "invert_border", PROPERTY_HINT_RANGE, "0.1,16384,0.1"), "set_invert_border", "get_invert_border");

	ADD_GROUP("Data", "");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "uv"), "set_uv", "get_uv");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_COLOR_ARRAY, "vertex_colors"), "set_vertex_colors", "get_vertex_colors");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons"), "set_polygons", "get_polygons");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_bones", "_get_bones");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "internal_vertex_count", PROPERTY_HINT_RANGE, "0,1000"), "set_internal_vertex_count", "get_internal_vertex_count");
}

Polygon2D::Polygon2D() {

	invert = 0;
	invert_border = 100;
	antialiased = false;
	tex_rot = 0;
	tex_tile = true;
	tex_scale = Vector2(1, 1);
	color = Color(1, 1, 1);
	rect_cache_dirty = true;
	internal_vertices = 0;
	current_skeleton_id = 0;
}
