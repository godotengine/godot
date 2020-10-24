/*************************************************************************/
/*  tile_map.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "tile_map.h"

#include "collision_object_2d.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"
#include "scene/2d/area_2d.h"
#include "servers/navigation_server_2d.h"
#include "servers/physics_server_2d.h"

int TileMap::_get_quadrant_size() const {
	if (use_y_sort) {
		return 1;
	} else {
		return quadrant_size;
	}
}

void TileMap::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node2D *c = this;
			while (c) {
				navigation = Object::cast_to<Navigation2D>(c);
				if (navigation) {
					break;
				}

				c = Object::cast_to<Node2D>(c->get_parent());
			}

			if (use_parent) {
				_clear_quadrants();
				collision_parent = Object::cast_to<CollisionObject2D>(get_parent());
			}

			pending_update = true;
			_recreate_quadrants();
			update_dirty_quadrants();
			RID space = get_world_2d()->get_space();
			_update_quadrant_transform();
			_update_quadrant_space(space);
			update_configuration_warning();

		} break;

		case NOTIFICATION_EXIT_TREE: {
			_update_quadrant_space(RID());
			for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
				Quadrant &q = E->get();
				if (navigation) {
					for (Map<PosKey, Quadrant::NavPoly>::Element *F = q.navpoly_ids.front(); F; F = F->next()) {
						NavigationServer2D::get_singleton()->region_set_map(F->get().region, RID());
					}
					q.navpoly_ids.clear();
				}

				if (collision_parent) {
					collision_parent->remove_shape_owner(q.shape_owner_id);
					q.shape_owner_id = -1;
				}

				for (Map<PosKey, Quadrant::Occluder>::Element *F = q.occluder_instances.front(); F; F = F->next()) {
					RS::get_singleton()->free(F->get().id);
				}
				q.occluder_instances.clear();
			}

			collision_parent = nullptr;
			navigation = nullptr;

		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			//move stuff
			_update_quadrant_transform();

		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (use_parent) {
				_recreate_quadrants();
			}

		} break;
	}
}

void TileMap::_update_quadrant_space(const RID &p_space) {
	if (!use_parent) {
		for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			Quadrant &q = E->get();
			PhysicsServer2D::get_singleton()->body_set_space(q.body, p_space);
		}
	}
}

void TileMap::_update_quadrant_transform() {
	if (!is_inside_tree()) {
		return;
	}

	Transform2D global_transform = get_global_transform();

	Transform2D local_transform;
	if (collision_parent) {
		local_transform = get_transform();
	}

	Transform2D nav_rel;
	if (navigation) {
		nav_rel = get_relative_transform_to_parent(navigation);
	}

	for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Quadrant &q = E->get();
		Transform2D xform;
		xform.set_origin(q.pos);

		if (!use_parent) {
			xform = global_transform * xform;
			PhysicsServer2D::get_singleton()->body_set_state(q.body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
		}

		if (navigation) {
			for (Map<PosKey, Quadrant::NavPoly>::Element *F = q.navpoly_ids.front(); F; F = F->next()) {
				NavigationServer2D::get_singleton()->region_set_transform(F->get().region, nav_rel * F->get().xform);
			}
		}

		for (Map<PosKey, Quadrant::Occluder>::Element *F = q.occluder_instances.front(); F; F = F->next()) {
			RS::get_singleton()->canvas_light_occluder_set_transform(F->get().id, global_transform * F->get().xform);
		}
	}
}

void TileMap::set_tileset(const Ref<TileSet> &p_tileset) {
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileMap::_recreate_quadrants));
		tile_set->remove_change_receptor(this);
	}

	_clear_quadrants();
	tile_set = p_tileset;

	if (tile_set.is_valid()) {
		tile_set->connect("changed", callable_mp(this, &TileMap::_recreate_quadrants));
		tile_set->add_change_receptor(this);
	} else {
		clear();
	}

	_recreate_quadrants();
	emit_signal("settings_changed");
}

Ref<TileSet> TileMap::get_tileset() const {
	return tile_set;
}

void TileMap::set_cell_size(Size2 p_size) {
	ERR_FAIL_COND(p_size.x < 1 || p_size.y < 1);

	_clear_quadrants();
	cell_size = p_size;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

Size2 TileMap::get_cell_size() const {
	return cell_size;
}

void TileMap::set_quadrant_size(int p_size) {
	ERR_FAIL_COND_MSG(p_size < 1, "Quadrant size cannot be smaller than 1.");

	_clear_quadrants();
	quadrant_size = p_size;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

int TileMap::get_quadrant_size() const {
	return quadrant_size;
}

void TileMap::_fix_cell_transform(Transform2D &xform, const Cell &p_cell, const Vector2 &p_offset, const Size2 &p_sc) {
	Size2 s = p_sc;
	Vector2 offset = p_offset;

	if (compatibility_mode && !centered_textures) {
		if (tile_origin == TILE_ORIGIN_BOTTOM_LEFT) {
			offset.y += cell_size.y;
		} else if (tile_origin == TILE_ORIGIN_CENTER) {
			offset += cell_size / 2;
		}

		if (s.y > s.x) {
			if ((p_cell.flip_h && (p_cell.flip_v || p_cell.transpose)) || (p_cell.flip_v && !p_cell.transpose)) {
				offset.y += s.y - s.x;
			}
		} else if (s.y < s.x) {
			if ((p_cell.flip_v && (p_cell.flip_h || p_cell.transpose)) || (p_cell.flip_h && !p_cell.transpose)) {
				offset.x += s.x - s.y;
			}
		}
	}

	if (p_cell.transpose) {
		SWAP(xform.elements[0].x, xform.elements[0].y);
		SWAP(xform.elements[1].x, xform.elements[1].y);
		SWAP(offset.x, offset.y);
		SWAP(s.x, s.y);
	}

	if (p_cell.flip_h) {
		xform.elements[0].x = -xform.elements[0].x;
		xform.elements[1].x = -xform.elements[1].x;
		if (compatibility_mode && !centered_textures) {
			if (tile_origin == TILE_ORIGIN_TOP_LEFT || tile_origin == TILE_ORIGIN_BOTTOM_LEFT) {
				offset.x = s.x - offset.x;
			} else if (tile_origin == TILE_ORIGIN_CENTER) {
				offset.x = s.x - offset.x / 2;
			}
		} else {
			offset.x = s.x - offset.x;
		}
	}

	if (p_cell.flip_v) {
		xform.elements[0].y = -xform.elements[0].y;
		xform.elements[1].y = -xform.elements[1].y;
		if (compatibility_mode && !centered_textures) {
			if (tile_origin == TILE_ORIGIN_TOP_LEFT) {
				offset.y = s.y - offset.y;
			} else if (tile_origin == TILE_ORIGIN_BOTTOM_LEFT) {
				offset.y += s.y;
			} else if (tile_origin == TILE_ORIGIN_CENTER) {
				offset.y += s.y;
			}
		} else {
			offset.y = s.y - offset.y;
		}
	}

	if (centered_textures) {
		offset += cell_size / 2 - s / 2;
	}
	xform.elements[2] += offset;
}

void TileMap::_add_shape(int &shape_idx, const Quadrant &p_q, const Ref<Shape2D> &p_shape, const TileSet::ShapeData &p_shape_data, const Transform2D &p_xform, const Vector2 &p_metadata) {
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();

	if (!use_parent) {
		ps->body_add_shape(p_q.body, p_shape->get_rid(), p_xform);
		ps->body_set_shape_metadata(p_q.body, shape_idx, p_metadata);
		ps->body_set_shape_as_one_way_collision(p_q.body, shape_idx, p_shape_data.one_way_collision, p_shape_data.one_way_collision_margin);

	} else if (collision_parent) {
		Transform2D xform = p_xform;
		xform.set_origin(xform.get_origin() + p_q.pos);

		collision_parent->shape_owner_add_shape(p_q.shape_owner_id, p_shape);

		int real_index = collision_parent->shape_owner_get_shape_index(p_q.shape_owner_id, shape_idx);
		RID rid = collision_parent->get_rid();

		if (Object::cast_to<Area2D>(collision_parent) != nullptr) {
			ps->area_set_shape_transform(rid, real_index, get_transform() * xform);
		} else {
			ps->body_set_shape_transform(rid, real_index, get_transform() * xform);
			ps->body_set_shape_metadata(rid, real_index, p_metadata);
			ps->body_set_shape_as_one_way_collision(rid, real_index, p_shape_data.one_way_collision, p_shape_data.one_way_collision_margin);
		}
	}
	shape_idx++;
}

void TileMap::update_dirty_quadrants() {
	if (!pending_update) {
		return;
	}
	if (!is_inside_tree() || !tile_set.is_valid()) {
		pending_update = false;
		return;
	}

	RenderingServer *vs = RenderingServer::get_singleton();
	PhysicsServer2D *ps = PhysicsServer2D::get_singleton();
	Vector2 tofs = get_cell_draw_offset();
	Transform2D nav_rel;
	if (navigation) {
		nav_rel = get_relative_transform_to_parent(navigation);
	}

	Vector2 qofs;

	SceneTree *st = SceneTree::get_singleton();
	Color debug_collision_color;
	Color debug_navigation_color;

	bool debug_shapes = st && st->is_debugging_collisions_hint();
	if (debug_shapes) {
		debug_collision_color = st->get_debug_collisions_color();
	}

	bool debug_navigation = st && st->is_debugging_navigation_hint();
	if (debug_navigation) {
		debug_navigation_color = st->get_debug_navigation_color();
	}

	while (dirty_quadrant_list.first()) {
		Quadrant &q = *dirty_quadrant_list.first()->self();

		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			vs->free(E->get());
		}

		q.canvas_items.clear();

		if (!use_parent) {
			ps->body_clear_shapes(q.body);
		} else if (collision_parent) {
			collision_parent->shape_owner_clear_shapes(q.shape_owner_id);
		}
		int shape_idx = 0;

		if (navigation) {
			for (Map<PosKey, Quadrant::NavPoly>::Element *E = q.navpoly_ids.front(); E; E = E->next()) {
				NavigationServer2D::get_singleton()->region_set_map(E->get().region, RID());
			}
			q.navpoly_ids.clear();
		}

		for (Map<PosKey, Quadrant::Occluder>::Element *E = q.occluder_instances.front(); E; E = E->next()) {
			RS::get_singleton()->free(E->get().id);
		}
		q.occluder_instances.clear();
		Ref<ShaderMaterial> prev_material;
		int prev_z_index = 0;
		RID prev_canvas_item;
		RID prev_debug_canvas_item;

		for (int i = 0; i < q.cells.size(); i++) {
			Map<PosKey, Cell>::Element *E = tile_map.find(q.cells[i]);
			Cell &c = E->get();
			//moment of truth
			if (!tile_set->has_tile(c.id)) {
				continue;
			}
			Ref<Texture2D> tex = tile_set->tile_get_texture(c.id);
			Vector2 tile_ofs = tile_set->tile_get_texture_offset(c.id);

			Vector2 wofs = _map_to_world(E->key().x, E->key().y);
			Vector2 offset = wofs - q.pos + tofs;

			if (!tex.is_valid()) {
				continue;
			}

			Ref<ShaderMaterial> mat = tile_set->tile_get_material(c.id);
			int z_index = tile_set->tile_get_z_index(c.id);

			if (tile_set->tile_get_tile_mode(c.id) == TileSet::AUTO_TILE ||
					tile_set->tile_get_tile_mode(c.id) == TileSet::ATLAS_TILE) {
				z_index += tile_set->autotile_get_z_index(c.id, Vector2(c.autotile_coord_x, c.autotile_coord_y));
			}

			RID canvas_item;
			RID debug_canvas_item;

			if (prev_canvas_item == RID() || prev_material != mat || prev_z_index != z_index) {
				canvas_item = vs->canvas_item_create();
				if (mat.is_valid()) {
					vs->canvas_item_set_material(canvas_item, mat->get_rid());
				}
				vs->canvas_item_set_parent(canvas_item, get_canvas_item());
				_update_item_material_state(canvas_item);
				Transform2D xform;
				xform.set_origin(q.pos);
				vs->canvas_item_set_transform(canvas_item, xform);
				vs->canvas_item_set_light_mask(canvas_item, get_light_mask());
				vs->canvas_item_set_z_index(canvas_item, z_index);

				vs->canvas_item_set_default_texture_filter(canvas_item, RS::CanvasItemTextureFilter(CanvasItem::get_texture_filter()));
				vs->canvas_item_set_default_texture_repeat(canvas_item, RS::CanvasItemTextureRepeat(CanvasItem::get_texture_repeat()));

				q.canvas_items.push_back(canvas_item);

				if (debug_shapes) {
					debug_canvas_item = vs->canvas_item_create();
					vs->canvas_item_set_parent(debug_canvas_item, canvas_item);
					vs->canvas_item_set_z_as_relative_to_parent(debug_canvas_item, false);
					vs->canvas_item_set_z_index(debug_canvas_item, RS::CANVAS_ITEM_Z_MAX - 1);
					q.canvas_items.push_back(debug_canvas_item);
					prev_debug_canvas_item = debug_canvas_item;
				}

				prev_canvas_item = canvas_item;
				prev_material = mat;
				prev_z_index = z_index;

			} else {
				canvas_item = prev_canvas_item;
				if (debug_shapes) {
					debug_canvas_item = prev_debug_canvas_item;
				}
			}

			Rect2 r = tile_set->tile_get_region(c.id);
			if (tile_set->tile_get_tile_mode(c.id) == TileSet::AUTO_TILE || tile_set->tile_get_tile_mode(c.id) == TileSet::ATLAS_TILE) {
				int spacing = tile_set->autotile_get_spacing(c.id);
				r.size = tile_set->autotile_get_size(c.id);
				r.position += (r.size + Vector2(spacing, spacing)) * Vector2(c.autotile_coord_x, c.autotile_coord_y);
			}

			Size2 s;
			if (r == Rect2()) {
				s = tex->get_size();
			} else {
				s = r.size;
			}

			Rect2 rect;
			rect.position = offset.floor();
			rect.size = s;
			rect.size.x += fp_adjust;
			rect.size.y += fp_adjust;

			if (compatibility_mode && !centered_textures) {
				if (rect.size.y > rect.size.x) {
					if ((c.flip_h && (c.flip_v || c.transpose)) || (c.flip_v && !c.transpose)) {
						tile_ofs.y += rect.size.y - rect.size.x;
					}
				} else if (rect.size.y < rect.size.x) {
					if ((c.flip_v && (c.flip_h || c.transpose)) || (c.flip_h && !c.transpose)) {
						tile_ofs.x += rect.size.x - rect.size.y;
					}
				}
			}

			if (c.transpose) {
				SWAP(tile_ofs.x, tile_ofs.y);
				if (centered_textures) {
					rect.position.x += cell_size.x / 2 - rect.size.y / 2;
					rect.position.y += cell_size.y / 2 - rect.size.x / 2;
				}
			} else if (centered_textures) {
				rect.position += cell_size / 2 - rect.size / 2;
			}

			if (c.flip_h) {
				rect.size.x = -rect.size.x;
				tile_ofs.x = -tile_ofs.x;
			}

			if (c.flip_v) {
				rect.size.y = -rect.size.y;
				tile_ofs.y = -tile_ofs.y;
			}

			if (compatibility_mode && !centered_textures) {
				if (tile_origin == TILE_ORIGIN_TOP_LEFT) {
					rect.position += tile_ofs;

				} else if (tile_origin == TILE_ORIGIN_BOTTOM_LEFT) {
					rect.position += tile_ofs;

					if (c.transpose) {
						if (c.flip_h) {
							rect.position.x -= cell_size.x;
						} else {
							rect.position.x += cell_size.x;
						}
					} else {
						if (c.flip_v) {
							rect.position.y -= cell_size.y;
						} else {
							rect.position.y += cell_size.y;
						}
					}

				} else if (tile_origin == TILE_ORIGIN_CENTER) {
					rect.position += tile_ofs;

					if (c.flip_h) {
						rect.position.x -= cell_size.x / 2;
					} else {
						rect.position.x += cell_size.x / 2;
					}

					if (c.flip_v) {
						rect.position.y -= cell_size.y / 2;
					} else {
						rect.position.y += cell_size.y / 2;
					}
				}
			} else {
				rect.position += tile_ofs;
			}

			Color modulate = tile_set->tile_get_modulate(c.id);
			Color self_modulate = get_self_modulate();
			modulate = Color(modulate.r * self_modulate.r, modulate.g * self_modulate.g,
					modulate.b * self_modulate.b, modulate.a * self_modulate.a);
			if (r == Rect2()) {
				tex->draw_rect(canvas_item, rect, false, modulate, c.transpose);
			} else {
				tex->draw_rect_region(canvas_item, rect, r, modulate, c.transpose, clip_uv);
			}

			Vector<TileSet::ShapeData> shapes = tile_set->tile_get_shapes(c.id);

			for (int j = 0; j < shapes.size(); j++) {
				Ref<Shape2D> shape = shapes[j].shape;
				if (shape.is_valid()) {
					if (tile_set->tile_get_tile_mode(c.id) == TileSet::SINGLE_TILE || (shapes[j].autotile_coord.x == c.autotile_coord_x && shapes[j].autotile_coord.y == c.autotile_coord_y)) {
						Transform2D xform;
						xform.set_origin(offset.floor());

						Vector2 shape_ofs = shapes[j].shape_transform.get_origin();

						_fix_cell_transform(xform, c, shape_ofs, s);

						xform *= shapes[j].shape_transform.untranslated();

						if (debug_canvas_item.is_valid()) {
							vs->canvas_item_add_set_transform(debug_canvas_item, xform);
							shape->draw(debug_canvas_item, debug_collision_color);
						}

						if (shape->has_meta("decomposed")) {
							Array _shapes = shape->get_meta("decomposed");
							for (int k = 0; k < _shapes.size(); k++) {
								Ref<ConvexPolygonShape2D> convex = _shapes[k];
								if (convex.is_valid()) {
									_add_shape(shape_idx, q, convex, shapes[j], xform, Vector2(E->key().x, E->key().y));
#ifdef DEBUG_ENABLED
								} else {
									print_error("The TileSet assigned to the TileMap " + get_name() + " has an invalid convex shape.");
#endif
								}
							}
						} else {
							_add_shape(shape_idx, q, shape, shapes[j], xform, Vector2(E->key().x, E->key().y));
						}
					}
				}
			}

			if (debug_canvas_item.is_valid()) {
				vs->canvas_item_add_set_transform(debug_canvas_item, Transform2D());
			}

			if (navigation) {
				Ref<NavigationPolygon> navpoly;
				Vector2 npoly_ofs;
				if (tile_set->tile_get_tile_mode(c.id) == TileSet::AUTO_TILE || tile_set->tile_get_tile_mode(c.id) == TileSet::ATLAS_TILE) {
					navpoly = tile_set->autotile_get_navigation_polygon(c.id, Vector2(c.autotile_coord_x, c.autotile_coord_y));
					npoly_ofs = Vector2();
				} else {
					navpoly = tile_set->tile_get_navigation_polygon(c.id);
					npoly_ofs = tile_set->tile_get_navigation_polygon_offset(c.id);
				}

				if (navpoly.is_valid()) {
					Transform2D xform;
					xform.set_origin(offset.floor() + q.pos);
					_fix_cell_transform(xform, c, npoly_ofs, s);

					RID region = NavigationServer2D::get_singleton()->region_create();
					NavigationServer2D::get_singleton()->region_set_map(region, navigation->get_rid());
					NavigationServer2D::get_singleton()->region_set_transform(region, nav_rel * xform);
					NavigationServer2D::get_singleton()->region_set_navpoly(region, navpoly);

					Quadrant::NavPoly np;
					np.region = region;
					np.xform = xform;
					q.navpoly_ids[E->key()] = np;

					if (debug_navigation) {
						RID debug_navigation_item = vs->canvas_item_create();
						vs->canvas_item_set_parent(debug_navigation_item, canvas_item);
						vs->canvas_item_set_z_as_relative_to_parent(debug_navigation_item, false);
						vs->canvas_item_set_z_index(debug_navigation_item, RS::CANVAS_ITEM_Z_MAX - 2); // Display one below collision debug

						if (debug_navigation_item.is_valid()) {
							Vector<Vector2> navigation_polygon_vertices = navpoly->get_vertices();
							int vsize = navigation_polygon_vertices.size();

							if (vsize > 2) {
								Vector<Color> colors;
								Vector<Vector2> vertices;
								vertices.resize(vsize);
								colors.resize(vsize);
								{
									const Vector2 *vr = navigation_polygon_vertices.ptr();
									for (int j = 0; j < vsize; j++) {
										vertices.write[j] = vr[j];
										colors.write[j] = debug_navigation_color;
									}
								}

								Vector<int> indices;

								for (int j = 0; j < navpoly->get_polygon_count(); j++) {
									Vector<int> polygon = navpoly->get_polygon(j);

									for (int k = 2; k < polygon.size(); k++) {
										int kofs[3] = { 0, k - 1, k };
										for (int l = 0; l < 3; l++) {
											int idx = polygon[kofs[l]];
											ERR_FAIL_INDEX(idx, vsize);
											indices.push_back(idx);
										}
									}
								}
								Transform2D navxform;
								navxform.set_origin(offset.floor());
								_fix_cell_transform(navxform, c, npoly_ofs, s);

								vs->canvas_item_set_transform(debug_navigation_item, navxform);
								vs->canvas_item_add_triangle_array(debug_navigation_item, indices, vertices, colors);
							}
						}
					}
				}
			}

			Ref<OccluderPolygon2D> occluder;
			if (tile_set->tile_get_tile_mode(c.id) == TileSet::AUTO_TILE || tile_set->tile_get_tile_mode(c.id) == TileSet::ATLAS_TILE) {
				occluder = tile_set->autotile_get_light_occluder(c.id, Vector2(c.autotile_coord_x, c.autotile_coord_y));
			} else {
				occluder = tile_set->tile_get_light_occluder(c.id);
			}
			if (occluder.is_valid()) {
				Vector2 occluder_ofs = tile_set->tile_get_occluder_offset(c.id);
				Transform2D xform;
				xform.set_origin(offset.floor() + q.pos);
				_fix_cell_transform(xform, c, occluder_ofs, s);

				RID orid = RS::get_singleton()->canvas_light_occluder_create();
				RS::get_singleton()->canvas_light_occluder_set_transform(orid, get_global_transform() * xform);
				RS::get_singleton()->canvas_light_occluder_set_polygon(orid, occluder->get_rid());
				RS::get_singleton()->canvas_light_occluder_attach_to_canvas(orid, get_canvas());
				RS::get_singleton()->canvas_light_occluder_set_light_mask(orid, occluder_light_mask);
				Quadrant::Occluder oc;
				oc.xform = xform;
				oc.id = orid;
				q.occluder_instances[E->key()] = oc;
			}
		}

		dirty_quadrant_list.remove(dirty_quadrant_list.first());
		quadrant_order_dirty = true;
	}

	pending_update = false;

	if (quadrant_order_dirty) {
		int index = -(int64_t)0x80000000; //always must be drawn below children
		for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			Quadrant &q = E->get();
			for (List<RID>::Element *F = q.canvas_items.front(); F; F = F->next()) {
				RS::get_singleton()->canvas_item_set_draw_index(F->get(), index++);
			}
		}

		quadrant_order_dirty = false;
	}

	_recompute_rect_cache();
}

void TileMap::_recompute_rect_cache() {
#ifdef DEBUG_ENABLED

	if (!rect_cache_dirty) {
		return;
	}

	Rect2 r_total;
	for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Rect2 r;
		r.position = _map_to_world(E->key().x * _get_quadrant_size(), E->key().y * _get_quadrant_size());
		r.expand_to(_map_to_world(E->key().x * _get_quadrant_size() + _get_quadrant_size(), E->key().y * _get_quadrant_size()));
		r.expand_to(_map_to_world(E->key().x * _get_quadrant_size() + _get_quadrant_size(), E->key().y * _get_quadrant_size() + _get_quadrant_size()));
		r.expand_to(_map_to_world(E->key().x * _get_quadrant_size(), E->key().y * _get_quadrant_size() + _get_quadrant_size()));
		if (E == quadrant_map.front()) {
			r_total = r;
		} else {
			r_total = r_total.merge(r);
		}
	}

	rect_cache = r_total;

	item_rect_changed();

	rect_cache_dirty = false;
#endif
}

Map<TileMap::PosKey, TileMap::Quadrant>::Element *TileMap::_create_quadrant(const PosKey &p_qk) {
	Transform2D xform;
	//xform.set_origin(Point2(p_qk.x,p_qk.y)*cell_size*quadrant_size);
	Quadrant q;
	q.pos = _map_to_world(p_qk.x * _get_quadrant_size(), p_qk.y * _get_quadrant_size());
	q.pos += get_cell_draw_offset();
	if (tile_origin == TILE_ORIGIN_CENTER) {
		q.pos += cell_size / 2;
	} else if (tile_origin == TILE_ORIGIN_BOTTOM_LEFT) {
		q.pos.y += cell_size.y;
	}

	xform.set_origin(q.pos);
	//q.canvas_item = RenderingServer::get_singleton()->canvas_item_create();
	if (!use_parent) {
		q.body = PhysicsServer2D::get_singleton()->body_create();
		PhysicsServer2D::get_singleton()->body_set_mode(q.body, use_kinematic ? PhysicsServer2D::BODY_MODE_KINEMATIC : PhysicsServer2D::BODY_MODE_STATIC);

		PhysicsServer2D::get_singleton()->body_attach_object_instance_id(q.body, get_instance_id());
		PhysicsServer2D::get_singleton()->body_set_collision_layer(q.body, collision_layer);
		PhysicsServer2D::get_singleton()->body_set_collision_mask(q.body, collision_mask);
		PhysicsServer2D::get_singleton()->body_set_param(q.body, PhysicsServer2D::BODY_PARAM_FRICTION, friction);
		PhysicsServer2D::get_singleton()->body_set_param(q.body, PhysicsServer2D::BODY_PARAM_BOUNCE, bounce);

		if (is_inside_tree()) {
			xform = get_global_transform() * xform;
			RID space = get_world_2d()->get_space();
			PhysicsServer2D::get_singleton()->body_set_space(q.body, space);
		}

		PhysicsServer2D::get_singleton()->body_set_state(q.body, PhysicsServer2D::BODY_STATE_TRANSFORM, xform);
	} else if (collision_parent) {
		xform = get_transform() * xform;
		q.shape_owner_id = collision_parent->create_shape_owner(this);
	} else {
		q.shape_owner_id = -1;
	}

	rect_cache_dirty = true;
	quadrant_order_dirty = true;
	return quadrant_map.insert(p_qk, q);
}

void TileMap::_erase_quadrant(Map<PosKey, Quadrant>::Element *Q) {
	Quadrant &q = Q->get();
	if (!use_parent) {
		PhysicsServer2D::get_singleton()->free(q.body);
	} else if (collision_parent) {
		collision_parent->remove_shape_owner(q.shape_owner_id);
	}

	for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
		RenderingServer::get_singleton()->free(E->get());
	}
	q.canvas_items.clear();
	if (q.dirty_list.in_list()) {
		dirty_quadrant_list.remove(&q.dirty_list);
	}

	if (navigation) {
		for (Map<PosKey, Quadrant::NavPoly>::Element *E = q.navpoly_ids.front(); E; E = E->next()) {
			NavigationServer2D::get_singleton()->region_set_map(E->get().region, RID());
		}
		q.navpoly_ids.clear();
	}

	for (Map<PosKey, Quadrant::Occluder>::Element *E = q.occluder_instances.front(); E; E = E->next()) {
		RS::get_singleton()->free(E->get().id);
	}
	q.occluder_instances.clear();

	quadrant_map.erase(Q);
	rect_cache_dirty = true;
}

void TileMap::_make_quadrant_dirty(Map<PosKey, Quadrant>::Element *Q, bool update) {
	Quadrant &q = Q->get();
	if (!q.dirty_list.in_list()) {
		dirty_quadrant_list.add(&q.dirty_list);
	}

	if (pending_update) {
		return;
	}
	pending_update = true;
	if (!is_inside_tree()) {
		return;
	}

	if (update) {
		call_deferred("update_dirty_quadrants");
	}
}

void TileMap::set_cellv(const Vector2 &p_pos, int p_tile, bool p_flip_x, bool p_flip_y, bool p_transpose) {
	set_cell(p_pos.x, p_pos.y, p_tile, p_flip_x, p_flip_y, p_transpose);
}

void TileMap::_set_celld(const Vector2 &p_pos, const Dictionary &p_data) {
	Variant v_pos_x = p_pos.x, v_pos_y = p_pos.y, v_tile = p_data["id"], v_flip_h = p_data["flip_h"], v_flip_v = p_data["flip_y"], v_transpose = p_data["transpose"], v_autotile_coord = p_data["auto_coord"];
	const Variant *args[7] = { &v_pos_x, &v_pos_y, &v_tile, &v_flip_h, &v_flip_v, &v_transpose, &v_autotile_coord };
	Callable::CallError ce;
	call("set_cell", args, 7, ce);
}

void TileMap::set_cell(int p_x, int p_y, int p_tile, bool p_flip_x, bool p_flip_y, bool p_transpose, Vector2 p_autotile_coord) {
	PosKey pk(p_x, p_y);

	Map<PosKey, Cell>::Element *E = tile_map.find(pk);
	if (!E && p_tile == INVALID_CELL) {
		return; //nothing to do
	}

	PosKey qk = pk.to_quadrant(_get_quadrant_size());
	if (p_tile == INVALID_CELL) {
		//erase existing
		tile_map.erase(pk);
		Map<PosKey, Quadrant>::Element *Q = quadrant_map.find(qk);
		ERR_FAIL_COND(!Q);
		Quadrant &q = Q->get();
		q.cells.erase(pk);
		if (q.cells.size() == 0) {
			_erase_quadrant(Q);
		} else {
			_make_quadrant_dirty(Q);
		}

		used_size_cache_dirty = true;
		return;
	}

	Map<PosKey, Quadrant>::Element *Q = quadrant_map.find(qk);

	if (!E) {
		E = tile_map.insert(pk, Cell());
		if (!Q) {
			Q = _create_quadrant(qk);
		}
		Quadrant &q = Q->get();
		q.cells.insert(pk);
	} else {
		ERR_FAIL_COND(!Q); // quadrant should exist...

		if (E->get().id == p_tile && E->get().flip_h == p_flip_x && E->get().flip_v == p_flip_y && E->get().transpose == p_transpose && E->get().autotile_coord_x == (uint16_t)p_autotile_coord.x && E->get().autotile_coord_y == (uint16_t)p_autotile_coord.y) {
			return; //nothing changed
		}
	}

	Cell &c = E->get();

	c.id = p_tile;
	c.flip_h = p_flip_x;
	c.flip_v = p_flip_y;
	c.transpose = p_transpose;
	c.autotile_coord_x = (uint16_t)p_autotile_coord.x;
	c.autotile_coord_y = (uint16_t)p_autotile_coord.y;

	_make_quadrant_dirty(Q);
	used_size_cache_dirty = true;
}

int TileMap::get_cellv(const Vector2 &p_pos) const {
	return get_cell(p_pos.x, p_pos.y);
}

void TileMap::make_bitmask_area_dirty(const Vector2 &p_pos) {
	for (int x = p_pos.x - 1; x <= p_pos.x + 1; x++) {
		for (int y = p_pos.y - 1; y <= p_pos.y + 1; y++) {
			PosKey p(x, y);
			if (dirty_bitmask.find(p) == nullptr) {
				dirty_bitmask.push_back(p);
			}
		}
	}
}

void TileMap::update_bitmask_area(const Vector2 &p_pos) {
	for (int x = p_pos.x - 1; x <= p_pos.x + 1; x++) {
		for (int y = p_pos.y - 1; y <= p_pos.y + 1; y++) {
			update_cell_bitmask(x, y);
		}
	}
}

void TileMap::update_bitmask_region(const Vector2 &p_start, const Vector2 &p_end) {
	if ((p_end.x < p_start.x || p_end.y < p_start.y) || (p_end.x == p_start.x && p_end.y == p_start.y)) {
		Array a = get_used_cells();
		for (int i = 0; i < a.size(); i++) {
			Vector2 vector = (Vector2)a[i];
			update_cell_bitmask(vector.x, vector.y);
		}
		return;
	}
	for (int x = p_start.x - 1; x <= p_end.x + 1; x++) {
		for (int y = p_start.y - 1; y <= p_end.y + 1; y++) {
			update_cell_bitmask(x, y);
		}
	}
}

void TileMap::update_cell_bitmask(int p_x, int p_y) {
	ERR_FAIL_COND_MSG(tile_set.is_null(), "Cannot update cell bitmask if Tileset is not open.");
	PosKey p(p_x, p_y);
	Map<PosKey, Cell>::Element *E = tile_map.find(p);
	if (E != nullptr) {
		int id = get_cell(p_x, p_y);
		if (tile_set->tile_get_tile_mode(id) == TileSet::AUTO_TILE) {
			uint16_t mask = 0;
			if (tile_set->autotile_get_bitmask_mode(id) == TileSet::BITMASK_2X2) {
				if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y))) {
					mask |= TileSet::BIND_TOPLEFT;
				}
				if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y))) {
					mask |= TileSet::BIND_TOPRIGHT;
				}
				if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y))) {
					mask |= TileSet::BIND_BOTTOMLEFT;
				}
				if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y))) {
					mask |= TileSet::BIND_BOTTOMRIGHT;
				}
			} else {
				if (tile_set->autotile_get_bitmask_mode(id) == TileSet::BITMASK_3X3_MINIMAL) {
					if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y))) {
						mask |= TileSet::BIND_TOPLEFT;
					}
					if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y - 1)) && tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y))) {
						mask |= TileSet::BIND_TOPRIGHT;
					}
					if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y))) {
						mask |= TileSet::BIND_BOTTOMLEFT;
					}
					if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x, p_y + 1)) && tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y))) {
						mask |= TileSet::BIND_BOTTOMRIGHT;
					}
				} else {
					if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y - 1))) {
						mask |= TileSet::BIND_TOPLEFT;
					}
					if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y - 1))) {
						mask |= TileSet::BIND_TOPRIGHT;
					}
					if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y + 1))) {
						mask |= TileSet::BIND_BOTTOMLEFT;
					}
					if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y + 1))) {
						mask |= TileSet::BIND_BOTTOMRIGHT;
					}
				}
				if (tile_set->is_tile_bound(id, get_cell(p_x, p_y - 1))) {
					mask |= TileSet::BIND_TOP;
				}
				if (tile_set->is_tile_bound(id, get_cell(p_x - 1, p_y))) {
					mask |= TileSet::BIND_LEFT;
				}
				mask |= TileSet::BIND_CENTER;
				if (tile_set->is_tile_bound(id, get_cell(p_x + 1, p_y))) {
					mask |= TileSet::BIND_RIGHT;
				}
				if (tile_set->is_tile_bound(id, get_cell(p_x, p_y + 1))) {
					mask |= TileSet::BIND_BOTTOM;
				}
			}
			Vector2 coord = tile_set->autotile_get_subtile_for_bitmask(id, mask, this, Vector2(p_x, p_y));
			E->get().autotile_coord_x = (int)coord.x;
			E->get().autotile_coord_y = (int)coord.y;

			PosKey qk = p.to_quadrant(_get_quadrant_size());
			Map<PosKey, Quadrant>::Element *Q = quadrant_map.find(qk);
			_make_quadrant_dirty(Q);

		} else if (tile_set->tile_get_tile_mode(id) == TileSet::SINGLE_TILE) {
			E->get().autotile_coord_x = 0;
			E->get().autotile_coord_y = 0;
		} else if (tile_set->tile_get_tile_mode(id) == TileSet::ATLAS_TILE) {
			if (tile_set->autotile_get_bitmask(id, Vector2(p_x, p_y)) == TileSet::BIND_CENTER) {
				Vector2 coord = tile_set->atlastile_get_subtile_by_priority(id, this, Vector2(p_x, p_y));

				E->get().autotile_coord_x = (int)coord.x;
				E->get().autotile_coord_y = (int)coord.y;
			}
		}
	}
}

void TileMap::update_dirty_bitmask() {
	while (dirty_bitmask.size() > 0) {
		update_cell_bitmask(dirty_bitmask[0].x, dirty_bitmask[0].y);
		dirty_bitmask.pop_front();
	}
}

void TileMap::fix_invalid_tiles() {
	ERR_FAIL_COND_MSG(tile_set.is_null(), "Cannot fix invalid tiles if Tileset is not open.");
	for (Map<PosKey, Cell>::Element *E = tile_map.front(); E; E = E->next()) {
		if (!tile_set->has_tile(get_cell(E->key().x, E->key().y))) {
			set_cell(E->key().x, E->key().y, INVALID_CELL);
		}
	}
}

int TileMap::get_cell(int p_x, int p_y) const {
	PosKey pk(p_x, p_y);

	const Map<PosKey, Cell>::Element *E = tile_map.find(pk);

	if (!E) {
		return INVALID_CELL;
	}

	return E->get().id;
}

bool TileMap::is_cell_x_flipped(int p_x, int p_y) const {
	PosKey pk(p_x, p_y);

	const Map<PosKey, Cell>::Element *E = tile_map.find(pk);

	if (!E) {
		return false;
	}

	return E->get().flip_h;
}

bool TileMap::is_cell_y_flipped(int p_x, int p_y) const {
	PosKey pk(p_x, p_y);

	const Map<PosKey, Cell>::Element *E = tile_map.find(pk);

	if (!E) {
		return false;
	}

	return E->get().flip_v;
}

bool TileMap::is_cell_transposed(int p_x, int p_y) const {
	PosKey pk(p_x, p_y);

	const Map<PosKey, Cell>::Element *E = tile_map.find(pk);

	if (!E) {
		return false;
	}

	return E->get().transpose;
}

void TileMap::set_cell_autotile_coord(int p_x, int p_y, const Vector2 &p_coord) {
	PosKey pk(p_x, p_y);

	const Map<PosKey, Cell>::Element *E = tile_map.find(pk);

	if (!E) {
		return;
	}

	Cell c = E->get();
	c.autotile_coord_x = p_coord.x;
	c.autotile_coord_y = p_coord.y;
	tile_map[pk] = c;

	PosKey qk = pk.to_quadrant(_get_quadrant_size());
	Map<PosKey, Quadrant>::Element *Q = quadrant_map.find(qk);

	if (!Q) {
		return;
	}

	_make_quadrant_dirty(Q);
}

Vector2 TileMap::get_cell_autotile_coord(int p_x, int p_y) const {
	PosKey pk(p_x, p_y);

	const Map<PosKey, Cell>::Element *E = tile_map.find(pk);

	if (!E) {
		return Vector2();
	}

	return Vector2(E->get().autotile_coord_x, E->get().autotile_coord_y);
}

void TileMap::_recreate_quadrants() {
	_clear_quadrants();

	for (Map<PosKey, Cell>::Element *E = tile_map.front(); E; E = E->next()) {
		PosKey qk = PosKey(E->key().x, E->key().y).to_quadrant(_get_quadrant_size());

		Map<PosKey, Quadrant>::Element *Q = quadrant_map.find(qk);
		if (!Q) {
			Q = _create_quadrant(qk);
			dirty_quadrant_list.add(&Q->get().dirty_list);
		}

		Q->get().cells.insert(E->key());
		_make_quadrant_dirty(Q, false);
	}
	update_dirty_quadrants();
}

void TileMap::_clear_quadrants() {
	while (quadrant_map.size()) {
		_erase_quadrant(quadrant_map.front());
	}
}

void TileMap::set_material(const Ref<Material> &p_material) {
	CanvasItem::set_material(p_material);
	_update_all_items_material_state();
}

void TileMap::set_use_parent_material(bool p_use_parent_material) {
	CanvasItem::set_use_parent_material(p_use_parent_material);
	_update_all_items_material_state();
}

void TileMap::_update_all_items_material_state() {
	for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		Quadrant &q = E->get();
		for (List<RID>::Element *F = q.canvas_items.front(); F; F = F->next()) {
			_update_item_material_state(F->get());
		}
	}
}

void TileMap::_update_item_material_state(const RID &p_canvas_item) {
	RS::get_singleton()->canvas_item_set_use_parent_material(p_canvas_item, get_use_parent_material() || get_material().is_valid());
}

void TileMap::clear() {
	_clear_quadrants();
	tile_map.clear();
	used_size_cache_dirty = true;
}

void TileMap::_set_tile_data(const Vector<int> &p_data) {
	ERR_FAIL_COND(format > FORMAT_2);

	int c = p_data.size();
	const int *r = p_data.ptr();

	int offset = (format == FORMAT_2) ? 3 : 2;

	clear();
	for (int i = 0; i < c; i += offset) {
		const uint8_t *ptr = (const uint8_t *)&r[i];
		uint8_t local[12];
		for (int j = 0; j < ((format == FORMAT_2) ? 12 : 8); j++) {
			local[j] = ptr[j];
		}

#ifdef BIG_ENDIAN_ENABLED

		SWAP(local[0], local[3]);
		SWAP(local[1], local[2]);
		SWAP(local[4], local[7]);
		SWAP(local[5], local[6]);
		//TODO: ask someone to check this...
		if (FORMAT == FORMAT_2) {
			SWAP(local[8], local[11]);
			SWAP(local[9], local[10]);
		}
#endif

		uint16_t x = decode_uint16(&local[0]);
		uint16_t y = decode_uint16(&local[2]);
		uint32_t v = decode_uint32(&local[4]);
		bool flip_h = v & (1 << 29);
		bool flip_v = v & (1 << 30);
		bool transpose = v & (1 << 31);
		v &= (1 << 29) - 1;
		int16_t coord_x = 0;
		int16_t coord_y = 0;
		if (format == FORMAT_2) {
			coord_x = decode_uint16(&local[8]);
			coord_y = decode_uint16(&local[10]);
		}

		set_cell(x, y, v, flip_h, flip_v, transpose, Vector2(coord_x, coord_y));
	}
}

Vector<int> TileMap::_get_tile_data() const {
	Vector<int> data;
	data.resize(tile_map.size() * 3);
	int *w = data.ptrw();

	// Save in highest format

	int idx = 0;
	for (const Map<PosKey, Cell>::Element *E = tile_map.front(); E; E = E->next()) {
		uint8_t *ptr = (uint8_t *)&w[idx];
		encode_uint16(E->key().x, &ptr[0]);
		encode_uint16(E->key().y, &ptr[2]);
		uint32_t val = E->get().id;
		if (E->get().flip_h) {
			val |= (1 << 29);
		}
		if (E->get().flip_v) {
			val |= (1 << 30);
		}
		if (E->get().transpose) {
			val |= (1 << 31);
		}
		encode_uint32(val, &ptr[4]);
		encode_uint16(E->get().autotile_coord_x, &ptr[8]);
		encode_uint16(E->get().autotile_coord_y, &ptr[10]);
		idx += 3;
	}

	return data;
}

#ifdef TOOLS_ENABLED
Rect2 TileMap::_edit_get_rect() const {
	if (pending_update) {
		const_cast<TileMap *>(this)->update_dirty_quadrants();
	} else {
		const_cast<TileMap *>(this)->_recompute_rect_cache();
	}
	return rect_cache;
}
#endif

void TileMap::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	if (!use_parent) {
		for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			Quadrant &q = E->get();
			PhysicsServer2D::get_singleton()->body_set_collision_layer(q.body, collision_layer);
		}
	}
}

void TileMap::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	if (!use_parent) {
		for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			Quadrant &q = E->get();
			PhysicsServer2D::get_singleton()->body_set_collision_mask(q.body, collision_mask);
		}
	}
}

void TileMap::set_collision_layer_bit(int p_bit, bool p_value) {
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << p_bit;
	} else {
		layer &= ~(1 << p_bit);
	}
	set_collision_layer(layer);
}

void TileMap::set_collision_mask_bit(int p_bit, bool p_value) {
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool TileMap::get_collision_use_kinematic() const {
	return use_kinematic;
}

void TileMap::set_collision_use_kinematic(bool p_use_kinematic) {
	_clear_quadrants();
	use_kinematic = p_use_kinematic;
	_recreate_quadrants();
}

bool TileMap::get_collision_use_parent() const {
	return use_parent;
}

void TileMap::set_collision_use_parent(bool p_use_parent) {
	if (use_parent == p_use_parent) {
		return;
	}

	_clear_quadrants();

	use_parent = p_use_parent;
	set_notify_local_transform(use_parent);

	if (use_parent && is_inside_tree()) {
		collision_parent = Object::cast_to<CollisionObject2D>(get_parent());
	} else {
		collision_parent = nullptr;
	}

	_recreate_quadrants();
	_change_notify();
	update_configuration_warning();
}

void TileMap::set_collision_friction(float p_friction) {
	friction = p_friction;
	if (!use_parent) {
		for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			Quadrant &q = E->get();
			PhysicsServer2D::get_singleton()->body_set_param(q.body, PhysicsServer2D::BODY_PARAM_FRICTION, p_friction);
		}
	}
}

float TileMap::get_collision_friction() const {
	return friction;
}

void TileMap::set_collision_bounce(float p_bounce) {
	bounce = p_bounce;
	if (!use_parent) {
		for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
			Quadrant &q = E->get();
			PhysicsServer2D::get_singleton()->body_set_param(q.body, PhysicsServer2D::BODY_PARAM_BOUNCE, p_bounce);
		}
	}
}

float TileMap::get_collision_bounce() const {
	return bounce;
}

uint32_t TileMap::get_collision_layer() const {
	return collision_layer;
}

uint32_t TileMap::get_collision_mask() const {
	return collision_mask;
}

bool TileMap::get_collision_layer_bit(int p_bit) const {
	return get_collision_layer() & (1 << p_bit);
}

bool TileMap::get_collision_mask_bit(int p_bit) const {
	return get_collision_mask() & (1 << p_bit);
}

void TileMap::set_mode(Mode p_mode) {
	_clear_quadrants();
	mode = p_mode;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

TileMap::Mode TileMap::get_mode() const {
	return mode;
}

void TileMap::set_half_offset(HalfOffset p_half_offset) {
	_clear_quadrants();
	half_offset = p_half_offset;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

void TileMap::set_tile_origin(TileOrigin p_tile_origin) {
	_clear_quadrants();
	tile_origin = p_tile_origin;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

TileMap::TileOrigin TileMap::get_tile_origin() const {
	return tile_origin;
}

Vector2 TileMap::get_cell_draw_offset() const {
	switch (mode) {
		case MODE_SQUARE: {
			return Vector2();
		} break;
		case MODE_ISOMETRIC: {
			return Vector2(-cell_size.x * 0.5, 0);

		} break;
		case MODE_CUSTOM: {
			Vector2 min;
			min.x = MIN(custom_transform[0].x, min.x);
			min.y = MIN(custom_transform[0].y, min.y);
			min.x = MIN(custom_transform[1].x, min.x);
			min.y = MIN(custom_transform[1].y, min.y);
			return min;
		} break;
	}

	return Vector2();
}

TileMap::HalfOffset TileMap::get_half_offset() const {
	return half_offset;
}

Transform2D TileMap::get_cell_transform() const {
	switch (mode) {
		case MODE_SQUARE: {
			Transform2D m;
			m[0] *= cell_size.x;
			m[1] *= cell_size.y;
			return m;
		} break;
		case MODE_ISOMETRIC: {
			//isometric only makes sense when y is positive in both x and y vectors, otherwise
			//the drawing of tiles will overlap
			Transform2D m;
			m[0] = Vector2(cell_size.x * 0.5, cell_size.y * 0.5);
			m[1] = Vector2(-cell_size.x * 0.5, cell_size.y * 0.5);
			return m;

		} break;
		case MODE_CUSTOM: {
			return custom_transform;
		} break;
	}

	return Transform2D();
}

void TileMap::set_custom_transform(const Transform2D &p_xform) {
	_clear_quadrants();
	custom_transform = p_xform;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

Transform2D TileMap::get_custom_transform() const {
	return custom_transform;
}

Vector2 TileMap::_map_to_world(int p_x, int p_y, bool p_ignore_ofs) const {
	Vector2 ret = get_cell_transform().xform(Vector2(p_x, p_y));
	if (!p_ignore_ofs) {
		switch (half_offset) {
			case HALF_OFFSET_X:
			case HALF_OFFSET_NEGATIVE_X: {
				if (ABS(p_y) & 1) {
					ret += get_cell_transform()[0] * (half_offset == HALF_OFFSET_X ? 0.5 : -0.5);
				}
			} break;
			case HALF_OFFSET_Y:
			case HALF_OFFSET_NEGATIVE_Y: {
				if (ABS(p_x) & 1) {
					ret += get_cell_transform()[1] * (half_offset == HALF_OFFSET_Y ? 0.5 : -0.5);
				}
			} break;
			case HALF_OFFSET_DISABLED: {
				// Nothing to do.
			}
		}
	}
	return ret;
}

bool TileMap::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "format") {
		if (p_value.get_type() == Variant::INT) {
			format = (DataFormat)(p_value.operator int64_t()); // Set format used for loading
			return true;
		}
	} else if (p_name == "tile_data") {
		if (p_value.is_array()) {
			_set_tile_data(p_value);
			return true;
		}
		return false;
	}
	return false;
}

bool TileMap::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "format") {
		r_ret = FORMAT_2; // When saving, always save highest format
		return true;
	} else if (p_name == "tile_data") {
		r_ret = _get_tile_data();
		return true;
	}
	return false;
}

void TileMap::_get_property_list(List<PropertyInfo> *p_list) const {
	PropertyInfo p(Variant::INT, "format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL);
	p_list->push_back(p);

	p = PropertyInfo(Variant::OBJECT, "tile_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL);
	p_list->push_back(p);
}

void TileMap::_validate_property(PropertyInfo &property) const {
	if (use_parent && property.name != "collision_use_parent" && property.name.begins_with("collision_")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

Vector2 TileMap::map_to_world(const Vector2 &p_pos, bool p_ignore_ofs) const {
	return _map_to_world(p_pos.x, p_pos.y, p_ignore_ofs);
}

Vector2 TileMap::world_to_map(const Vector2 &p_pos) const {
	Vector2 ret = get_cell_transform().affine_inverse().xform(p_pos);

	switch (half_offset) {
		case HALF_OFFSET_X: {
			if (int(floor(ret.y)) & 1) {
				ret.x -= 0.5;
			}
		} break;
		case HALF_OFFSET_NEGATIVE_X: {
			if (int(floor(ret.y)) & 1) {
				ret.x += 0.5;
			}
		} break;
		case HALF_OFFSET_Y: {
			if (int(floor(ret.x)) & 1) {
				ret.y -= 0.5;
			}
		} break;
		case HALF_OFFSET_NEGATIVE_Y: {
			if (int(floor(ret.x)) & 1) {
				ret.y += 0.5;
			}
		} break;
		case HALF_OFFSET_DISABLED: {
			// Nothing to do.
		}
	}

	// Account for precision errors on the border (GH-23250).
	// 0.00005 is 5*CMP_EPSILON, results would start being unpredictable if
	// cell size is > 15,000, but we can hardly have more precision anyway with
	// floating point.
	ret += Vector2(0.00005, 0.00005);
	return ret.floor();
}

void TileMap::set_y_sort_enabled(bool p_enable) {
	_clear_quadrants();
	use_y_sort = p_enable;
	RS::get_singleton()->canvas_item_set_sort_children_by_y(get_canvas_item(), use_y_sort);
	_recreate_quadrants();
	emit_signal("settings_changed");
}

bool TileMap::is_y_sort_enabled() const {
	return use_y_sort;
}

void TileMap::set_compatibility_mode(bool p_enable) {
	_clear_quadrants();
	compatibility_mode = p_enable;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

bool TileMap::is_compatibility_mode_enabled() const {
	return compatibility_mode;
}

void TileMap::set_centered_textures(bool p_enable) {
	_clear_quadrants();
	centered_textures = p_enable;
	_recreate_quadrants();
	emit_signal("settings_changed");
}

bool TileMap::is_centered_textures_enabled() const {
	return centered_textures;
}

TypedArray<Vector2i> TileMap::get_used_cells() const {
	TypedArray<Vector2i> a;
	a.resize(tile_map.size());
	int i = 0;
	for (Map<PosKey, Cell>::Element *E = tile_map.front(); E; E = E->next()) {
		Vector2i p(E->key().x, E->key().y);
		a[i++] = p;
	}

	return a;
}

TypedArray<Vector2i> TileMap::get_used_cells_by_index(int p_id) const {
	TypedArray<Vector2i> a;
	for (Map<PosKey, Cell>::Element *E = tile_map.front(); E; E = E->next()) {
		if (E->value().id == p_id) {
			Vector2i p(E->key().x, E->key().y);
			a.push_back(p);
		}
	}

	return a;
}

Rect2 TileMap::get_used_rect() { // Not const because of cache

	if (used_size_cache_dirty) {
		if (tile_map.size() > 0) {
			used_size_cache = Rect2(tile_map.front()->key().x, tile_map.front()->key().y, 0, 0);

			for (Map<PosKey, Cell>::Element *E = tile_map.front(); E; E = E->next()) {
				used_size_cache.expand_to(Vector2(E->key().x, E->key().y));
			}

			used_size_cache.size += Vector2(1, 1);
		} else {
			used_size_cache = Rect2();
		}

		used_size_cache_dirty = false;
	}

	return used_size_cache;
}

void TileMap::set_occluder_light_mask(int p_mask) {
	occluder_light_mask = p_mask;
	for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		for (Map<PosKey, Quadrant::Occluder>::Element *F = E->get().occluder_instances.front(); F; F = F->next()) {
			RenderingServer::get_singleton()->canvas_light_occluder_set_light_mask(F->get().id, occluder_light_mask);
		}
	}
}

int TileMap::get_occluder_light_mask() const {
	return occluder_light_mask;
}

void TileMap::set_light_mask(int p_light_mask) {
	CanvasItem::set_light_mask(p_light_mask);
	for (Map<PosKey, Quadrant>::Element *E = quadrant_map.front(); E; E = E->next()) {
		for (List<RID>::Element *F = E->get().canvas_items.front(); F; F = F->next()) {
			RenderingServer::get_singleton()->canvas_item_set_light_mask(F->get(), get_light_mask());
		}
	}
}

void TileMap::set_clip_uv(bool p_enable) {
	if (clip_uv == p_enable) {
		return;
	}

	_clear_quadrants();
	clip_uv = p_enable;
	_recreate_quadrants();
}

bool TileMap::get_clip_uv() const {
	return clip_uv;
}

void TileMap::set_texture_filter(TextureFilter p_texture_filter) {
	CanvasItem::set_texture_filter(p_texture_filter);
	for (Map<PosKey, Quadrant>::Element *F = quadrant_map.front(); F; F = F->next()) {
		Quadrant &q = F->get();
		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			RenderingServer::get_singleton()->canvas_item_set_default_texture_filter(E->get(), RS::CanvasItemTextureFilter(p_texture_filter));
			_make_quadrant_dirty(F);
		}
	}
}

void TileMap::set_texture_repeat(CanvasItem::TextureRepeat p_texture_repeat) {
	CanvasItem::set_texture_repeat(p_texture_repeat);
	for (Map<PosKey, Quadrant>::Element *F = quadrant_map.front(); F; F = F->next()) {
		Quadrant &q = F->get();
		for (List<RID>::Element *E = q.canvas_items.front(); E; E = E->next()) {
			RenderingServer::get_singleton()->canvas_item_set_default_texture_repeat(E->get(), RS::CanvasItemTextureRepeat(p_texture_repeat));
			_make_quadrant_dirty(F);
		}
	}
}

String TileMap::get_configuration_warning() const {
	String warning = Node2D::get_configuration_warning();

	if (use_parent && !collision_parent) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		return TTR("TileMap with Use Parent on needs a parent CollisionObject2D to give shapes to. Please use it as a child of Area2D, StaticBody2D, RigidBody2D, KinematicBody2D, etc. to give them a shape.");
	}

	return warning;
}

void TileMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tileset", "tileset"), &TileMap::set_tileset);
	ClassDB::bind_method(D_METHOD("get_tileset"), &TileMap::get_tileset);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &TileMap::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &TileMap::get_mode);

	ClassDB::bind_method(D_METHOD("set_half_offset", "half_offset"), &TileMap::set_half_offset);
	ClassDB::bind_method(D_METHOD("get_half_offset"), &TileMap::get_half_offset);

	ClassDB::bind_method(D_METHOD("set_custom_transform", "custom_transform"), &TileMap::set_custom_transform);
	ClassDB::bind_method(D_METHOD("get_custom_transform"), &TileMap::get_custom_transform);

	ClassDB::bind_method(D_METHOD("set_cell_size", "size"), &TileMap::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &TileMap::get_cell_size);

	ClassDB::bind_method(D_METHOD("_set_old_cell_size", "size"), &TileMap::_set_old_cell_size);
	ClassDB::bind_method(D_METHOD("_get_old_cell_size"), &TileMap::_get_old_cell_size);

	ClassDB::bind_method(D_METHOD("set_quadrant_size", "size"), &TileMap::set_quadrant_size);
	ClassDB::bind_method(D_METHOD("get_quadrant_size"), &TileMap::get_quadrant_size);

	ClassDB::bind_method(D_METHOD("set_tile_origin", "origin"), &TileMap::set_tile_origin);
	ClassDB::bind_method(D_METHOD("get_tile_origin"), &TileMap::get_tile_origin);

	ClassDB::bind_method(D_METHOD("set_clip_uv", "enable"), &TileMap::set_clip_uv);
	ClassDB::bind_method(D_METHOD("get_clip_uv"), &TileMap::get_clip_uv);

	ClassDB::bind_method(D_METHOD("set_y_sort_enabled", "enable"), &TileMap::set_y_sort_enabled);
	ClassDB::bind_method(D_METHOD("is_y_sort_enabled"), &TileMap::is_y_sort_enabled);

	ClassDB::bind_method(D_METHOD("set_compatibility_mode", "enable"), &TileMap::set_compatibility_mode);
	ClassDB::bind_method(D_METHOD("is_compatibility_mode_enabled"), &TileMap::is_compatibility_mode_enabled);

	ClassDB::bind_method(D_METHOD("set_centered_textures", "enable"), &TileMap::set_centered_textures);
	ClassDB::bind_method(D_METHOD("is_centered_textures_enabled"), &TileMap::is_centered_textures_enabled);

	ClassDB::bind_method(D_METHOD("set_collision_use_kinematic", "use_kinematic"), &TileMap::set_collision_use_kinematic);
	ClassDB::bind_method(D_METHOD("get_collision_use_kinematic"), &TileMap::get_collision_use_kinematic);

	ClassDB::bind_method(D_METHOD("set_collision_use_parent", "use_parent"), &TileMap::set_collision_use_parent);
	ClassDB::bind_method(D_METHOD("get_collision_use_parent"), &TileMap::get_collision_use_parent);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &TileMap::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &TileMap::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &TileMap::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &TileMap::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_layer_bit", "bit", "value"), &TileMap::set_collision_layer_bit);
	ClassDB::bind_method(D_METHOD("get_collision_layer_bit", "bit"), &TileMap::get_collision_layer_bit);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &TileMap::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &TileMap::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_collision_friction", "value"), &TileMap::set_collision_friction);
	ClassDB::bind_method(D_METHOD("get_collision_friction"), &TileMap::get_collision_friction);

	ClassDB::bind_method(D_METHOD("set_collision_bounce", "value"), &TileMap::set_collision_bounce);
	ClassDB::bind_method(D_METHOD("get_collision_bounce"), &TileMap::get_collision_bounce);

	ClassDB::bind_method(D_METHOD("set_occluder_light_mask", "mask"), &TileMap::set_occluder_light_mask);
	ClassDB::bind_method(D_METHOD("get_occluder_light_mask"), &TileMap::get_occluder_light_mask);

	ClassDB::bind_method(D_METHOD("set_cell", "x", "y", "tile", "flip_x", "flip_y", "transpose", "autotile_coord"), &TileMap::set_cell, DEFVAL(false), DEFVAL(false), DEFVAL(false), DEFVAL(Vector2()));
	ClassDB::bind_method(D_METHOD("set_cellv", "position", "tile", "flip_x", "flip_y", "transpose"), &TileMap::set_cellv, DEFVAL(false), DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_set_celld", "position", "data"), &TileMap::_set_celld);
	ClassDB::bind_method(D_METHOD("get_cell", "x", "y"), &TileMap::get_cell);
	ClassDB::bind_method(D_METHOD("get_cellv", "position"), &TileMap::get_cellv);
	ClassDB::bind_method(D_METHOD("is_cell_x_flipped", "x", "y"), &TileMap::is_cell_x_flipped);
	ClassDB::bind_method(D_METHOD("is_cell_y_flipped", "x", "y"), &TileMap::is_cell_y_flipped);
	ClassDB::bind_method(D_METHOD("is_cell_transposed", "x", "y"), &TileMap::is_cell_transposed);

	ClassDB::bind_method(D_METHOD("get_cell_autotile_coord", "x", "y"), &TileMap::get_cell_autotile_coord);

	ClassDB::bind_method(D_METHOD("fix_invalid_tiles"), &TileMap::fix_invalid_tiles);
	ClassDB::bind_method(D_METHOD("clear"), &TileMap::clear);

	ClassDB::bind_method(D_METHOD("get_used_cells"), &TileMap::get_used_cells);
	ClassDB::bind_method(D_METHOD("get_used_cells_by_index", "index"), &TileMap::get_used_cells_by_index);
	ClassDB::bind_method(D_METHOD("get_used_rect"), &TileMap::get_used_rect);

	ClassDB::bind_method(D_METHOD("map_to_world", "map_position", "ignore_half_ofs"), &TileMap::map_to_world, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("world_to_map", "world_position"), &TileMap::world_to_map);

	ClassDB::bind_method(D_METHOD("_clear_quadrants"), &TileMap::_clear_quadrants);
	ClassDB::bind_method(D_METHOD("update_dirty_quadrants"), &TileMap::update_dirty_quadrants);

	ClassDB::bind_method(D_METHOD("update_bitmask_area", "position"), &TileMap::update_bitmask_area);
	ClassDB::bind_method(D_METHOD("update_bitmask_region", "start", "end"), &TileMap::update_bitmask_region, DEFVAL(Vector2()), DEFVAL(Vector2()));

	ClassDB::bind_method(D_METHOD("_set_tile_data"), &TileMap::_set_tile_data);
	ClassDB::bind_method(D_METHOD("_get_tile_data"), &TileMap::_get_tile_data);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Square,Isometric,Custom"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tile_set", PROPERTY_HINT_RESOURCE_TYPE, "TileSet"), "set_tileset", "get_tileset");

	ADD_GROUP("Cell", "cell_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "cell_size", PROPERTY_HINT_RANGE, "1,8192,1"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_quadrant_size", PROPERTY_HINT_RANGE, "1,128,1"), "set_quadrant_size", "get_quadrant_size");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "cell_custom_transform"), "set_custom_transform", "get_custom_transform");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_half_offset", PROPERTY_HINT_ENUM, "Offset X,Offset Y,Disabled,Offset Negative X,Offset Negative Y"), "set_half_offset", "get_half_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_tile_origin", PROPERTY_HINT_ENUM, "Top Left,Center,Bottom Left"), "set_tile_origin", "get_tile_origin");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cell_y_sort"), "set_y_sort_enabled", "is_y_sort_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "compatibility_mode"), "set_compatibility_mode", "is_compatibility_mode_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "centered_textures"), "set_centered_textures", "is_centered_textures_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cell_clip_uv"), "set_clip_uv", "get_clip_uv");

	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_use_parent", PROPERTY_HINT_NONE, ""), "set_collision_use_parent", "get_collision_use_parent");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collision_use_kinematic", PROPERTY_HINT_NONE, ""), "set_collision_use_kinematic", "get_collision_use_kinematic");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_friction", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_collision_friction", "get_collision_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_bounce", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_collision_bounce", "get_collision_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_GROUP("Occluder", "occluder_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "occluder_light_mask", PROPERTY_HINT_LAYERS_2D_RENDER), "set_occluder_light_mask", "get_occluder_light_mask");

	ADD_PROPERTY_DEFAULT("format", FORMAT_1);

	ADD_SIGNAL(MethodInfo("settings_changed"));

	BIND_CONSTANT(INVALID_CELL);

	BIND_ENUM_CONSTANT(MODE_SQUARE);
	BIND_ENUM_CONSTANT(MODE_ISOMETRIC);
	BIND_ENUM_CONSTANT(MODE_CUSTOM);

	BIND_ENUM_CONSTANT(HALF_OFFSET_X);
	BIND_ENUM_CONSTANT(HALF_OFFSET_Y);
	BIND_ENUM_CONSTANT(HALF_OFFSET_DISABLED);
	BIND_ENUM_CONSTANT(HALF_OFFSET_NEGATIVE_X);
	BIND_ENUM_CONSTANT(HALF_OFFSET_NEGATIVE_Y);

	BIND_ENUM_CONSTANT(TILE_ORIGIN_TOP_LEFT);
	BIND_ENUM_CONSTANT(TILE_ORIGIN_CENTER);
	BIND_ENUM_CONSTANT(TILE_ORIGIN_BOTTOM_LEFT);
}

void TileMap::_changed_callback(Object *p_changed, const char *p_prop) {
	if (tile_set.is_valid() && tile_set.ptr() == p_changed) {
		emit_signal("settings_changed");
	}
}

TileMap::TileMap() {
	rect_cache_dirty = true;
	used_size_cache_dirty = true;
	pending_update = false;
	quadrant_order_dirty = false;
	quadrant_size = 16;
	cell_size = Size2(64, 64);
	custom_transform = Transform2D(64, 0, 0, 64, 0, 0);
	collision_layer = 1;
	collision_mask = 1;
	friction = 1;
	bounce = 0;
	mode = MODE_SQUARE;
	half_offset = HALF_OFFSET_DISABLED;
	use_parent = false;
	collision_parent = nullptr;
	use_kinematic = false;
	navigation = nullptr;
	use_y_sort = false;
	compatibility_mode = false;
	centered_textures = false;
	occluder_light_mask = 1;
	clip_uv = false;
	format = FORMAT_1; // Assume lowest possible format if none is present

	fp_adjust = 0.00001;
	tile_origin = TILE_ORIGIN_TOP_LEFT;
	set_notify_transform(true);
	set_notify_local_transform(false);
}

TileMap::~TileMap() {
	if (tile_set.is_valid()) {
		tile_set->remove_change_receptor(this);
	}

	clear();
}
