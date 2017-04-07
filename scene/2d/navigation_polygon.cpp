/*************************************************************************/
/*  navigation_polygon.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "navigation_polygon.h"
#include "core_string_names.h"
#include "navigation2d.h"
#include "triangulator.h"

void NavigationPolygon::set_vertices(const PoolVector<Vector2> &p_vertices) {

	vertices = p_vertices;
}

PoolVector<Vector2> NavigationPolygon::get_vertices() const {

	return vertices;
}

void NavigationPolygon::_set_polygons(const Array &p_array) {

	polygons.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		polygons[i].indices = p_array[i];
	}
}

Array NavigationPolygon::_get_polygons() const {

	Array ret;
	ret.resize(polygons.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = polygons[i].indices;
	}

	return ret;
}

void NavigationPolygon::_set_outlines(const Array &p_array) {

	outlines.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		outlines[i] = p_array[i];
	}
}

Array NavigationPolygon::_get_outlines() const {

	Array ret;
	ret.resize(outlines.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = outlines[i];
	}

	return ret;
}

void NavigationPolygon::add_polygon(const Vector<int> &p_polygon) {

	Polygon polygon;
	polygon.indices = p_polygon;
	polygons.push_back(polygon);
}

void NavigationPolygon::add_outline_at_index(const PoolVector<Vector2> &p_outline, int p_index) {

	outlines.insert(p_index, p_outline);
}

int NavigationPolygon::get_polygon_count() const {

	return polygons.size();
}
Vector<int> NavigationPolygon::get_polygon(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, polygons.size(), Vector<int>());
	return polygons[p_idx].indices;
}
void NavigationPolygon::clear_polygons() {

	polygons.clear();
}

void NavigationPolygon::add_outline(const PoolVector<Vector2> &p_outline) {

	outlines.push_back(p_outline);
}

int NavigationPolygon::get_outline_count() const {

	return outlines.size();
}

void NavigationPolygon::set_outline(int p_idx, const PoolVector<Vector2> &p_outline) {
	ERR_FAIL_INDEX(p_idx, outlines.size());
	outlines[p_idx] = p_outline;
}

void NavigationPolygon::remove_outline(int p_idx) {

	ERR_FAIL_INDEX(p_idx, outlines.size());
	outlines.remove(p_idx);
}

PoolVector<Vector2> NavigationPolygon::get_outline(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, outlines.size(), PoolVector<Vector2>());
	return outlines[p_idx];
}

void NavigationPolygon::clear_outlines() {

	outlines.clear();
}
void NavigationPolygon::make_polygons_from_outlines() {

	List<TriangulatorPoly> in_poly, out_poly;

	Vector2 outside_point(-1e10, -1e10);

	for (int i = 0; i < outlines.size(); i++) {

		PoolVector<Vector2> ol = outlines[i];
		int olsize = ol.size();
		if (olsize < 3)
			continue;
		PoolVector<Vector2>::Read r = ol.read();
		for (int j = 0; j < olsize; j++) {
			outside_point.x = MAX(r[j].x, outside_point.x);
			outside_point.y = MAX(r[j].y, outside_point.y);
		}
	}

	outside_point += Vector2(0.7239784, 0.819238); //avoid precision issues

	for (int i = 0; i < outlines.size(); i++) {

		PoolVector<Vector2> ol = outlines[i];
		int olsize = ol.size();
		if (olsize < 3)
			continue;
		PoolVector<Vector2>::Read r = ol.read();

		int interscount = 0;
		//test if this is an outer outline
		for (int k = 0; k < outlines.size(); k++) {

			if (i == k)
				continue; //no self intersect

			PoolVector<Vector2> ol2 = outlines[k];
			int olsize2 = ol2.size();
			if (olsize2 < 3)
				continue;
			PoolVector<Vector2>::Read r2 = ol2.read();

			for (int l = 0; l < olsize2; l++) {

				if (Geometry::segment_intersects_segment_2d(r[0], outside_point, r2[l], r2[(l + 1) % olsize2], NULL)) {
					interscount++;
				}
			}
		}

		bool outer = (interscount % 2) == 0;

		TriangulatorPoly tp;
		tp.Init(olsize);
		for (int j = 0; j < olsize; j++) {
			tp[j] = r[j];
		}

		if (outer)
			tp.SetOrientation(TRIANGULATOR_CCW);
		else {
			tp.SetOrientation(TRIANGULATOR_CW);
			tp.SetHole(true);
		}

		in_poly.push_back(tp);
	}

	TriangulatorPartition tpart;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) { //failed!
		print_line("convex partition failed!");
		return;
	}

	polygons.clear();
	vertices.resize(0);

	Map<Vector2, int> points;
	for (List<TriangulatorPoly>::Element *I = out_poly.front(); I; I = I->next()) {

		TriangulatorPoly &tp = I->get();

		struct Polygon p;

		for (int i = 0; i < tp.GetNumPoints(); i++) {

			Map<Vector2, int>::Element *E = points.find(tp[i]);
			if (!E) {
				E = points.insert(tp[i], vertices.size());
				vertices.push_back(tp[i]);
			}
			p.indices.push_back(E->get());
		}

		polygons.push_back(p);
	}

	emit_signal(CoreStringNames::get_singleton()->changed);
}

void NavigationPolygon::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationPolygon::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationPolygon::get_vertices);

	ClassDB::bind_method(D_METHOD("add_polygon", "polygon"), &NavigationPolygon::add_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon_count"), &NavigationPolygon::get_polygon_count);
	ClassDB::bind_method(D_METHOD("get_polygon", "idx"), &NavigationPolygon::get_polygon);
	ClassDB::bind_method(D_METHOD("clear_polygons"), &NavigationPolygon::clear_polygons);

	ClassDB::bind_method(D_METHOD("add_outline", "outline"), &NavigationPolygon::add_outline);
	ClassDB::bind_method(D_METHOD("add_outline_at_index", "outline", "index"), &NavigationPolygon::add_outline_at_index);
	ClassDB::bind_method(D_METHOD("get_outline_count"), &NavigationPolygon::get_outline_count);
	ClassDB::bind_method(D_METHOD("set_outline", "idx", "outline"), &NavigationPolygon::set_outline);
	ClassDB::bind_method(D_METHOD("get_outline", "idx"), &NavigationPolygon::get_outline);
	ClassDB::bind_method(D_METHOD("remove_outline", "idx"), &NavigationPolygon::remove_outline);
	ClassDB::bind_method(D_METHOD("clear_outlines"), &NavigationPolygon::clear_outlines);
	ClassDB::bind_method(D_METHOD("make_polygons_from_outlines"), &NavigationPolygon::make_polygons_from_outlines);

	ClassDB::bind_method(D_METHOD("_set_polygons", "polygons"), &NavigationPolygon::_set_polygons);
	ClassDB::bind_method(D_METHOD("_get_polygons"), &NavigationPolygon::_get_polygons);

	ClassDB::bind_method(D_METHOD("_set_outlines", "outlines"), &NavigationPolygon::_set_outlines);
	ClassDB::bind_method(D_METHOD("_get_outlines"), &NavigationPolygon::_get_outlines);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR3_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_polygons", "_get_polygons");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "outlines", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_outlines", "_get_outlines");
}

NavigationPolygon::NavigationPolygon() {
}

void NavigationPolygonInstance::set_enabled(bool p_enabled) {

	if (enabled == p_enabled)
		return;
	enabled = p_enabled;

	if (!is_inside_tree())
		return;

	if (!enabled) {

		if (nav_id != -1) {
			navigation->navpoly_remove(nav_id);
			nav_id = -1;
		}
	} else {

		if (navigation) {

			if (navpoly.is_valid()) {

				nav_id = navigation->navpoly_create(navpoly, get_relative_transform_to_parent(navigation), this);
			}
		}
	}

	if (get_tree()->is_editor_hint() || get_tree()->is_debugging_navigation_hint())
		update();

	//update_gizmo();
}

bool NavigationPolygonInstance::is_enabled() const {

	return enabled;
}

/////////////////////////////

void NavigationPolygonInstance::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {

			Node2D *c = this;
			while (c) {

				navigation = c->cast_to<Navigation2D>();
				if (navigation) {

					if (enabled && navpoly.is_valid()) {

						nav_id = navigation->navpoly_create(navpoly, get_relative_transform_to_parent(navigation), this);
					}
					break;
				}

				c = c->get_parent()->cast_to<Node2D>();
			}

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (navigation && nav_id != -1) {
				navigation->navpoly_set_transform(nav_id, get_relative_transform_to_parent(navigation));
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (navigation) {

				if (nav_id != -1) {
					navigation->navpoly_remove(nav_id);
					nav_id = -1;
				}
			}
			navigation = NULL;
		} break;
		case NOTIFICATION_DRAW: {

			if (is_inside_tree() && (get_tree()->is_editor_hint() || get_tree()->is_debugging_navigation_hint()) && navpoly.is_valid()) {

				PoolVector<Vector2> verts = navpoly->get_vertices();
				int vsize = verts.size();
				if (vsize < 3)
					return;

				Color color;
				if (enabled) {
					color = get_tree()->get_debug_navigation_color();
				} else {
					color = get_tree()->get_debug_navigation_disabled_color();
				}
				Vector<Color> colors;
				Vector<Vector2> vertices;
				vertices.resize(vsize);
				colors.resize(vsize);
				{
					PoolVector<Vector2>::Read vr = verts.read();
					for (int i = 0; i < vsize; i++) {
						vertices[i] = vr[i];
						colors[i] = color;
					}
				}

				Vector<int> indices;

				for (int i = 0; i < navpoly->get_polygon_count(); i++) {
					Vector<int> polygon = navpoly->get_polygon(i);

					for (int j = 2; j < polygon.size(); j++) {

						int kofs[3] = { 0, j - 1, j };
						for (int k = 0; k < 3; k++) {

							int idx = polygon[kofs[k]];
							ERR_FAIL_INDEX(idx, vsize);
							indices.push_back(idx);
						}
					}
				}
				VS::get_singleton()->canvas_item_add_triangle_array(get_canvas_item(), indices, vertices, colors);
			}
		} break;
	}
}

void NavigationPolygonInstance::set_navigation_polygon(const Ref<NavigationPolygon> &p_navpoly) {

	if (p_navpoly == navpoly)
		return;

	if (navigation && nav_id != -1) {
		navigation->navpoly_remove(nav_id);
		nav_id = -1;
	}
	if (navpoly.is_valid()) {
		navpoly->disconnect(CoreStringNames::get_singleton()->changed, this, "_navpoly_changed");
	}
	navpoly = p_navpoly;

	if (navpoly.is_valid()) {
		navpoly->connect(CoreStringNames::get_singleton()->changed, this, "_navpoly_changed");
	}

	if (navigation && navpoly.is_valid() && enabled) {
		nav_id = navigation->navpoly_create(navpoly, get_relative_transform_to_parent(navigation), this);
	}
	//update_gizmo();
	_change_notify("navpoly");
	update_configuration_warning();
}

Ref<NavigationPolygon> NavigationPolygonInstance::get_navigation_polygon() const {

	return navpoly;
}

void NavigationPolygonInstance::_navpoly_changed() {

	if (is_inside_tree() && (get_tree()->is_editor_hint() || get_tree()->is_debugging_navigation_hint()))
		update();
}

String NavigationPolygonInstance::get_configuration_warning() const {

	if (!is_visible_in_tree() || !is_inside_tree())
		return String();

	if (!navpoly.is_valid()) {
		return TTR("A NavigationPolygon resource must be set or created for this node to work. Please set a property or draw a polygon.");
	}
	const Node2D *c = this;
	while (c) {

		if (c->cast_to<Navigation2D>()) {
			return String();
		}

		c = c->get_parent()->cast_to<Node2D>();
	}

	return TTR("NavigationPolygonInstance must be a child or grandchild to a Navigation2D node. It only provides navigation data.");
}

void NavigationPolygonInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_navigation_polygon", "navpoly:NavigationPolygon"), &NavigationPolygonInstance::set_navigation_polygon);
	ClassDB::bind_method(D_METHOD("get_navigation_polygon:NavigationPolygon"), &NavigationPolygonInstance::get_navigation_polygon);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationPolygonInstance::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationPolygonInstance::is_enabled);

	ClassDB::bind_method(D_METHOD("_navpoly_changed"), &NavigationPolygonInstance::_navpoly_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navpoly", PROPERTY_HINT_RESOURCE_TYPE, "NavigationPolygon"), "set_navigation_polygon", "get_navigation_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
}

NavigationPolygonInstance::NavigationPolygonInstance() {

	navigation = NULL;
	nav_id = -1;
	enabled = true;
	set_notify_transform(true);
}
