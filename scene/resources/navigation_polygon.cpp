/**************************************************************************/
/*  navigation_polygon.cpp                                                */
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

#include "navigation_polygon.h"

#include "core/math/geometry_2d.h"
#include "core/os/mutex.h"
#include "servers/navigation_server_2d.h"

#include "thirdparty/misc/polypartition.h"

#ifdef TOOLS_ENABLED
Rect2 NavigationPolygon::_edit_get_rect() const {
	if (rect_cache_dirty) {
		item_rect = Rect2();
		bool first = true;

		for (int i = 0; i < outlines.size(); i++) {
			const Vector<Vector2> &outline = outlines[i];
			const int outline_size = outline.size();
			if (outline_size < 3) {
				continue;
			}
			const Vector2 *p = outline.ptr();
			for (int j = 0; j < outline_size; j++) {
				if (first) {
					item_rect = Rect2(p[j], Vector2(0, 0));
					first = false;
				} else {
					item_rect.expand_to(p[j]);
				}
			}
		}

		rect_cache_dirty = false;
	}
	return item_rect;
}

bool NavigationPolygon::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	for (int i = 0; i < outlines.size(); i++) {
		const Vector<Vector2> &outline = outlines[i];
		const int outline_size = outline.size();
		if (outline_size < 3) {
			continue;
		}
		if (Geometry2D::is_point_in_polygon(p_point, Variant(outline))) {
			return true;
		}
	}
	return false;
}
#endif

void NavigationPolygon::set_vertices(const Vector<Vector2> &p_vertices) {
	{
		MutexLock lock(navigation_mesh_generation);
		navigation_mesh.unref();
	}
	vertices = p_vertices;
	rect_cache_dirty = true;
}

Vector<Vector2> NavigationPolygon::get_vertices() const {
	return vertices;
}

void NavigationPolygon::_set_polygons(const TypedArray<Vector<int32_t>> &p_array) {
	{
		MutexLock lock(navigation_mesh_generation);
		navigation_mesh.unref();
	}
	polygons.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		polygons.write[i].indices = p_array[i];
	}
}

TypedArray<Vector<int32_t>> NavigationPolygon::_get_polygons() const {
	TypedArray<Vector<int32_t>> ret;
	ret.resize(polygons.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = polygons[i].indices;
	}

	return ret;
}

void NavigationPolygon::_set_outlines(const TypedArray<Vector<Vector2>> &p_array) {
	outlines.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		outlines.write[i] = p_array[i];
	}
	rect_cache_dirty = true;
}

TypedArray<Vector<Vector2>> NavigationPolygon::_get_outlines() const {
	TypedArray<Vector<Vector2>> ret;
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
	{
		MutexLock lock(navigation_mesh_generation);
		navigation_mesh.unref();
	}
}

void NavigationPolygon::add_outline_at_index(const Vector<Vector2> &p_outline, int p_index) {
	outlines.insert(p_index, p_outline);
	rect_cache_dirty = true;
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
	{
		MutexLock lock(navigation_mesh_generation);
		navigation_mesh.unref();
	}
}

void NavigationPolygon::clear() {
	polygons.clear();
	vertices.clear();
	{
		MutexLock lock(navigation_mesh_generation);
		navigation_mesh.unref();
	}
}

Ref<NavigationMesh> NavigationPolygon::get_navigation_mesh() {
	MutexLock lock(navigation_mesh_generation);

	if (navigation_mesh.is_null()) {
		navigation_mesh.instantiate();
		Vector<Vector3> verts;
		{
			verts.resize(get_vertices().size());
			Vector3 *w = verts.ptrw();

			const Vector2 *r = get_vertices().ptr();

			for (int i(0); i < get_vertices().size(); i++) {
				w[i] = Vector3(r[i].x, 0.0, r[i].y);
			}
		}
		navigation_mesh->set_vertices(verts);

		for (int i(0); i < get_polygon_count(); i++) {
			navigation_mesh->add_polygon(get_polygon(i));
		}
		navigation_mesh->set_cell_size(cell_size); // Needed to not fail the cell size check on the server
	}

	return navigation_mesh;
}

void NavigationPolygon::add_outline(const Vector<Vector2> &p_outline) {
	outlines.push_back(p_outline);
	rect_cache_dirty = true;
}

int NavigationPolygon::get_outline_count() const {
	return outlines.size();
}

void NavigationPolygon::set_outline(int p_idx, const Vector<Vector2> &p_outline) {
	ERR_FAIL_INDEX(p_idx, outlines.size());
	outlines.write[p_idx] = p_outline;
	rect_cache_dirty = true;
}

void NavigationPolygon::remove_outline(int p_idx) {
	ERR_FAIL_INDEX(p_idx, outlines.size());
	outlines.remove_at(p_idx);
	rect_cache_dirty = true;
}

Vector<Vector2> NavigationPolygon::get_outline(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, outlines.size(), Vector<Vector2>());
	return outlines[p_idx];
}

void NavigationPolygon::clear_outlines() {
	outlines.clear();
	rect_cache_dirty = true;
}

#ifndef DISABLE_DEPRECATED
void NavigationPolygon::make_polygons_from_outlines() {
	WARN_PRINT("Function make_polygons_from_outlines() is deprecated."
			   "\nUse NavigationServer2D.parse_source_geometry_data() and NavigationServer2D.bake_from_source_geometry_data() instead.");

	{
		MutexLock lock(navigation_mesh_generation);
		navigation_mesh.unref();
	}
	List<TPPLPoly> in_poly, out_poly;

	Vector2 outside_point(-1e10, -1e10);

	for (int i = 0; i < outlines.size(); i++) {
		Vector<Vector2> ol = outlines[i];
		int olsize = ol.size();
		if (olsize < 3) {
			continue;
		}
		const Vector2 *r = ol.ptr();
		for (int j = 0; j < olsize; j++) {
			outside_point.x = MAX(r[j].x, outside_point.x);
			outside_point.y = MAX(r[j].y, outside_point.y);
		}
	}

	outside_point += Vector2(0.7239784, 0.819238); //avoid precision issues

	for (int i = 0; i < outlines.size(); i++) {
		Vector<Vector2> ol = outlines[i];
		int olsize = ol.size();
		if (olsize < 3) {
			continue;
		}
		const Vector2 *r = ol.ptr();

		int interscount = 0;
		//test if this is an outer outline
		for (int k = 0; k < outlines.size(); k++) {
			if (i == k) {
				continue; //no self intersect
			}

			Vector<Vector2> ol2 = outlines[k];
			int olsize2 = ol2.size();
			if (olsize2 < 3) {
				continue;
			}
			const Vector2 *r2 = ol2.ptr();

			for (int l = 0; l < olsize2; l++) {
				if (Geometry2D::segment_intersects_segment(r[0], outside_point, r2[l], r2[(l + 1) % olsize2], nullptr)) {
					interscount++;
				}
			}
		}

		bool outer = (interscount % 2) == 0;

		TPPLPoly tp;
		tp.Init(olsize);
		for (int j = 0; j < olsize; j++) {
			tp[j] = r[j];
		}

		if (outer) {
			tp.SetOrientation(TPPL_ORIENTATION_CCW);
		} else {
			tp.SetOrientation(TPPL_ORIENTATION_CW);
			tp.SetHole(true);
		}

		in_poly.push_back(tp);
	}

	TPPLPartition tpart;
	if (tpart.ConvexPartition_HM(&in_poly, &out_poly) == 0) { //failed!
		ERR_PRINT("NavigationPolygon: Convex partition failed! Failed to convert outlines to a valid NavigationMesh."
				  "\nNavigationPolygon outlines can not overlap vertices or edges inside same outline or with other outlines or have any intersections."
				  "\nAdd the outmost and largest outline first. To add holes inside this outline add the smaller outlines with same winding order.");
		return;
	}

	polygons.clear();
	vertices.clear();

	HashMap<Vector2, int> points;
	for (List<TPPLPoly>::Element *I = out_poly.front(); I; I = I->next()) {
		TPPLPoly &tp = I->get();

		struct Polygon p;

		for (int64_t i = 0; i < tp.GetNumPoints(); i++) {
			HashMap<Vector2, int>::Iterator E = points.find(tp[i]);
			if (!E) {
				E = points.insert(tp[i], vertices.size());
				vertices.push_back(tp[i]);
			}
			p.indices.push_back(E->value);
		}

		polygons.push_back(p);
	}

	emit_changed();
}
#endif // DISABLE_DEPRECATED

void NavigationPolygon::set_cell_size(real_t p_cell_size) {
	cell_size = p_cell_size;
	get_navigation_mesh()->set_cell_size(cell_size);
}

real_t NavigationPolygon::get_cell_size() const {
	return cell_size;
}

void NavigationPolygon::set_parsed_geometry_type(ParsedGeometryType p_geometry_type) {
	ERR_FAIL_INDEX(p_geometry_type, PARSED_GEOMETRY_MAX);
	parsed_geometry_type = p_geometry_type;
	notify_property_list_changed();
}

NavigationPolygon::ParsedGeometryType NavigationPolygon::get_parsed_geometry_type() const {
	return parsed_geometry_type;
}

void NavigationPolygon::set_parsed_collision_mask(uint32_t p_mask) {
	parsed_collision_mask = p_mask;
}

uint32_t NavigationPolygon::get_parsed_collision_mask() const {
	return parsed_collision_mask;
}

void NavigationPolygon::set_parsed_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_parsed_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_parsed_collision_mask(mask);
}

bool NavigationPolygon::get_parsed_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_parsed_collision_mask() & (1 << (p_layer_number - 1));
}

void NavigationPolygon::set_source_geometry_mode(SourceGeometryMode p_geometry_mode) {
	ERR_FAIL_INDEX(p_geometry_mode, SOURCE_GEOMETRY_MAX);
	source_geometry_mode = p_geometry_mode;
	notify_property_list_changed();
}

NavigationPolygon::SourceGeometryMode NavigationPolygon::get_source_geometry_mode() const {
	return source_geometry_mode;
}

void NavigationPolygon::set_source_geometry_group_name(StringName p_group_name) {
	source_geometry_group_name = p_group_name;
}

StringName NavigationPolygon::get_source_geometry_group_name() const {
	return source_geometry_group_name;
}

void NavigationPolygon::set_agent_radius(real_t p_value) {
	ERR_FAIL_COND(p_value < 0);
	agent_radius = p_value;
}

real_t NavigationPolygon::get_agent_radius() const {
	return agent_radius;
}

void NavigationPolygon::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationPolygon::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationPolygon::get_vertices);

	ClassDB::bind_method(D_METHOD("add_polygon", "polygon"), &NavigationPolygon::add_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon_count"), &NavigationPolygon::get_polygon_count);
	ClassDB::bind_method(D_METHOD("get_polygon", "idx"), &NavigationPolygon::get_polygon);
	ClassDB::bind_method(D_METHOD("clear_polygons"), &NavigationPolygon::clear_polygons);
	ClassDB::bind_method(D_METHOD("get_navigation_mesh"), &NavigationPolygon::get_navigation_mesh);

	ClassDB::bind_method(D_METHOD("add_outline", "outline"), &NavigationPolygon::add_outline);
	ClassDB::bind_method(D_METHOD("add_outline_at_index", "outline", "index"), &NavigationPolygon::add_outline_at_index);
	ClassDB::bind_method(D_METHOD("get_outline_count"), &NavigationPolygon::get_outline_count);
	ClassDB::bind_method(D_METHOD("set_outline", "idx", "outline"), &NavigationPolygon::set_outline);
	ClassDB::bind_method(D_METHOD("get_outline", "idx"), &NavigationPolygon::get_outline);
	ClassDB::bind_method(D_METHOD("remove_outline", "idx"), &NavigationPolygon::remove_outline);
	ClassDB::bind_method(D_METHOD("clear_outlines"), &NavigationPolygon::clear_outlines);
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("make_polygons_from_outlines"), &NavigationPolygon::make_polygons_from_outlines);
#endif // DISABLE_DEPRECATED

	ClassDB::bind_method(D_METHOD("_set_polygons", "polygons"), &NavigationPolygon::_set_polygons);
	ClassDB::bind_method(D_METHOD("_get_polygons"), &NavigationPolygon::_get_polygons);

	ClassDB::bind_method(D_METHOD("_set_outlines", "outlines"), &NavigationPolygon::_set_outlines);
	ClassDB::bind_method(D_METHOD("_get_outlines"), &NavigationPolygon::_get_outlines);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &NavigationPolygon::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &NavigationPolygon::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_parsed_geometry_type", "geometry_type"), &NavigationPolygon::set_parsed_geometry_type);
	ClassDB::bind_method(D_METHOD("get_parsed_geometry_type"), &NavigationPolygon::get_parsed_geometry_type);

	ClassDB::bind_method(D_METHOD("set_parsed_collision_mask", "mask"), &NavigationPolygon::set_parsed_collision_mask);
	ClassDB::bind_method(D_METHOD("get_parsed_collision_mask"), &NavigationPolygon::get_parsed_collision_mask);

	ClassDB::bind_method(D_METHOD("set_parsed_collision_mask_value", "layer_number", "value"), &NavigationPolygon::set_parsed_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_parsed_collision_mask_value", "layer_number"), &NavigationPolygon::get_parsed_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_source_geometry_mode", "geometry_mode"), &NavigationPolygon::set_source_geometry_mode);
	ClassDB::bind_method(D_METHOD("get_source_geometry_mode"), &NavigationPolygon::get_source_geometry_mode);

	ClassDB::bind_method(D_METHOD("set_source_geometry_group_name", "group_name"), &NavigationPolygon::set_source_geometry_group_name);
	ClassDB::bind_method(D_METHOD("get_source_geometry_group_name"), &NavigationPolygon::get_source_geometry_group_name);

	ClassDB::bind_method(D_METHOD("set_agent_radius", "agent_radius"), &NavigationPolygon::set_agent_radius);
	ClassDB::bind_method(D_METHOD("get_agent_radius"), &NavigationPolygon::get_agent_radius);

	ClassDB::bind_method(D_METHOD("clear"), &NavigationPolygon::clear);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_polygons", "_get_polygons");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "outlines", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_outlines", "_get_outlines");

	ADD_GROUP("Geometry", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "parsed_geometry_type", PROPERTY_HINT_ENUM, "Mesh Instances,Static Colliders,Meshes and Static Colliders"), "set_parsed_geometry_type", "get_parsed_geometry_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "parsed_collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_parsed_collision_mask", "get_parsed_collision_mask");
	ADD_PROPERTY_DEFAULT("parsed_collision_mask", 0xFFFFFFFF);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "source_geometry_mode", PROPERTY_HINT_ENUM, "Root Node Children,Group With Children,Group Explicit"), "set_source_geometry_mode", "get_source_geometry_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "source_geometry_group_name"), "set_source_geometry_group_name", "get_source_geometry_group_name");
	ADD_PROPERTY_DEFAULT("source_geometry_group_name", StringName("navigation_polygon_source_geometry_group"));
	ADD_GROUP("Cells", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_size", PROPERTY_HINT_RANGE, "1.0,50.0,1.0,or_greater,suffix:px"), "set_cell_size", "get_cell_size");
	ADD_GROUP("Agents", "agent_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_radius", PROPERTY_HINT_RANGE, "0.0,500.0,0.01,or_greater,suffix:px"), "set_agent_radius", "get_agent_radius");

	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_MESH_INSTANCES);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_STATIC_COLLIDERS);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_BOTH);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_MAX);

	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_ROOT_NODE_CHILDREN);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_GROUPS_WITH_CHILDREN);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_GROUPS_EXPLICIT);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_MAX);
}

void NavigationPolygon::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "parsed_collision_mask") {
		if (parsed_geometry_type == PARSED_GEOMETRY_MESH_INSTANCES) {
			p_property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}

	if (p_property.name == "parsed_source_group_name") {
		if (source_geometry_mode == SOURCE_GEOMETRY_ROOT_NODE_CHILDREN) {
			p_property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}
}

NavigationPolygon::NavigationPolygon() {}
