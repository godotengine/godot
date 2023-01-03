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

#include "core/core_string_names.h"
#include "core/math/geometry_2d.h"
#include "core/os/mutex.h"
#include "servers/navigation/navigation_mesh_generator.h"

#include "thirdparty/clipper2/include/clipper2/clipper.h"
#include "thirdparty/misc/polypartition.h"

void NavigationPolygon::_bind_methods() {
	ClassDB::bind_method(D_METHOD("commit_changes"), &NavigationPolygon::commit_changes);

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationPolygon::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationPolygon::get_vertices);

	ClassDB::bind_method(D_METHOD("add_polygon", "polygon"), &NavigationPolygon::add_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon_count"), &NavigationPolygon::get_polygon_count);
	ClassDB::bind_method(D_METHOD("get_polygon", "index"), &NavigationPolygon::get_polygon);
	ClassDB::bind_method(D_METHOD("clear_polygons"), &NavigationPolygon::clear_polygons);
	ClassDB::bind_method(D_METHOD("get_navigation_mesh"), &NavigationPolygon::get_navigation_mesh);

	ClassDB::bind_method(D_METHOD("add_outline", "outline"), &NavigationPolygon::add_outline);
	ClassDB::bind_method(D_METHOD("add_outline_at_index", "outline", "index"), &NavigationPolygon::add_outline_at_index);
	ClassDB::bind_method(D_METHOD("get_outline_count"), &NavigationPolygon::get_outline_count);
	ClassDB::bind_method(D_METHOD("set_outline", "index", "outline"), &NavigationPolygon::set_outline);
	ClassDB::bind_method(D_METHOD("get_outline", "index"), &NavigationPolygon::get_outline);
	ClassDB::bind_method(D_METHOD("remove_outline", "index"), &NavigationPolygon::remove_outline);
	ClassDB::bind_method(D_METHOD("clear_outlines"), &NavigationPolygon::clear_outlines);

	ClassDB::bind_method(D_METHOD("set_polygons", "polygons"), &NavigationPolygon::set_polygons);
	ClassDB::bind_method(D_METHOD("get_polygons"), &NavigationPolygon::get_polygons);

	ClassDB::bind_method(D_METHOD("set_outlines", "outlines"), &NavigationPolygon::set_outlines);
	ClassDB::bind_method(D_METHOD("get_outlines"), &NavigationPolygon::get_outlines);

	ClassDB::bind_method(D_METHOD("set_parsed_geometry_type", "geometry_type"), &NavigationPolygon::set_parsed_geometry_type);
	ClassDB::bind_method(D_METHOD("get_parsed_geometry_type"), &NavigationPolygon::get_parsed_geometry_type);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &NavigationPolygon::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &NavigationPolygon::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &NavigationPolygon::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &NavigationPolygon::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_source_geometry_mode", "mask"), &NavigationPolygon::set_source_geometry_mode);
	ClassDB::bind_method(D_METHOD("get_source_geometry_mode"), &NavigationPolygon::get_source_geometry_mode);

	ClassDB::bind_method(D_METHOD("set_source_group_name", "mask"), &NavigationPolygon::set_source_group_name);
	ClassDB::bind_method(D_METHOD("get_source_group_name"), &NavigationPolygon::get_source_group_name);

	ClassDB::bind_method(D_METHOD("set_polygon_bake_fillrule", "polygon_fillrule"), &NavigationPolygon::set_polygon_bake_fillrule);
	ClassDB::bind_method(D_METHOD("get_polygon_bake_fillrule"), &NavigationPolygon::get_polygon_bake_fillrule);

	ClassDB::bind_method(D_METHOD("set_offsetting_jointype", "offsetting_jointype"), &NavigationPolygon::set_offsetting_jointype);
	ClassDB::bind_method(D_METHOD("get_offsetting_jointype"), &NavigationPolygon::get_offsetting_jointype);

	ClassDB::bind_method(D_METHOD("set_agent_radius", "agent_radius"), &NavigationPolygon::set_agent_radius);
	ClassDB::bind_method(D_METHOD("get_agent_radius"), &NavigationPolygon::get_agent_radius);

	ClassDB::bind_method(D_METHOD("set_baked_outlines", "baked_outlines"), &NavigationPolygon::set_baked_outlines);
	ClassDB::bind_method(D_METHOD("get_baked_outlines"), &NavigationPolygon::get_baked_outlines);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("make_polygons_from_outlines"), &NavigationPolygon::make_polygons_from_outlines);
#endif // DISABLE_DEPRECATED

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_polygons", "get_polygons");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "outlines", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_outlines", "get_outlines");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "baked_outlines", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_baked_outlines", "get_baked_outlines");

	ADD_GROUP("Polygons", "polygon_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "polygon_bake_fillrule", PROPERTY_HINT_ENUM, "EvenOdd,NonZero,Positive,Negative"), "set_polygon_bake_fillrule", "get_polygon_bake_fillrule");
	ADD_GROUP("Geometry", "geometry_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_parsed_geometry_type", PROPERTY_HINT_ENUM, "Mesh Instances,Static Colliders,Both"), "set_parsed_geometry_type", "get_parsed_geometry_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY_DEFAULT("geometry_collision_mask", 0xFFFFFFFF);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_source_geometry_mode", PROPERTY_HINT_ENUM, "Root Node Children,Group With Children,Group Explicit"), "set_source_geometry_mode", "get_source_geometry_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "geometry_source_group_name"), "set_source_group_name", "get_source_group_name");
	ADD_PROPERTY_DEFAULT("geometry_source_group_name", StringName("navigation_polygon_source_group"));
	ADD_GROUP("Agents", "agent_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_radius", PROPERTY_HINT_RANGE, "0.0,500.0,0.01,or_greater,suffix:px"), "set_agent_radius", "get_agent_radius");
	ADD_GROUP("Offsetting", "offsetting_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "offsetting_jointype", PROPERTY_HINT_ENUM, "Square,Round,Miter"), "set_offsetting_jointype", "get_offsetting_jointype");

	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_MESH_INSTANCES);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_STATIC_COLLIDERS);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_BOTH);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_MAX);

	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_ROOT_NODE_CHILDREN);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_GROUPS_WITH_CHILDREN);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_GROUPS_EXPLICIT);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_MAX);

	BIND_ENUM_CONSTANT(POLYGON_FILLRULE_EVENODD);
	BIND_ENUM_CONSTANT(POLYGON_FILLRULE_NONZERO);
	BIND_ENUM_CONSTANT(POLYGON_FILLRULE_POSITIVE);
	BIND_ENUM_CONSTANT(POLYGON_FILLRULE_NEGATIVE);
	BIND_ENUM_CONSTANT(POLYGON_FILLRULE_MAX);

	BIND_ENUM_CONSTANT(OFFSETTING_JOINTYPE_SQUARE);
	BIND_ENUM_CONSTANT(OFFSETTING_JOINTYPE_ROUND);
	BIND_ENUM_CONSTANT(OFFSETTING_JOINTYPE_MITER);
	BIND_ENUM_CONSTANT(OFFSETTING_JOINTYPE_MAX);
}

void NavigationPolygon::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "geometry_collision_mask") {
		if (parsed_geometry_type == PARSED_GEOMETRY_MESH_INSTANCES) {
			p_property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}

	if (p_property.name == "geometry_source_group_name") {
		if (source_geometry_mode == SOURCE_GEOMETRY_ROOT_NODE_CHILDREN) {
			p_property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}
}

RID NavigationPolygon::get_rid() const {
	if (navigation_mesh.is_valid()) {
		navigation_mesh->get_rid();
	}
	return RID();
}

void NavigationPolygon::set_vertices(const Vector<Vector2> &p_vertices) {
	vertices = p_vertices;
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

Vector<Vector2> NavigationPolygon::get_vertices() const {
	return vertices;
}

void NavigationPolygon::set_polygons(const TypedArray<Vector<int32_t>> &p_array) {
	polygons.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		polygons.write[i] = p_array[i];
	}
	navigation_polygon_dirty = true;
}

TypedArray<Vector<int32_t>> NavigationPolygon::get_polygons() const {
	TypedArray<Vector<int32_t>> ret;
	ret.resize(polygons.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = polygons[i];
	}

	return ret;
}

void NavigationPolygon::set_outlines(const TypedArray<Vector<Vector2>> &p_array) {
	outlines.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		outlines.write[i] = p_array[i];
	}
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

TypedArray<Vector<Vector2>> NavigationPolygon::get_outlines() const {
	TypedArray<Vector<Vector2>> ret;
	ret.resize(outlines.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = outlines[i];
	}

	return ret;
}

void NavigationPolygon::add_polygon(const Vector<int> &p_polygon) {
	polygons.push_back(p_polygon);
	navigation_polygon_dirty = true;
}

void NavigationPolygon::set_baked_outlines(const TypedArray<Vector<Vector2>> &p_baked_outlines) {
	baked_outlines.resize(p_baked_outlines.size());
	for (int i = 0; i < p_baked_outlines.size(); i++) {
		baked_outlines.write[i] = p_baked_outlines[i];
	}
	navigation_polygon_dirty = true;
}

TypedArray<Vector<Vector2>> NavigationPolygon::get_baked_outlines() const {
	TypedArray<Vector<Vector2>> _typed_baked_outlines;
	_typed_baked_outlines.resize(baked_outlines.size());
	for (int i = 0; i < _typed_baked_outlines.size(); i++) {
		_typed_baked_outlines[i] = baked_outlines[i];
	}

	return _typed_baked_outlines;
}

void NavigationPolygon::add_outline_at_index(const Vector<Vector2> &p_outline, int p_index) {
	outlines.insert(p_index, p_outline);
	internal_set_baked_outlines(outlines);
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

int NavigationPolygon::get_polygon_count() const {
	return polygons.size();
}

Vector<int> NavigationPolygon::get_polygon(int p_index) {
	ERR_FAIL_INDEX_V(p_index, polygons.size(), Vector<int>());
	return polygons[p_index];
}

void NavigationPolygon::clear_polygons() {
	polygons.clear();
	navigation_polygon_dirty = true;
}

void NavigationPolygon::add_outline(const Vector<Vector2> &p_outline) {
	outlines.push_back(p_outline);
	internal_set_baked_outlines(outlines);
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

int NavigationPolygon::get_outline_count() const {
	return outlines.size();
}

void NavigationPolygon::set_outline(int p_index, const Vector<Vector2> &p_outline) {
	ERR_FAIL_INDEX(p_index, outlines.size());
	outlines.write[p_index] = p_outline;
	internal_set_baked_outlines(outlines);
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

void NavigationPolygon::remove_outline(int p_index) {
	ERR_FAIL_INDEX(p_index, outlines.size());
	outlines.remove_at(p_index);
	internal_set_baked_outlines(outlines);
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

Vector<Vector2> NavigationPolygon::get_outline(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, outlines.size(), Vector<Vector2>());
	return outlines[p_index];
}

void NavigationPolygon::clear_outlines() {
	outlines.clear();
	baked_outlines.clear();
	rect_cache_dirty = true;
	navigation_polygon_dirty = true;
}

void NavigationPolygon::commit_changes() {
	if (navigation_polygon_dirty) {
		navigation_polygon_dirty = false;

		Vector<Vector3> new_navigation_mesh_vertices;
		Vector<Vector<int>> new_navigation_mesh_polygons;

		const Vector<Vector2> &navigation_polygon_vertices = get_vertices();
		new_navigation_mesh_vertices.resize(navigation_polygon_vertices.size());

		Vector3 *new_navigation_mesh_vertices_ptrw = new_navigation_mesh_vertices.ptrw();
		const Vector2 *navigation_polygon_vertices_ptr = navigation_polygon_vertices.ptr();

		for (int i = 0; i < navigation_polygon_vertices.size(); i++) {
			new_navigation_mesh_vertices_ptrw[i] = Vector3(navigation_polygon_vertices_ptr[i].x, 0.0, navigation_polygon_vertices_ptr[i].y);
		}

		for (int i = 0; i < get_polygon_count(); i++) {
			Vector<int> new_navigation_mesh_polygon = get_polygon(i);
			new_navigation_mesh_polygons.push_back(new_navigation_mesh_polygon);
		}

		navigation_mesh->set_vertices(new_navigation_mesh_vertices);
		navigation_mesh->internal_set_polygons(new_navigation_mesh_polygons);
		navigation_mesh->commit_changes();

		emit_changed();
	}
}

void NavigationPolygon::clear() {
	clear_outlines();
	clear_polygons();
	set_vertices(Vector<Vector2>());
	navigation_polygon_dirty = true;
}

void NavigationPolygon::set_parsed_geometry_type(ParsedGeometryType p_value) {
	ERR_FAIL_INDEX(p_value, PARSED_GEOMETRY_MAX);
	parsed_geometry_type = p_value;
	notify_property_list_changed();
}

NavigationPolygon::ParsedGeometryType NavigationPolygon::get_parsed_geometry_type() const {
	return parsed_geometry_type;
}

void NavigationPolygon::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t NavigationPolygon::get_collision_mask() const {
	return collision_mask;
}

void NavigationPolygon::set_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_collision_mask(mask);
}

bool NavigationPolygon::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

void NavigationPolygon::set_source_geometry_mode(SourceGeometryMode p_geometry_mode) {
	ERR_FAIL_INDEX(p_geometry_mode, SOURCE_GEOMETRY_MAX);
	source_geometry_mode = p_geometry_mode;
	notify_property_list_changed();
}

NavigationPolygon::SourceGeometryMode NavigationPolygon::get_source_geometry_mode() const {
	return source_geometry_mode;
}

void NavigationPolygon::set_polygon_bake_fillrule(PolygonFillRule p_polygon_fillrule) {
	ERR_FAIL_INDEX(p_polygon_fillrule, POLYGON_FILLRULE_MAX);
	polygon_bake_fillrule = p_polygon_fillrule;
	notify_property_list_changed();
}

NavigationPolygon::PolygonFillRule NavigationPolygon::get_polygon_bake_fillrule() const {
	return polygon_bake_fillrule;
}

void NavigationPolygon::set_offsetting_jointype(OffsettingJoinType p_offsetting_jointype) {
	ERR_FAIL_INDEX(p_offsetting_jointype, OFFSETTING_JOINTYPE_MAX);
	offsetting_jointype = p_offsetting_jointype;
	notify_property_list_changed();
}

NavigationPolygon::OffsettingJoinType NavigationPolygon::get_offsetting_jointype() const {
	return offsetting_jointype;
}

void NavigationPolygon::set_source_group_name(StringName p_group_name) {
	source_group_name = p_group_name;
}

StringName NavigationPolygon::get_source_group_name() const {
	return source_group_name;
}

void NavigationPolygon::set_agent_radius(real_t p_value) {
	ERR_FAIL_COND(p_value < 0);
	agent_radius = p_value;
}

real_t NavigationPolygon::get_agent_radius() const {
	return agent_radius;
}

Ref<NavigationMesh> NavigationPolygon::get_navigation_mesh() {
	if (navigation_polygon_dirty) {
		commit_changes();
	}
	return navigation_mesh;
}

void NavigationPolygon::internal_set_polygons(const Vector<Vector<int>> &p_polygons) {
	polygons = p_polygons;
	navigation_polygon_dirty = true;
}

const Vector<Vector<int>> &NavigationPolygon::internal_get_polygons() const {
	return polygons;
}

void NavigationPolygon::internal_set_baked_outlines(const Vector<Vector<Vector2>> &p_baked_outlines) {
	baked_outlines = p_baked_outlines;
	navigation_polygon_dirty = true;
}

const Vector<Vector<Vector2>> &NavigationPolygon::internal_get_baked_outlines() const {
	return baked_outlines;
}

NavigationPolygon::NavigationPolygon() {
	navigation_mesh.instantiate();
	navigation_polygon_dirty = true;
	//call_deferred(SNAME("commit_changes"));
}

NavigationPolygon::~NavigationPolygon() {
}

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
#endif // TOOLS_ENABLED

#ifndef DISABLE_DEPRECATED
void NavigationPolygon::make_polygons_from_outlines() {
	if (outlines.size() == 0) {
		set_vertices(Vector<Vector2>());
		internal_set_polygons(Vector<Vector<int>>());
		commit_changes();
		return;
	}

	WARN_PRINT("Function make_polygons_from_outlines() is deprecated."
			   "\nUse NavigationMeshGenerator bake functions to create polygons instead."
			   "\n NavigationMeshGenerator.bake_2d_from_source_geometry_data() will be called now");
	NavigationMeshGenerator::get_singleton()->bake_2d_from_source_geometry_data(this, Ref<NavigationMeshSourceGeometryData2D>());
	commit_changes();
}
#endif // DISABLE_DEPRECATED
