/**************************************************************************/
/*  navigation_mesh_area_3d.cpp                                           */
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

#include "navigation_mesh_area_3d.h"

#include "core/math/geometry_2d.h"
#include "scene/resources/3d/navigation_mesh_source_geometry_data_3d.h"
#include "scene/resources/navigation_mesh.h"
#include "servers/navigation_server_3d.h"

Callable NavigationMeshArea3D::_navmesh_source_geometry_parsing_callback;
RID NavigationMeshArea3D::_navmesh_source_geometry_parser;

void NavigationMeshArea3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;

#ifdef DEBUG_ENABLED
	update_gizmos();
	_update_debug();
#endif // DEBUG_ENABLED
}

bool NavigationMeshArea3D::is_enabled() const {
	return enabled;
}

void NavigationMeshArea3D::set_height(float p_height) {
	ERR_FAIL_COND_MSG(p_height < 0, "NavigationMeshArea3D height cannot be negative.");
	height = p_height;

	bounds_dirty = true;
	update_gizmos();
#ifdef DEBUG_ENABLED
	_update_debug();
#endif // DEBUG_ENABLED
}

float NavigationMeshArea3D::get_height() const {
	return height;
}

void NavigationMeshArea3D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}

	navigation_layers = p_navigation_layers;
}

uint32_t NavigationMeshArea3D::get_navigation_layers() const {
	return navigation_layers;
}

void NavigationMeshArea3D::set_navigation_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Navigation layer number must be between 1 and 32 inclusive.");
	uint32_t _navigation_layers = get_navigation_layers();
	if (p_value) {
		_navigation_layers |= 1 << (p_layer_number - 1);
	} else {
		_navigation_layers &= ~(1 << (p_layer_number - 1));
	}
	set_navigation_layers(_navigation_layers);
}

bool NavigationMeshArea3D::get_navigation_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Navigation layer number must be between 1 and 32 inclusive.");
	return get_navigation_layers() & (1 << (p_layer_number - 1));
}

void NavigationMeshArea3D::set_priority(int p_priority) {
	priority = p_priority;
}

int NavigationMeshArea3D::get_priority() const {
	return priority;
}

AABB NavigationMeshArea3D::get_bounds() {
	if (bounds_dirty) {
		bounds_dirty = false;
		_update_bounds();
	}
	return bounds;
}

void NavigationMeshArea3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			bounds_dirty = true;
			update_gizmos();
#ifdef DEBUG_ENABLED
			_update_debug();
#endif // DEBUG_ENABLED
		} break;

#ifdef DEBUG_ENABLED
		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_debug();
		} break;
#endif // DEBUG_ENABLED

		case NOTIFICATION_EXIT_TREE: {
		} break;
	}
}

void NavigationMeshAreaBox3D::_update_bounds() {
	Vector<Vector3> vertices;
	vertices.resize(4);
	vertices.write[0] = Vector3(-size.x * 0.5, 0.0, -size.z * 0.5);
	vertices.write[1] = Vector3(size.x * 0.5, 0.0, -size.z * 0.5);
	vertices.write[2] = Vector3(size.x * 0.5, 0.0, size.z * 0.5);
	vertices.write[3] = Vector3(-size.x * 0.5, 0.0, size.z * 0.5);

	const Vector3 gp = is_inside_tree() ? get_global_position() : get_position();
	const Basis basis = is_inside_tree() ? get_global_basis() : get_basis();
	const Vector3 safe_scale = basis.get_scale().abs().maxf(0.001);
	const Transform3D gt = Transform3D(Basis().scaled(safe_scale), gp);

	bounds = _xform_bounds(vertices, gt, height);
}

void NavigationMeshAreaCylinder3D::_update_bounds() {
	Vector<Vector3> vertices;
	vertices.resize(4);
	vertices.write[0] = Vector3(-radius, 0.0, -radius);
	vertices.write[1] = Vector3(radius, 0.0, -radius);
	vertices.write[2] = Vector3(radius, 0.0, radius);
	vertices.write[3] = Vector3(-radius, 0.0, radius);

	const Vector3 gp = is_inside_tree() ? get_global_position() : get_position();
	const Basis basis = is_inside_tree() ? get_global_basis() : get_basis();
	const Vector3 safe_scale = basis.get_scale().abs().maxf(0.001);
	const Transform3D gt = Transform3D(Basis().scaled(safe_scale), gp);

	bounds = _xform_bounds(vertices, gt, height);
}

void NavigationMeshArea3D::_update_bounds() {
}

AABB NavigationMeshArea3D::_xform_bounds(const Vector<Vector3> &p_vertices, const Transform3D &p_gt, float p_height) {
	if (p_vertices.size() == 0) {
		return AABB();
	}

	AABB new_bounds;
	new_bounds.position = p_gt.xform(p_vertices[0]);

	for (const Vector3 &vertex : p_vertices) {
		new_bounds.expand_to(p_gt.xform(vertex));
	}
	const Vector3 height_offset = Vector3(0.0, p_height, 0.0);
	for (const Vector3 &vertex : p_vertices) {
		new_bounds.expand_to(p_gt.xform(vertex + height_offset));
	}

	return new_bounds;
}

void NavigationMeshAreaPolygon3D::_update_bounds() {
	if (get_vertices().is_empty()) {
		bounds = AABB();
		return;
	}

	const Vector3 gp = is_inside_tree() ? get_global_position() : get_position();
	const Basis basis = is_inside_tree() ? get_global_basis() : get_basis();
	const float rotation_y = is_inside_tree() ? get_global_rotation().y : get_rotation().y;
	const Vector3 safe_scale = basis.get_scale().abs().maxf(0.001);
	const Transform3D gt = Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), rotation_y), gp);

	bounds = _xform_bounds(get_vertices(), gt, height);
}

void NavigationMeshArea3D::navmesh_parse_init() {
	ERR_FAIL_NULL(NavigationServer3D::get_singleton());
	if (!_navmesh_source_geometry_parser.is_valid()) {
		_navmesh_source_geometry_parsing_callback = callable_mp_static(&NavigationMeshArea3D::navmesh_parse_source_geometry);
		_navmesh_source_geometry_parser = NavigationServer3D::get_singleton()->source_geometry_parser_create();
		NavigationServer3D::get_singleton()->source_geometry_parser_set_callback(_navmesh_source_geometry_parser, _navmesh_source_geometry_parsing_callback);
	}
}

void NavigationMeshArea3D::navmesh_parse_source_geometry(const Ref<NavigationMesh> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_node) {
	NavigationMeshArea3D *area = Object::cast_to<NavigationMeshArea3D>(p_node);

	if (area == nullptr) {
		return;
	}

	if (!area->is_enabled()) {
		return;
	}

	uint32_t area_navigation_layers = area->get_navigation_layers();
	int area_priority = area->get_priority();

	{
		NavigationMeshAreaBox3D *node = Object::cast_to<NavigationMeshAreaBox3D>(p_node);
		if (node) {
			const Transform3D gt = p_source_geometry_data->root_node_transform * node->get_global_transform();

			const Vector3 size = node->get_size();
			const float height = node->get_height();

			LocalVector<Vector3> b_vertices;
			b_vertices.resize(4);
			b_vertices[0] = Vector3(-size.x * 0.5, 0.0, -size.z * 0.5);
			b_vertices[1] = Vector3(size.x * 0.5, 0.0, -size.z * 0.5);
			b_vertices[2] = Vector3(size.x * 0.5, 0.0, size.z * 0.5);
			b_vertices[3] = Vector3(-size.x * 0.5, 0.0, size.z * 0.5);

			AABB area_bounds;
			area_bounds.position = gt.xform(b_vertices[0]);

			const Vector3 height_offset = Vector3(0.0, height, 0.0);

			for (const Vector3 &vertex : b_vertices) {
				area_bounds.expand_to(gt.xform(vertex));
				area_bounds.expand_to(gt.xform(vertex + height_offset));
			}

			p_source_geometry_data->add_projected_area_box(area_bounds, area_navigation_layers, area_priority);
			return;
		}
	}

	{
		NavigationMeshAreaCylinder3D *node = Object::cast_to<NavigationMeshAreaCylinder3D>(p_node);
		if (node) {
			const Transform3D gt = p_source_geometry_data->root_node_transform * node->get_global_transform();
			const Vector3 position = gt.origin;
			Vector3 safe_scale = gt.basis.get_scale().abs().maxf(0.001);
			if (safe_scale.x > safe_scale.z) {
				safe_scale.z = safe_scale.x;
			}
			if (safe_scale.z > safe_scale.x) {
				safe_scale.x = safe_scale.z;
			}

			p_source_geometry_data->add_projected_area_cylinder(position, safe_scale.x * node->get_radius(), safe_scale.y * node->get_height(), area_navigation_layers, area_priority);
			return;
		}
	}

	{
		NavigationMeshAreaPolygon3D *node = Object::cast_to<NavigationMeshAreaPolygon3D>(p_node);
		if (node) {
			const float elevation = node->get_global_position().y + p_source_geometry_data->root_node_transform.origin.y;
			const Vector3 safe_scale = node->get_global_basis().get_scale().abs().maxf(0.001);
			const Transform3D node_xform = p_source_geometry_data->root_node_transform * Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), node->get_global_rotation().y), node->get_global_position());

			const Vector<Vector3> &area_vertices = node->get_vertices();

			if (area_vertices.is_empty()) {
				return;
			}

			Vector<Vector3> xformed_area_vertices;
			xformed_area_vertices.resize(area_vertices.size());

			const Vector3 *area_vertices_ptr = area_vertices.ptr();
			Vector3 *xformed_area_vertices_ptrw = xformed_area_vertices.ptrw();

			for (int i = 0; i < area_vertices.size(); i++) {
				xformed_area_vertices_ptrw[i] = node_xform.xform(area_vertices_ptr[i]);
				xformed_area_vertices_ptrw[i].y = 0.0;
			}
			p_source_geometry_data->add_projected_area_polygon(xformed_area_vertices, elevation, safe_scale.y * node->get_height(), area_navigation_layers, area_priority);
			return;
		}
	}
}

#ifdef DEBUG_ENABLED
void NavigationMeshArea3D::_update_debug() {
}

void NavigationMeshArea3D::_clear_debug() {
	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);
	rs->mesh_clear(debug_mesh_rid);
	rs->instance_set_scenario(debug_instance_rid, RID());
}
#endif // DEBUG_ENABLED

void NavigationMeshArea3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationMeshArea3D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationMeshArea3D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &NavigationMeshArea3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &NavigationMeshArea3D::get_height);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationMeshArea3D::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationMeshArea3D::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_navigation_layer_value", "layer_number", "value"), &NavigationMeshArea3D::set_navigation_layer_value);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_value", "layer_number"), &NavigationMeshArea3D::get_navigation_layer_value);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &NavigationMeshArea3D::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &NavigationMeshArea3D::get_priority);

	ClassDB::bind_method(D_METHOD("get_bounds"), &NavigationMeshArea3D::get_bounds);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.0,64.0,0.01,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "0,100000,1,or_greater,or_less"), "set_priority", "get_priority");
}

NavigationMeshArea3D::NavigationMeshArea3D() {
#ifdef TOOLS_ENABLED
	set_notify_transform(true);
#endif // TOOLS_ENABLED

#ifdef DEBUG_ENABLED
	RenderingServer *rs = RenderingServer::get_singleton();
	debug_mesh_rid = rs->mesh_create();
	debug_instance_rid = rs->instance_create();
	rs->instance_set_base(debug_instance_rid, debug_mesh_rid);
#endif // DEBUG_ENABLED
}

NavigationMeshArea3D::~NavigationMeshArea3D() {
#ifdef DEBUG_ENABLED
	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);
	if (debug_instance_rid.is_valid()) {
		rs->free(debug_instance_rid);
		debug_instance_rid = RID();
	}
	if (debug_mesh_rid.is_valid()) {
		rs->free(debug_mesh_rid);
		debug_mesh_rid = RID();
	}
#endif // DEBUG_ENABLED
}

void NavigationMeshAreaBox3D::set_size(const Vector3 &p_size) {
	ERR_FAIL_COND_MSG(p_size.x < 0 || p_size.y < 0 || p_size.z < 0, "NavigationMeshAreaBox3D size cannot be negative.");
	size = p_size;

	bounds_dirty = true;
	update_gizmos();
#ifdef DEBUG_ENABLED
	_update_debug();
#endif // DEBUG_ENABLED
}

const Vector3 &NavigationMeshAreaBox3D::get_size() const {
	return size;
}

#ifdef DEBUG_ENABLED
void NavigationMeshAreaBox3D::_update_debug() {
	if (Engine::get_singleton()->is_editor_hint()) {
		// Editor gizmo handles debug.
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	NavigationServer3D *ns3d = NavigationServer3D::get_singleton();

	if (!is_inside_tree() || (!ns3d->get_debug_enabled() || !ns3d->get_debug_navigation_enabled())) {
		rs->mesh_clear(debug_mesh_rid);
		rs->instance_set_scenario(debug_instance_rid, RID());
		return;
	}

	rs->instance_set_visible(debug_instance_rid, is_visible_in_tree());
	rs->instance_set_scenario(debug_instance_rid, get_world_3d()->get_scenario());
	const Vector3 safe_scale = get_global_basis().get_scale().abs().maxf(0.001);
	rs->instance_set_transform(debug_instance_rid, Transform3D(Basis().scaled(safe_scale), get_global_position()));

	rs->mesh_clear(debug_mesh_rid);

	Vector<Vector3> edge_vertex_array;
	edge_vertex_array.resize(12 * 2); // 12 edges, 2 points per edge.

	Vector3 *edge_vertex_array_ptrw = edge_vertex_array.ptrw();
	int vertex_index = 0;

	AABB aabb;
	aabb.position = -size / 2;
	aabb.size = size;
	aabb.position.y = 0.0;
	aabb.size.y = get_height();

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		edge_vertex_array_ptrw[vertex_index++] = a;
		edge_vertex_array_ptrw[vertex_index++] = b;
	}

	Array edge_mesh_array;
	edge_mesh_array.resize(Mesh::ARRAY_MAX);
	edge_mesh_array[Mesh::ARRAY_VERTEX] = edge_vertex_array;

	rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINES, edge_mesh_array);

	Ref<StandardMaterial3D> edge_material;

	if (is_enabled()) {
		edge_material = ns3d->get_debug_area_edge_material();
	} else {
		edge_material = ns3d->get_debug_area_edge_disabled_material();
	}
	rs->instance_set_surface_override_material(debug_instance_rid, 0, edge_material->get_rid());
}
#endif // DEBUG_ENABLED

PackedStringArray NavigationMeshAreaBox3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (get_global_rotation().x != 0.0 || get_global_rotation().z != 0.0) {
		warnings.push_back(RTR("NavigationMeshAreaBox3D is a Plane projected object. It is fully axis-aligned like an AABB and can not be rotated."));
	}

	const Vector3 global_scale = get_global_basis().get_scale();
	if (global_scale.x < 0.001 || global_scale.y < 0.001 || global_scale.z < 0.001) {
		warnings.push_back(RTR("NavigationMeshAreaBox3D does not support negative or zero scaling."));
	}

	if (size.x < 0.0 || size.y < 0.0 || size.z < 0.0) {
		warnings.push_back(RTR("NavigationMeshAreaBox3D does not support negative size."));
	}

	return warnings;
}

void NavigationMeshAreaBox3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &NavigationMeshAreaBox3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &NavigationMeshAreaBox3D::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
}

NavigationMeshAreaBox3D::NavigationMeshAreaBox3D() {
}

NavigationMeshAreaBox3D::~NavigationMeshAreaBox3D() {
}

void NavigationMeshAreaCylinder3D::set_radius(float p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0, "NavigationMeshAreaCylinder3D radius cannot be negative.");
	radius = p_radius;

	bounds_dirty = true;
	update_gizmos();

#ifdef DEBUG_ENABLED
	_update_debug();
#endif // DEBUG_ENABLED
}

float NavigationMeshAreaCylinder3D::get_radius() const {
	return radius;
}

#ifdef DEBUG_ENABLED
void NavigationMeshAreaCylinder3D::_update_debug() {
	if (Engine::get_singleton()->is_editor_hint()) {
		// Editor gizmo handles debug.
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	NavigationServer3D *ns3d = NavigationServer3D::get_singleton();

	if (!is_inside_tree() || (!ns3d->get_debug_enabled() || !ns3d->get_debug_navigation_enabled())) {
		rs->mesh_clear(debug_mesh_rid);
		rs->instance_set_scenario(debug_instance_rid, RID());
		return;
	}

	rs->instance_set_visible(debug_instance_rid, is_visible_in_tree());
	rs->instance_set_scenario(debug_instance_rid, get_world_3d()->get_scenario());
	Vector3 safe_scale = get_global_basis().get_scale().abs().maxf(0.001);
	if (safe_scale.x > safe_scale.z) {
		safe_scale.z = safe_scale.x;
	}
	if (safe_scale.z > safe_scale.x) {
		safe_scale.x = safe_scale.z;
	}
	rs->instance_set_transform(debug_instance_rid, Transform3D(Basis().scaled(safe_scale), get_global_position()));

	rs->mesh_clear(debug_mesh_rid);

	float c_radius = get_radius();
	float c_height = get_height();

	Vector<Vector3> edge_vertex_array;

	const uint32_t points_in_octant = 4;
	real_t inc = (Math_PI / (4 * points_in_octant));

	// points_in_octant * 8 * 2 per circle * 2 circles + 2*points_in_octant for vertical edge for each octant.
	edge_vertex_array.resize(points_in_octant * 8 * 2 * 2 + 2 * points_in_octant);
	uint32_t index = 0;
	Vector3 *lines_ptrw = edge_vertex_array.ptrw();

	real_t c_radius_squared = c_radius * c_radius;

	real_t r = 0;
	Vector2 a = Vector2(c_radius, 0);
	for (uint32_t i = 0; i < points_in_octant; i++) {
		r += inc;
		real_t x = Math::cos(r) * c_radius;
		real_t y = Math::sqrt(c_radius_squared - (x * x));

#define PUSH_OCTANT(_position, a, b)                          \
	lines_ptrw[index++] = _position + Vector3(a.x, b, a.y);   \
	lines_ptrw[index++] = _position + Vector3(x, b, y);       \
	lines_ptrw[index++] = _position + Vector3(-a.x, b, a.y);  \
	lines_ptrw[index++] = _position + Vector3(-x, b, y);      \
	lines_ptrw[index++] = _position + Vector3(a.x, b, -a.y);  \
	lines_ptrw[index++] = _position + Vector3(x, b, -y);      \
	lines_ptrw[index++] = _position + Vector3(-a.x, b, -a.y); \
	lines_ptrw[index++] = _position + Vector3(-x, b, -y);     \
	lines_ptrw[index++] = _position + Vector3(a.y, b, a.x);   \
	lines_ptrw[index++] = _position + Vector3(y, b, x);       \
	lines_ptrw[index++] = _position + Vector3(-a.y, b, a.x);  \
	lines_ptrw[index++] = _position + Vector3(-y, b, x);      \
	lines_ptrw[index++] = _position + Vector3(a.y, b, -a.x);  \
	lines_ptrw[index++] = _position + Vector3(y, b, -x);      \
	lines_ptrw[index++] = _position + Vector3(-a.y, b, -a.x); \
	lines_ptrw[index++] = _position + Vector3(-y, b, -x);

		PUSH_OCTANT(Vector3(), a, 0.0)
		PUSH_OCTANT(Vector3(), a, c_height)
#undef PUSH_OCTANT

		a.x = x;
		a.y = y;
	}

	lines_ptrw[index++] = Vector3(-c_radius, 0, 0.0);
	lines_ptrw[index++] = Vector3(-c_radius, c_height, 0.0);
	lines_ptrw[index++] = Vector3(c_radius, 0, 0.0);
	lines_ptrw[index++] = Vector3(c_radius, c_height, 0.0);
	lines_ptrw[index++] = Vector3(0.0, 0, -c_radius);
	lines_ptrw[index++] = Vector3(0.0, c_height, -c_radius);
	lines_ptrw[index++] = Vector3(0.0, 0, c_radius);
	lines_ptrw[index++] = Vector3(0.0, c_height, c_radius);

	Array edge_mesh_array;
	edge_mesh_array.resize(Mesh::ARRAY_MAX);
	edge_mesh_array[Mesh::ARRAY_VERTEX] = edge_vertex_array;

	rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINES, edge_mesh_array);

	Ref<StandardMaterial3D> edge_material;

	if (is_enabled()) {
		edge_material = ns3d->get_debug_area_edge_material();
	} else {
		edge_material = ns3d->get_debug_area_edge_disabled_material();
	}
	rs->instance_set_surface_override_material(debug_instance_rid, 0, edge_material->get_rid());
}
#endif // DEBUG_ENABLED

PackedStringArray NavigationMeshAreaCylinder3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (get_global_rotation().x != 0.0 || get_global_rotation().z != 0.0) {
		warnings.push_back(RTR("NavigationMeshAreaCylinder3D is a Plane projected object. It only takes global rotation around the y-axis into account. Rotations around the x-axis or z-axis are ignored."));
	}

	const Vector3 global_scale = get_global_basis().get_scale();
	if (global_scale.x < 0.001 || global_scale.y < 0.001 || global_scale.z < 0.001) {
		warnings.push_back(RTR("NavigationMeshAreaCylinder3D does not support negative or zero scaling."));
	}

	if (radius > 0.0 && global_scale.x != global_scale.z) {
		warnings.push_back(RTR("NavigationMeshAreaCylinder3D radius can only be scaled uniformly. The largest scale value along the horizontal axes will be used."));
	}

	return warnings;
}

void NavigationMeshAreaCylinder3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationMeshAreaCylinder3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationMeshAreaCylinder3D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.0,100,0.01,or_greater,suffix:m"), "set_radius", "get_radius");
}

NavigationMeshAreaCylinder3D::NavigationMeshAreaCylinder3D() {
}

NavigationMeshAreaCylinder3D::~NavigationMeshAreaCylinder3D() {
}

void NavigationMeshAreaPolygon3D::set_vertices(const Vector<Vector3> &p_vertices) {
	vertices = p_vertices;

	bounds_dirty = true;

	Vector<Vector2> vertices_2d;
	vertices_2d.resize(vertices.size());

	const Vector3 *vertices_ptr = vertices.ptr();
	Vector2 *vertices_2d_ptrw = vertices_2d.ptrw();

	for (int i = 0; i < vertices.size(); i++) {
		vertices_2d_ptrw[i].x = vertices_ptr[i].x;
		vertices_2d_ptrw[i].y = vertices_ptr[i].z;
	}

	vertices_are_clockwise = !Geometry2D::is_polygon_clockwise(vertices_2d); // Geometry2D is inverted. A true legacy gift that keeps on giving.
	vertices_are_valid = !Geometry2D::triangulate_polygon(vertices_2d).is_empty();

	update_gizmos();

#ifdef DEBUG_ENABLED
	_update_debug();
#endif // DEBUG_ENABLED
}

const Vector<Vector3> &NavigationMeshAreaPolygon3D::get_vertices() const {
	return vertices;
}

bool NavigationMeshAreaPolygon3D::are_vertices_clockwise() const {
	return vertices_are_clockwise;
}

bool NavigationMeshAreaPolygon3D::are_vertices_valid() const {
	return vertices_are_valid;
}

#ifdef DEBUG_ENABLED
void NavigationMeshAreaPolygon3D::_update_debug() {
	if (Engine::get_singleton()->is_editor_hint()) {
		// Editor gizmo handles debug.
		return;
	}

	RenderingServer *rs = RenderingServer::get_singleton();
	NavigationServer3D *ns3d = NavigationServer3D::get_singleton();

	if (!is_inside_tree() || (!ns3d->get_debug_enabled() || !ns3d->get_debug_navigation_enabled())) {
		rs->mesh_clear(debug_mesh_rid);
		rs->instance_set_scenario(debug_instance_rid, RID());
		return;
	}

	rs->instance_set_visible(debug_instance_rid, is_visible_in_tree());
	rs->instance_set_scenario(debug_instance_rid, get_world_3d()->get_scenario());
	const Basis safe_basis = Basis(Vector3(0.0, 1.0, 0.0), get_global_rotation().y, get_global_basis().get_scale().abs().maxf(0.001));
	rs->instance_set_transform(debug_instance_rid, Transform3D(safe_basis, get_global_position()));

	rs->mesh_clear(debug_mesh_rid);

	const int vertex_count = vertices.size();

	if (vertex_count < 3) {
		return;
	}

	Vector<Vector3> edge_vertex_array;
	edge_vertex_array.resize(vertex_count * 6);

	Vector3 *edge_vertex_array_ptrw = edge_vertex_array.ptrw();

	int vertex_index = 0;

	for (int i = 0; i < vertex_count; i++) {
		Vector3 point = vertices[i];
		Vector3 next_point = vertices[(i + 1) % vertex_count];

		edge_vertex_array_ptrw[vertex_index++] = point;
		edge_vertex_array_ptrw[vertex_index++] = next_point;

		edge_vertex_array_ptrw[vertex_index++] = Vector3(point.x, height, point.z);
		edge_vertex_array_ptrw[vertex_index++] = Vector3(next_point.x, height, next_point.z);

		edge_vertex_array_ptrw[vertex_index++] = point;
		edge_vertex_array_ptrw[vertex_index++] = Vector3(point.x, height, point.z);
	}

	Array edge_mesh_array;
	edge_mesh_array.resize(Mesh::ARRAY_MAX);
	edge_mesh_array[Mesh::ARRAY_VERTEX] = edge_vertex_array;

	rs->mesh_add_surface_from_arrays(debug_mesh_rid, RS::PRIMITIVE_LINES, edge_mesh_array);

	Ref<StandardMaterial3D> edge_material;

	if (are_vertices_valid() && is_enabled()) {
		edge_material = ns3d->get_debug_area_edge_material();
	} else if (are_vertices_valid() && !is_enabled()) {
		edge_material = ns3d->get_debug_area_edge_disabled_material();
	} else {
		edge_material = ns3d->get_debug_area_edge_invalid_material();
	}

	rs->instance_set_surface_override_material(debug_instance_rid, 0, edge_material->get_rid());
}
#endif // DEBUG_ENABLED

PackedStringArray NavigationMeshAreaPolygon3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (get_global_rotation().x != 0.0 || get_global_rotation().z != 0.0) {
		warnings.push_back(RTR("NavigationMeshAreaPolygon3D is a plane projected shape. It only takes global rotation around the y-axis into account. Rotations around the x-axis or z-axis might lead to unexpected results."));
	}

	const Vector3 global_scale = get_global_basis().get_scale();
	if (global_scale.x < 0.001 || global_scale.y < 0.001 || global_scale.z < 0.001) {
		warnings.push_back(RTR("NavigationMeshAreaPolygon3D does not support negative or zero scaling."));
	}

	if (!are_vertices_valid()) {
		warnings.push_back(RTR("NavigationMeshAreaPolygon3D vertices are invalid for triangulation. This is commonly caused by duplicated vertices or self-intersecting edges and might lead to unexpected results."));
	}

	return warnings;
}

void NavigationMeshAreaPolygon3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationMeshAreaPolygon3D::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationMeshAreaPolygon3D::get_vertices);

	ClassDB::bind_method(D_METHOD("are_vertices_clockwise"), &NavigationMeshAreaPolygon3D::are_vertices_clockwise);
	ClassDB::bind_method(D_METHOD("are_vertices_valid"), &NavigationMeshAreaPolygon3D::are_vertices_valid);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR3_ARRAY, "vertices"), "set_vertices", "get_vertices");
}

NavigationMeshAreaPolygon3D::NavigationMeshAreaPolygon3D() {
}

NavigationMeshAreaPolygon3D::~NavigationMeshAreaPolygon3D() {
}
