/**************************************************************************/
/*  openxr_spatial_plane_tracking.cpp                                     */
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

#include "openxr_spatial_plane_tracking.h"

#include "../../openxr_api.h"
#include "core/config/project_settings.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "servers/xr/xr_server.h"

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialCapabilityConfigurationPlaneTracking

void OpenXRSpatialCapabilityConfigurationPlaneTracking::_bind_methods() {
	ClassDB::bind_method(D_METHOD("supports_mesh_2d"), &OpenXRSpatialCapabilityConfigurationPlaneTracking::get_supports_mesh_2d);
	ClassDB::bind_method(D_METHOD("supports_polygons"), &OpenXRSpatialCapabilityConfigurationPlaneTracking::get_supports_polygons);
	ClassDB::bind_method(D_METHOD("supports_labels"), &OpenXRSpatialCapabilityConfigurationPlaneTracking::get_supports_labels);

	ClassDB::bind_method(D_METHOD("get_enabled_components"), &OpenXRSpatialCapabilityConfigurationPlaneTracking::_get_enabled_components);
}

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::has_valid_configuration() const {
	OpenXRSpatialPlaneTrackingCapability *capability = OpenXRSpatialPlaneTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, false);

	return capability->is_supported();
}

XrSpatialCapabilityConfigurationBaseHeaderEXT *OpenXRSpatialCapabilityConfigurationPlaneTracking::get_configuration() {
	OpenXRSpatialPlaneTrackingCapability *capability = OpenXRSpatialPlaneTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, nullptr);

	if (capability->is_supported()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, nullptr);

		// Guaranteed components:
		plane_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_BOUNDED_2D_EXT);
		plane_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_PLANE_ALIGNMENT_EXT);

		// Optional components:
		if (get_supports_mesh_2d()) {
			plane_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_MESH_2D_EXT);
		} else if (get_supports_polygons()) {
			plane_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_POLYGON_2D_EXT);
		}
		if (get_supports_labels()) {
			plane_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_PLANE_SEMANTIC_LABEL_EXT);
		}

		// Set up our enabled components.
		plane_config.enabledComponentCount = plane_enabled_components.size();
		plane_config.enabledComponents = plane_enabled_components.ptr();

		// and return this.
		return (XrSpatialCapabilityConfigurationBaseHeaderEXT *)&plane_config;
	}

	return nullptr;
}

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::get_supports_mesh_2d() {
	if (supports_mesh_2d == -1) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, false);

		supports_mesh_2d = se_extension->supports_component_type(XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT, XR_SPATIAL_COMPONENT_TYPE_MESH_2D_EXT) ? 1 : 0;
	}

	return supports_mesh_2d == 1;
}

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::get_supports_polygons() {
	if (supports_polygons == -1) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, false);

		supports_polygons = se_extension->supports_component_type(XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT, XR_SPATIAL_COMPONENT_TYPE_POLYGON_2D_EXT) ? 1 : 0;
	}

	return supports_polygons == 1;
}

bool OpenXRSpatialCapabilityConfigurationPlaneTracking::get_supports_labels() {
	if (supports_labels == -1) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, false);

		supports_labels = se_extension->supports_component_type(XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT, XR_SPATIAL_COMPONENT_TYPE_PLANE_SEMANTIC_LABEL_EXT) ? 1 : 0;
	}

	return supports_labels == 1;
}

PackedInt64Array OpenXRSpatialCapabilityConfigurationPlaneTracking::_get_enabled_components() const {
	PackedInt64Array components;

	for (const XrSpatialComponentTypeEXT &component_type : plane_enabled_components) {
		components.push_back((int64_t)component_type);
	}

	return components;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentPlaneAlignmentList

void OpenXRSpatialComponentPlaneAlignmentList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_plane_alignment", "index"), &OpenXRSpatialComponentPlaneAlignmentList::_get_plane_alignment);

	BIND_ENUM_CONSTANT(PLANE_ALIGNMENT_HORIZONTAL_UPWARD);
	BIND_ENUM_CONSTANT(PLANE_ALIGNMENT_HORIZONTAL_DOWNWARD);
	BIND_ENUM_CONSTANT(PLANE_ALIGNMENT_VERTICAL);
	BIND_ENUM_CONSTANT(PLANE_ALIGNMENT_ARBITRARY);
}

void OpenXRSpatialComponentPlaneAlignmentList::set_capacity(uint32_t p_capacity) {
	plane_alignment_data.resize(p_capacity);

	plane_alignment_list.planeAlignmentCount = uint32_t(plane_alignment_data.size());
	plane_alignment_list.planeAlignments = plane_alignment_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentPlaneAlignmentList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_PLANE_ALIGNMENT_EXT;
}

void *OpenXRSpatialComponentPlaneAlignmentList::get_structure_data(void *p_next) {
	plane_alignment_list.next = p_next;
	return &plane_alignment_list;
}

XrSpatialPlaneAlignmentEXT OpenXRSpatialComponentPlaneAlignmentList::get_plane_alignment(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, plane_alignment_data.size(), XR_SPATIAL_PLANE_ALIGNMENT_MAX_ENUM_EXT);

	return plane_alignment_data[p_index];
}

OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment OpenXRSpatialComponentPlaneAlignmentList::_get_plane_alignment(int64_t p_index) const {
	return (PlaneAlignment)get_plane_alignment(p_index);
}

////////////////////////////////////////////////////////////////////////////
// Spatial component polygon2d list

void OpenXRSpatialComponentPolygon2DList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_transform", "index"), &OpenXRSpatialComponentPolygon2DList::get_transform);
	ClassDB::bind_method(D_METHOD("get_vertices", "snapshot", "index"), &OpenXRSpatialComponentPolygon2DList::get_vertices);
}

void OpenXRSpatialComponentPolygon2DList::set_capacity(uint32_t p_capacity) {
	polygon2d_data.resize(p_capacity);

	polygon2d_list.polygonCount = uint32_t(polygon2d_data.size());
	polygon2d_list.polygons = polygon2d_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentPolygon2DList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_POLYGON_2D_EXT;
}

void *OpenXRSpatialComponentPolygon2DList::get_structure_data(void *p_next) {
	polygon2d_list.next = p_next;
	return &polygon2d_list;
}

Transform3D OpenXRSpatialComponentPolygon2DList::get_transform(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, polygon2d_data.size(), Transform3D());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	return openxr_api->transform_from_pose(polygon2d_data[p_index].origin);
}

PackedVector2Array OpenXRSpatialComponentPolygon2DList::get_vertices(RID p_snapshot, int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, polygon2d_data.size(), PackedVector2Array());

	const XrSpatialBufferEXT &buffer = polygon2d_data[p_index].vertexBuffer;
	if (buffer.bufferId == XR_NULL_SPATIAL_BUFFER_ID_EXT) {
		// We don't have data (yet).
		return PackedVector2Array();
	}

	ERR_FAIL_COND_V(buffer.bufferType != XR_SPATIAL_BUFFER_TYPE_VECTOR2F_EXT, PackedVector2Array());

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, PackedVector2Array());

	return se_extension->get_vector2_buffer(p_snapshot, buffer.bufferId);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentPlaneSemanticLabelList

void OpenXRSpatialComponentPlaneSemanticLabelList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_plane_semantic_label", "index"), &OpenXRSpatialComponentPlaneSemanticLabelList::_get_plane_semantic_label);

	BIND_ENUM_CONSTANT(PLANE_SEMANTIC_LABEL_UNCATEGORIZED);
	BIND_ENUM_CONSTANT(PLANE_SEMANTIC_LABEL_FLOOR);
	BIND_ENUM_CONSTANT(PLANE_SEMANTIC_LABEL_WALL);
	BIND_ENUM_CONSTANT(PLANE_SEMANTIC_LABEL_CEILING);
	BIND_ENUM_CONSTANT(PLANE_SEMANTIC_LABEL_TABLE);
}

void OpenXRSpatialComponentPlaneSemanticLabelList::set_capacity(uint32_t p_capacity) {
	plane_semantic_label_data.resize(p_capacity);

	plane_semantic_label_list.semanticLabelCount = uint32_t(plane_semantic_label_data.size());
	plane_semantic_label_list.semanticLabels = plane_semantic_label_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentPlaneSemanticLabelList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_PLANE_SEMANTIC_LABEL_EXT;
}

void *OpenXRSpatialComponentPlaneSemanticLabelList::get_structure_data(void *p_next) {
	plane_semantic_label_list.next = p_next;
	return &plane_semantic_label_list;
}

XrSpatialPlaneSemanticLabelEXT OpenXRSpatialComponentPlaneSemanticLabelList::get_plane_semantic_label(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, plane_semantic_label_data.size(), XR_SPATIAL_PLANE_SEMANTIC_LABEL_MAX_ENUM_EXT);

	return plane_semantic_label_data[p_index];
}

OpenXRSpatialComponentPlaneSemanticLabelList::PlaneSemanticLabel OpenXRSpatialComponentPlaneSemanticLabelList::_get_plane_semantic_label(int64_t p_index) const {
	return (PlaneSemanticLabel)get_plane_semantic_label(p_index);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRPlaneTracker

void OpenXRPlaneTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bounds_size", "bounds_size"), &OpenXRPlaneTracker::set_bounds_size);
	ClassDB::bind_method(D_METHOD("get_bounds_size"), &OpenXRPlaneTracker::get_bounds_size);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bounds_size"), "set_bounds_size", "get_bounds_size");

	ClassDB::bind_method(D_METHOD("set_plane_alignment", "plane_alignment"), &OpenXRPlaneTracker::set_plane_alignment);
	ClassDB::bind_method(D_METHOD("get_plane_alignment"), &OpenXRPlaneTracker::get_plane_alignment);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "plane_alignment"), "set_plane_alignment", "get_plane_alignment");

	ClassDB::bind_method(D_METHOD("set_plane_label", "plane_label"), &OpenXRPlaneTracker::set_plane_label);
	ClassDB::bind_method(D_METHOD("get_plane_label"), &OpenXRPlaneTracker::get_plane_label);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "plane_label"), "set_plane_label", "get_plane_label");

	ClassDB::bind_method(D_METHOD("set_mesh_data", "origin", "vertices", "indices"), &OpenXRPlaneTracker::set_mesh_data, DEFVAL(PackedInt32Array()));
	ClassDB::bind_method(D_METHOD("clear_mesh_data"), &OpenXRPlaneTracker::clear_mesh_data);

	ClassDB::bind_method(D_METHOD("get_mesh_offset"), &OpenXRPlaneTracker::get_mesh_offset);
	ClassDB::bind_method(D_METHOD("get_mesh"), &OpenXRPlaneTracker::get_mesh);
	ClassDB::bind_method(D_METHOD("get_shape", "thickness"), &OpenXRPlaneTracker::get_shape, DEFVAL(0.01));

	ADD_SIGNAL(MethodInfo("mesh_changed"));
}

void OpenXRPlaneTracker::set_bounds_size(const Vector2 &p_bounds_size) {
	if (Math::abs(bounds_size.x - p_bounds_size.x) > 0.001 || Math::abs(bounds_size.y - p_bounds_size.y) > 0.001) {
		bounds_size = p_bounds_size;

		if (!mesh.has_mesh_data) {
			// Bounds changing only effects mesh data if we don't have polygon data.
			clear_mesh_data();
			emit_signal(SNAME("mesh_changed"));
		}
	}
}

Vector2 OpenXRPlaneTracker::get_bounds_size() const {
	return bounds_size;
}

void OpenXRPlaneTracker::set_plane_alignment(OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment p_plane_alignment) {
	if (plane_alignment != p_plane_alignment) {
		plane_alignment = p_plane_alignment;
	}
}

OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment OpenXRPlaneTracker::get_plane_alignment() const {
	return plane_alignment;
}

void OpenXRPlaneTracker::set_plane_label(const String &p_plane_label) {
	if (plane_label != p_plane_label) {
		plane_label = p_plane_label;

		// Also copy to description, should do something nicer here.
		set_tracker_desc(plane_label);
	}
}

String OpenXRPlaneTracker::get_plane_label() const {
	return plane_label;
}

void OpenXRPlaneTracker::set_mesh_data(const Transform3D &p_origin, const PackedVector2Array &p_vertices, const PackedInt32Array &p_indices) {
	if (p_vertices.size() < 3) {
		if (mesh.has_mesh_data) {
			clear_mesh_data();
			emit_signal(SNAME("mesh_changed"));
		}
	} else {
		bool has_changed = !mesh.has_mesh_data;

		mesh.has_mesh_data = true;
		mesh.origin = p_origin;

		if (mesh.vertices.size() != p_vertices.size()) {
			has_changed = true;
		} else {
			// Compare the vertices with a bit of margin, we ignore small jittering on vertices.
			for (uint32_t i = 0; i < p_vertices.size() && !has_changed; i++) {
				const Vector2 &a = p_vertices[i];
				const Vector2 &b = mesh.vertices[i];
				has_changed = (Math::abs(a.x - b.x) > 0.001) || (Math::abs(a.y - b.y) > 0.001);
			}
		}
		if (has_changed) {
			mesh.vertices = p_vertices;
		}

		// Q: Should we keep our indices list empty if we get polygon data
		// and create different meshes/collision shapes as a result?
		if (p_indices.is_empty()) {
			// Assume polygon, turn into triangle strip...
			int count = (p_vertices.size() - 2) * 3;

			// If our vertices haven't changed and our indices are already the correct size,
			// assume we don't need to rerun this.
			if (has_changed || mesh.indices.size() != count) {
				has_changed = true;

				int offset = 1;
				mesh.indices.resize(count);
				int32_t *idx = mesh.indices.ptrw();
				for (int i = 0; i < count; i += 3) {
					idx[i + 0] = 0;
					idx[i + 2] = offset++;
					idx[i + 1] = offset;
				}
			}
		} else {
			if (mesh.indices.size() != p_indices.size()) {
				has_changed = true;
			} else {
				for (uint32_t i = 0; i < p_indices.size() && !has_changed; i++) {
					has_changed = mesh.indices[i] != p_indices[i];
				}
			}
			if (has_changed) {
				mesh.indices = p_indices;
			}
		}

		if (has_changed) {
			mesh.mesh.unref();
			mesh.shape3d.unref();

			emit_signal(SNAME("mesh_changed"));
		}
	}
}

void OpenXRPlaneTracker::clear_mesh_data() {
	mesh.mesh.unref();
	mesh.shape3d.unref();

	if (mesh.has_mesh_data) {
		mesh.has_mesh_data = false;
		mesh.origin = Transform3D();
		mesh.vertices.clear();
		mesh.indices.clear();

		emit_signal(SNAME("mesh_changed"));
	}
}

Transform3D OpenXRPlaneTracker::get_mesh_offset() const {
	Transform3D offset;

	if (mesh.has_mesh_data) {
		offset = mesh.origin;

		Ref<XRPose> pose = get_pose(SNAME("default"));
		if (pose.is_valid()) {
			// Q is this offset * transform.inverse?
			offset = pose->get_transform().inverse() * offset;
		}

		// Reference frame will already be applied to pose used on our XRNode3D but we do need to apply our scale
		XRServer *xr_server = XRServer::get_singleton();
		if (xr_server) {
			offset.origin *= xr_server->get_world_scale();
		}
	}

	return offset;
}

Ref<Mesh> OpenXRPlaneTracker::get_mesh() {
	// We've already created this? Just return it!
	if (mesh.mesh.is_valid()) {
		return mesh.mesh;
	}

	if (mesh.has_mesh_data) {
		Ref<ArrayMesh> array_mesh;
		Array arr;

		// We need our vertices as Vector3
		PackedVector3Array vertices;
		vertices.resize(mesh.vertices.size());
		const Vector2 *read = mesh.vertices.ptr();
		Vector3 *write = vertices.ptrw();
		for (int v = 0; v < mesh.vertices.size(); v++) {
			write[v] = Vector3(read[v].x, read[v].y, 0.0);
		}

		// Build our array with data.
		arr.resize(RS::ARRAY_MAX);
		arr[RS::ARRAY_VERTEX] = vertices;
		arr[RS::ARRAY_INDEX] = mesh.indices;

		// Create our array mesh.
		array_mesh.instantiate();
		array_mesh->add_surface_from_arrays(Mesh::PrimitiveType::PRIMITIVE_TRIANGLES, arr);

		// Cache this.
		mesh.mesh = array_mesh;
	} else if (bounds_size.x > 0.0 && bounds_size.y > 0.0) {
		// We can use a plane mesh here.
		Ref<PlaneMesh> plane_mesh;

		plane_mesh.instantiate();
		plane_mesh->set_orientation(PlaneMesh::Orientation::FACE_Z);
		plane_mesh->set_size(bounds_size);

		// Cache this.
		mesh.mesh = plane_mesh;
	} else {
		print_verbose("OpenXR: Can't create mesh for plane, no data.");
	}
	return mesh.mesh;
}

Ref<Shape3D> OpenXRPlaneTracker::get_shape(real_t p_thickness) {
	// We've already created this? Just return it!
	if (mesh.shape3d.is_valid()) {
		return mesh.shape3d;
	}

	if (mesh.has_mesh_data) {
		Ref<ConcavePolygonShape3D> shape;
		Vector<Vector3> faces;

		// Get some direct access to our data.
		int isize = mesh.indices.size();
		const Vector2 *vr = mesh.vertices.ptr();
		const int32_t *ir = mesh.indices.ptr();

		// Find our edges.
		HashMap<Edge, int, Edge> edge_counts;
		for (int i = 0; i < isize; i += 3) {
			for (int j = 0; j < 3; j++) {
				Edge e(ir[i + j], ir[i + ((j + 1) % 3)]);
				edge_counts[e]++;
			}
		}

		// Find our outer edges.
		thread_local LocalVector<Edge> outer_edges;
		outer_edges.clear();
		for (const KeyValue<Edge, int> &e : edge_counts) {
			if (e.value > 1) {
				outer_edges.push_back(e.key);
			}
		}

		// Make space for these.
		faces.resize(2 * isize + 6 * outer_edges.size());
		Vector3 *write = faces.ptrw();

		// Add top and bottom.
		for (int i = 0; i < isize; i += 3) {
			Vector3 a = Vector3(vr[ir[i]].x, vr[ir[i]].y, 0.0);
			Vector3 b = Vector3(vr[ir[i + 1]].x, vr[ir[i + 1]].y, 0.0);
			Vector3 c = Vector3(vr[ir[i + 2]].x, vr[ir[i + 2]].y, 0.0);

			*write++ = a;
			*write++ = b;
			*write++ = c;

			a.z = -p_thickness;
			b.z = -p_thickness;
			c.z = -p_thickness;

			*write++ = a;
			*write++ = c;
			*write++ = b;
		}

		// Add outer edges.
		for (const Edge &edge : outer_edges) {
			Vector3 a = Vector3(vr[edge.a].x, vr[edge.a].y, 0.0);
			Vector3 b = Vector3(vr[edge.b].x, vr[edge.b].y, 0.0);
			Vector3 c = b + Vector3(0.0, 0.0, -p_thickness);
			Vector3 d = a + Vector3(0.0, 0.0, -p_thickness);

			*write++ = a;
			*write++ = b;
			*write++ = c;

			*write++ = a;
			*write++ = c;
			*write++ = d;
		}

		// Create our shape.
		shape.instantiate();
		shape->set_faces(faces);

		mesh.shape3d = shape;
	} else if (bounds_size.x > 0.0 && bounds_size.y > 0.0) {
		// We can use a box shape here
		Ref<BoxShape3D> box_shape;
		box_shape.instantiate();
		box_shape->set_size(Vector3(bounds_size.x, bounds_size.y, p_thickness));

		mesh.shape3d = box_shape;
	}

	return mesh.shape3d;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialPlaneTrackingCapability

OpenXRSpatialPlaneTrackingCapability *OpenXRSpatialPlaneTrackingCapability::singleton = nullptr;

OpenXRSpatialPlaneTrackingCapability *OpenXRSpatialPlaneTrackingCapability::get_singleton() {
	return singleton;
}

OpenXRSpatialPlaneTrackingCapability::OpenXRSpatialPlaneTrackingCapability() {
	singleton = this;
}

OpenXRSpatialPlaneTrackingCapability::~OpenXRSpatialPlaneTrackingCapability() {
	singleton = nullptr;
}

void OpenXRSpatialPlaneTrackingCapability::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_supported"), &OpenXRSpatialPlaneTrackingCapability::is_supported);
}

HashMap<String, bool *> OpenXRSpatialPlaneTrackingCapability::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enabled") && GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enable_plane_tracking")) {
		request_extensions[XR_EXT_SPATIAL_PLANE_TRACKING_EXTENSION_NAME] = &spatial_plane_tracking_ext;
	}

	return request_extensions;
}

void OpenXRSpatialPlaneTrackingCapability::on_session_created(const XrSession p_session) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	if (!spatial_plane_tracking_ext) {
		return;
	}

	spatial_plane_tracking_supported = se_extension->supports_capability(XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT);
	if (!spatial_plane_tracking_supported) {
		// Supported by XR runtime but not by device? We're done.
		return;
	}

	se_extension->connect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_recommended));

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enable_builtin_plane_detection")) {
		// Start by creating our spatial context
		_create_spatial_context();
	}
}

void OpenXRSpatialPlaneTrackingCapability::on_session_destroyed() {
	if (!spatial_plane_tracking_supported) {
		return;
	}
	spatial_plane_tracking_supported = false;

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	// Free and unregister our anchors
	for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRPlaneTracker>> &plane_tracker : plane_trackers) {
		xr_server->remove_tracker(plane_tracker.value);
	}
	plane_trackers.clear();

	// Free our spatial context
	if (spatial_context.is_valid()) {
		se_extension->free_spatial_context(spatial_context);
		spatial_context = RID();
	}

	se_extension->disconnect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_recommended));
}

void OpenXRSpatialPlaneTrackingCapability::on_process() {
	if (!spatial_context.is_valid()) {
		return;
	}

	// Protection against plane discovery happening too often.
	if (discovery_cooldown > 0) {
		discovery_cooldown--;
	}

	// Check if we need to start our discovery.
	if (need_discovery && discovery_cooldown == 0 && !discovery_query_result.is_valid()) {
		need_discovery = false;
		discovery_cooldown = 60; // Set our cooldown to 60 frames, it doesn't need to be an exact science.

		_start_entity_discovery();
	}
}

bool OpenXRSpatialPlaneTrackingCapability::is_supported() {
	return spatial_plane_tracking_supported;
}

////////////////////////////////////////////////////////////////////////////
// Discovery logic
Ref<OpenXRFutureResult> OpenXRSpatialPlaneTrackingCapability::_create_spatial_context() {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);

	TypedArray<OpenXRSpatialCapabilityConfigurationBaseHeader> capability_configurations;

	// Create our configuration objects.
	plane_configuration.instantiate();
	capability_configurations.push_back(plane_configuration);

	return se_extension->create_spatial_context(capability_configurations, nullptr, callable_mp(this, &OpenXRSpatialPlaneTrackingCapability::_on_spatial_context_created));
}

void OpenXRSpatialPlaneTrackingCapability::_on_spatial_context_created(RID p_spatial_context) {
	spatial_context = p_spatial_context;
	need_discovery = true;
}

void OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_recommended(RID p_spatial_context) {
	if (p_spatial_context == spatial_context) {
		// Trigger new discovery.
		need_discovery = true;
	}
}

Ref<OpenXRFutureResult> OpenXRSpatialPlaneTrackingCapability::_start_entity_discovery() {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);

	// Already running or ran discovery, cancel/clean up.
	if (discovery_query_result.is_valid()) {
		WARN_PRINT("OpenXR: Starting new discovery before previous discovery has been processed!");
		discovery_query_result->cancel_future();
		discovery_query_result.unref();
	}

	// Start our new snapshot.
	discovery_query_result = se_extension->discover_spatial_entities(spatial_context, plane_configuration->get_enabled_components(), nullptr, callable_mp(this, &OpenXRSpatialPlaneTrackingCapability::_process_snapshot));

	return discovery_query_result;
}

void OpenXRSpatialPlaneTrackingCapability::_process_snapshot(RID p_snapshot) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Make a copy of the planes we have right now, so we know which ones to clean up.
	LocalVector<XrSpatialEntityIdEXT> current_planes;
	current_planes.resize(plane_trackers.size());
	int p = 0;
	for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRPlaneTracker>> &plane : plane_trackers) {
		current_planes[p++] = plane.key;
	}

	// Build our component data
	TypedArray<OpenXRSpatialComponentData> component_data;

	// We always need a query result data object
	Ref<OpenXRSpatialQueryResultData> query_result_data;
	query_result_data.instantiate();
	component_data.push_back(query_result_data);

	// Add bounded2D
	Ref<OpenXRSpatialComponentBounded2DList> bounded2d_list;
	bounded2d_list.instantiate();
	component_data.push_back(bounded2d_list);

	// Plane alignment list
	Ref<OpenXRSpatialComponentPlaneAlignmentList> alignment_list;
	alignment_list.instantiate();
	component_data.push_back(alignment_list);

	Ref<OpenXRSpatialComponentMesh2DList> mesh2d_list;
	Ref<OpenXRSpatialComponentPolygon2DList> poly2d_list;
	if (plane_configuration->get_supports_mesh_2d()) {
		mesh2d_list.instantiate();
		component_data.push_back(mesh2d_list);
	} else if (plane_configuration->get_supports_polygons()) {
		poly2d_list.instantiate();
		component_data.push_back(poly2d_list);
	}

	// Plane semantic label
	Ref<OpenXRSpatialComponentPlaneSemanticLabelList> label_list;
	if (plane_configuration->get_supports_labels()) {
		label_list.instantiate();
		component_data.push_back(label_list);
	}

	if (se_extension->query_snapshot(p_snapshot, component_data, nullptr)) {
		// Now loop through our data and update our anchors.
		// Q we're assuming entity ID, size and state size are equal, is there ever a situation where they would not be?
		int64_t size = query_result_data->get_capacity();
		for (int64_t i = 0; i < size; i++) {
			XrSpatialEntityIdEXT entity_id = query_result_data->get_entity_id(i);
			XrSpatialEntityTrackingStateEXT entity_state = query_result_data->get_entity_state(i);

			// Erase it from our current planes (if we have it, else this is ignored).
			current_planes.erase(entity_id);

			if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT) {
				// We should only get this status on updates as a prelude to needing to remove this marker.
				// So we just update the status.
				if (plane_trackers.has(entity_id)) {
					Ref<OpenXRPlaneTracker> plane_tracker = plane_trackers[entity_id];
					plane_tracker->invalidate_pose(SNAME("default"));
					plane_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT);
				}
			} else {
				// Process our entity
				bool add_to_xr_server = false;
				Ref<OpenXRPlaneTracker> plane_tracker;

				if (plane_trackers.has(entity_id)) {
					// We know about this one already
					plane_tracker = plane_trackers[entity_id];
				} else {
					// Create a new anchor
					plane_tracker.instantiate();
					plane_tracker->set_entity(se_extension->make_spatial_entity(se_extension->get_spatial_snapshot_context(p_snapshot), entity_id));
					plane_trackers[entity_id] = plane_tracker;

					add_to_xr_server = true;
				}

				// Handle component data
				if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT) {
					plane_tracker->invalidate_pose(SNAME("default"));
					plane_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT);

					// No further component data will be valid in this state, we need to ignore it!
				} else if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT) {
					Transform3D transform = bounded2d_list->get_center_pose(i);
					plane_tracker->set_pose(SNAME("default"), transform, Vector3(), Vector3());
					plane_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT);

					// Process our component data.
					plane_tracker->set_bounds_size(bounded2d_list->get_size(i));
					plane_tracker->set_plane_alignment((OpenXRSpatialComponentPlaneAlignmentList::PlaneAlignment)alignment_list->get_plane_alignment(i));

					if (mesh2d_list.is_valid()) {
						plane_tracker->set_mesh_data(mesh2d_list->get_transform(i), mesh2d_list->get_vertices(p_snapshot, i), mesh2d_list->get_indices(p_snapshot, i));
					} else if (poly2d_list.is_valid()) {
						plane_tracker->set_mesh_data(poly2d_list->get_transform(i), poly2d_list->get_vertices(p_snapshot, i));
					} else {
						// Just in case we set this before.
						plane_tracker->clear_mesh_data();
					}

					if (label_list.is_valid()) {
						switch (label_list->get_plane_semantic_label(i)) {
							case XR_SPATIAL_PLANE_SEMANTIC_LABEL_UNCATEGORIZED_EXT: {
								plane_tracker->set_plane_label("Uncategorized plane");
							} break;
							case XR_SPATIAL_PLANE_SEMANTIC_LABEL_FLOOR_EXT: {
								plane_tracker->set_plane_label("Floor plane");
							} break;
							case XR_SPATIAL_PLANE_SEMANTIC_LABEL_WALL_EXT: {
								plane_tracker->set_plane_label("Wall plane");
							} break;
							case XR_SPATIAL_PLANE_SEMANTIC_LABEL_CEILING_EXT: {
								plane_tracker->set_plane_label("Ceiling plane");
							} break;
							case XR_SPATIAL_PLANE_SEMANTIC_LABEL_TABLE_EXT: {
								plane_tracker->set_plane_label("Table plane");
							} break;
							default: {
								plane_tracker->set_plane_label("Unknown plane");
							} break;
						}
					}
				}

				if (add_to_xr_server) {
					// Register with XR server
					xr_server->add_tracker(plane_tracker);
				}
			}
		}

		// Remove any planes that are no longer there...
		for (const XrSpatialEntityIdEXT &entity_id : current_planes) {
			if (plane_trackers.has(entity_id)) {
				Ref<OpenXRPlaneTracker> plane_tracker = plane_trackers[entity_id];

				// Just in case there are still references out there to this marker,
				// reset some stuff.
				plane_tracker->invalidate_pose(SNAME("default"));
				plane_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT);

				// Remove it from our XRServer
				xr_server->remove_tracker(plane_tracker);

				// Remove it from our trackers
				plane_trackers.erase(entity_id);
			}
		}
	}

	// Now that we're done, clean up our snapshot!
	se_extension->free_spatial_snapshot(p_snapshot);

	// And if this was our discovery snapshot, lets reset it
	if (discovery_query_result.is_valid() && discovery_query_result->get_result_value() == p_snapshot) {
		discovery_query_result.unref();
	}
}
