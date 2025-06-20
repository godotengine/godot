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
#include "scene/resources/3d/primitive_meshes.h"
#include "servers/xr_server.h"

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

		// Setup our enabled components
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
	ClassDB::bind_method(D_METHOD("get_plane_alignment", "index"), &OpenXRSpatialComponentPlaneAlignmentList::get_plane_alignment);

	XR_BIND_ENUM_CONSTANTS(XrSpatialPlaneAlignmentEXT);
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

	ERR_FAIL_COND_V(buffer.bufferId != XR_NULL_SPATIAL_BUFFER_ID_EXT, PackedVector2Array());
	ERR_FAIL_COND_V(buffer.bufferType != XR_SPATIAL_BUFFER_TYPE_VECTOR2F_EXT, PackedVector2Array());

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, PackedVector2Array());

	return se_extension->get_vector2_buffer(p_snapshot, buffer.bufferId);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentPlaneSemanticLabelList

void OpenXRSpatialComponentPlaneSemanticLabelList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_plane_semantic_label", "index"), &OpenXRSpatialComponentPlaneSemanticLabelList::get_plane_semantic_label);

	XR_BIND_ENUM_CONSTANTS(XrSpatialPlaneSemanticLabelEXT);
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
	ADD_PROPERTY(PropertyInfo(Variant::INT, "plane_label"), "set_plane_label", "get_plane_label");

	ClassDB::bind_method(D_METHOD("set_mesh_data", "origin", "vertices", "indices"), &OpenXRPlaneTracker::set_mesh_data, DEFVAL(PackedInt64Array()));
	ClassDB::bind_method(D_METHOD("clear_mesh_data"), &OpenXRPlaneTracker::clear_mesh_data);

	ClassDB::bind_method(D_METHOD("get_mesh_offset"), &OpenXRPlaneTracker::get_mesh_offset);
	ClassDB::bind_method(D_METHOD("get_mesh"), &OpenXRPlaneTracker::get_mesh);
	ClassDB::bind_method(D_METHOD("get_shape", "thickness"), &OpenXRPlaneTracker::get_shape, DEFVAL(0.01));

	ADD_SIGNAL(MethodInfo("mesh_changed"));
}

void OpenXRPlaneTracker::set_bounds_size(Vector2 p_bounds_size) {
	if (bounds_size != p_bounds_size) {
		bounds_size = p_bounds_size;

		if (!mesh.has_mesh) {
			emit_signal(SNAME("mesh_changed"));
		}
	}
}

Vector2 OpenXRPlaneTracker::get_bounds_size() const {
	return bounds_size;
}

void OpenXRPlaneTracker::set_plane_alignment(XrSpatialPlaneAlignmentEXT p_plane_alignment) {
	if (plane_alignment != p_plane_alignment) {
		plane_alignment = p_plane_alignment;
	}
}

XrSpatialPlaneAlignmentEXT OpenXRPlaneTracker::get_plane_alignment() const {
	return plane_alignment;
}

void OpenXRPlaneTracker::set_plane_label(XrSpatialPlaneSemanticLabelEXT p_plane_label) {
	if (plane_label != p_plane_label) {
		plane_label = p_plane_label;

		switch (plane_label) {
			case XR_SPATIAL_PLANE_SEMANTIC_LABEL_UNCATEGORIZED_EXT: {
				set_tracker_desc("Uncategorized plane");
			} break;
			case XR_SPATIAL_PLANE_SEMANTIC_LABEL_FLOOR_EXT: {
				set_tracker_desc("Floor plane");
			} break;
			case XR_SPATIAL_PLANE_SEMANTIC_LABEL_WALL_EXT: {
				set_tracker_desc("Wall plane");
			} break;
			case XR_SPATIAL_PLANE_SEMANTIC_LABEL_CEILING_EXT: {
				set_tracker_desc("Ceiling plane");
			} break;
			case XR_SPATIAL_PLANE_SEMANTIC_LABEL_TABLE_EXT: {
				set_tracker_desc("Table plane");
			} break;
			default: {
				set_tracker_desc("Unknown plane");
			} break;
		}
	}
}

XrSpatialPlaneSemanticLabelEXT OpenXRPlaneTracker::get_plane_label() const {
	return plane_label;
}

void OpenXRPlaneTracker::set_mesh_data(const Transform3D &p_origin, const PackedVector2Array &p_vertices, const PackedInt64Array &p_indices) {
	if (p_vertices.is_empty()) {
		clear_mesh_data();
	} else {
		mesh.has_mesh = true;
		mesh.origin = p_origin;
		mesh.vertices = p_vertices;
		if (p_indices.is_empty() && p_vertices.size() >= 3) {
			// Assume polygon, turn into triangle strip..
			int count = (p_vertices.size() - 2) * 3;
			int offset = 1;
			mesh.indices.resize(count);
			int64_t *idx = mesh.indices.ptrw();
			for (int i = 0; i < count; i += 3) {
				idx[i + 0] = 0;
				idx[i + 1] = offset++;
				idx[i + 2] = offset;
			}
		} else {
			// use as is
			mesh.indices = p_indices;
		}
	}

	// Assume in this case we have new mesh data.
	emit_signal(SNAME("mesh_changed"));
}

void OpenXRPlaneTracker::clear_mesh_data() {
	if (mesh.has_mesh) {
		mesh.has_mesh = false;
		mesh.origin = Transform3D();
		mesh.vertices.clear();
		mesh.indices.clear();

		emit_signal(SNAME("mesh_changed"));
	}
}

Transform3D OpenXRPlaneTracker::get_mesh_offset() const {
	Transform3D offset;

	if (mesh.has_mesh) {
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

	// Q is our pose already correctly oriented or do we need to add additional rotation here for plane alignment?
	// Won't know until we have equipment to test with.

	return offset;
}

Ref<Mesh> OpenXRPlaneTracker::get_mesh() const {
	if (mesh.has_mesh && false) {
		// TODO Use our mesh2d / polygon2d data to construct a mesh
		// Probably use array mesh
		Ref<Mesh> mesh3d;

		return mesh3d;
	} else {
		// We can use a plane mesh here.
		Ref<PlaneMesh> plane_mesh;

		plane_mesh.instantiate();
		plane_mesh->set_orientation(PlaneMesh::Orientation::FACE_Z);
		plane_mesh->set_size(bounds_size);

		return plane_mesh;
	}
}

Ref<Shape3D> OpenXRPlaneTracker::get_shape(real_t p_thickness) const {
	if (mesh.has_mesh && false) {
		// TODO Use our mesh2d / polygon2d data to construct a trimesh shape
		Ref<Shape3D> shape;

		return shape;
	} else {
		// We can use a box shape here
		Ref<BoxShape3D> box_shape;
		box_shape.instantiate();
		box_shape->set_size(Vector3(bounds_size.x, bounds_size.y, p_thickness));

		return box_shape;
	}
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

HashMap<String, bool *> OpenXRSpatialPlaneTrackingCapability::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enabled") && GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enable_plane_tracking")) {
		request_extensions[XR_EXT_SPATIAL_PLANE_TRACKING_EXTENSION_NAME] = &spatial_plane_tracking_ext;
	}

	return request_extensions;
}

void OpenXRSpatialPlaneTrackingCapability::on_instance_created(const XrInstance p_instance) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	if (spatial_plane_tracking_ext && !se_extension->supports_capability(XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT)) {
		// Supported by XR runtime but not by device? Disable this!
		spatial_plane_tracking_ext = false;
	}
}

void OpenXRSpatialPlaneTrackingCapability::on_session_created(const XrSession p_session) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	if (!spatial_plane_tracking_ext) {
		return;
	}

	se_extension->connect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_recommended));

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enable_builtin_plane_detection")) {
		// Start by creating our spatial context
		_create_spatial_context();
	}
}

void OpenXRSpatialPlaneTrackingCapability::on_session_destroyed() {
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

bool OpenXRSpatialPlaneTrackingCapability::is_supported() {
	if (!spatial_plane_tracking_ext) {
		return false;
	}

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, false);

	return se_extension->supports_capability(XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT);
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

	_start_entity_discovery();
}

void OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_recommended(RID p_spatial_context) {
	if (p_spatial_context == spatial_context) {
		// Trigger new discovery
		_start_entity_discovery();
	}
}

Ref<OpenXRFutureResult> OpenXRSpatialPlaneTrackingCapability::_start_entity_discovery() {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);

	// Already running, or ran discovery, cancel/cleanup
	if (discovery_query_result.is_valid()) {
		discovery_query_result->cancel_future();
		discovery_query_result.unref();
	}

	// Start our new snapshot.
	discovery_query_result = se_extension->discover_spatial_entities(spatial_context, plane_configuration->get_enabled_components(), nullptr, callable_mp(this, &OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_completed));

	return discovery_query_result;
}

void OpenXRSpatialPlaneTrackingCapability::_on_spatial_discovery_completed(RID p_snapshot) {
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
		// Q we're assuming entity id size and state size are equal, is there ever a situation where they would not be?
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
					plane_tracker->set_plane_alignment(alignment_list->get_plane_alignment(i));

					if (mesh2d_list.is_valid()) {
						plane_tracker->set_mesh_data(mesh2d_list->get_transform(i), mesh2d_list->get_vertices(p_snapshot, i), mesh2d_list->get_indices(p_snapshot, i));
					} else if (poly2d_list.is_valid()) {
						plane_tracker->set_mesh_data(poly2d_list->get_transform(i), poly2d_list->get_vertices(p_snapshot, i));
					} else {
						// Just in case we set this before.
						plane_tracker->clear_mesh_data();
					}

					if (label_list.is_valid()) {
						plane_tracker->set_plane_label(label_list->get_plane_semantic_label(i));
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
