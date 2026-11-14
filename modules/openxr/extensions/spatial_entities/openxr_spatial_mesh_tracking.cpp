/**************************************************************************/
/*  openxr_spatial_mesh_tracking.cpp                                      */
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

#include "openxr_spatial_mesh_tracking.h"

#include "../../openxr_api.h"
#include "openxr_spatial_entity_extension.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"

#ifndef PHYSICS_3D_DISABLED
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#endif

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialCapabilityConfigurationMeshTracking

bool OpenXRSpatialCapabilityConfigurationMeshTracking::has_valid_configuration() const {
	OpenXRSpatialMeshTrackingCapability *capability = OpenXRSpatialMeshTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, false);

	return capability->is_supported();
}

XrSpatialCapabilityConfigurationBaseHeaderEXT *OpenXRSpatialCapabilityConfigurationMeshTracking::get_configuration() {
	OpenXRSpatialMeshTrackingCapability *capability = OpenXRSpatialMeshTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, nullptr);

	if (capability->is_supported()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, nullptr);

		mesh_enabled_components.clear();

		// Guaranteed components:
		mesh_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_BOUNDED_3D_EXT);
		mesh_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_MESH_3D_EXT);

		// Optional components:
		if (get_supports_mesh_3d_normals()) {
			mesh_enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_MESH_3D_NORMALS_EXT);
		}

		// TODO investigate if we want to support mesh semantic labels (maybe)

		// Set up our enabled components.
		mesh_config.enabledComponentCount = mesh_enabled_components.size();
		mesh_config.enabledComponents = mesh_enabled_components.ptr();

		// Check our next chain.
		if (next_structure.is_valid()) {
			mesh_config.next = next_structure->get_header();
		} else {
			mesh_config.next = nullptr;
		}

		// And return this.
		return (XrSpatialCapabilityConfigurationBaseHeaderEXT *)&mesh_config;
	}

	return nullptr;
}

bool OpenXRSpatialCapabilityConfigurationMeshTracking::get_supports_mesh_3d_normals() {
	if (supports_mesh_3d_normals == -1) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, false);

		supports_mesh_3d_normals = se_extension->supports_component_type(XR_SPATIAL_CAPABILITY_MESH_TRACKING_EXT, XR_SPATIAL_COMPONENT_TYPE_MESH_3D_NORMALS_EXT) ? 1 : 0;
	}

	return supports_mesh_3d_normals == 1;
}

void OpenXRSpatialCapabilityConfigurationMeshTracking::add_next(Ref<OpenXRStructureBase> p_next) {
	ERR_FAIL_COND(p_next.is_null());

	if (next_structure.is_null()) {
		next_structure = p_next;
		return;
	}

	// Find the end of our next chain.
	Ref<OpenXRStructureBase> n = next_structure;
	while (n->get_next().is_valid()) {
		n = n->get_next();
	}

	// Set it.
	n->set_next(p_next);
}

void OpenXRSpatialCapabilityConfigurationMeshTracking::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_supports_mesh_3d_normals"), &OpenXRSpatialCapabilityConfigurationMeshTracking::get_supports_mesh_3d_normals);
	ClassDB::bind_method(D_METHOD("add_next", "next"), &OpenXRSpatialCapabilityConfigurationMeshTracking::add_next);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialContextWaterTightMeshConfig

void *OpenXRSpatialContextWaterTightMeshConfig::get_header(void *p_next) {
	if (get_next().is_valid()) {
		watertight_mesh_config.next = get_next()->get_header(p_next);
	} else {
		watertight_mesh_config.next = p_next;
	}

	return &watertight_mesh_config;
}

XrStructureType OpenXRSpatialContextWaterTightMeshConfig::get_structure_type() {
	return XR_TYPE_SPATIAL_FEATURE_TRACK_WATERTIGHT_MESH_EXT;
}

bool OpenXRSpatialContextWaterTightMeshConfig::get_enable_watertight_mesh() const {
	return watertight_mesh_config.enableWatertightMeshTracking;
}

void OpenXRSpatialContextWaterTightMeshConfig::set_enable_watertight_mesh(bool p_enabled) {
	watertight_mesh_config.enableWatertightMeshTracking = p_enabled;
}

void OpenXRSpatialContextWaterTightMeshConfig::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enable_watertight_mesh", "enable"), &OpenXRSpatialContextWaterTightMeshConfig::set_enable_watertight_mesh);
	ClassDB::bind_method(D_METHOD("get_enable_watertight_mesh"), &OpenXRSpatialContextWaterTightMeshConfig::get_enable_watertight_mesh);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_watertight_mesh"), "set_enable_watertight_mesh", "get_enable_watertight_mesh");
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentMesh3DNormalsList

void OpenXRSpatialComponentMesh3DNormalsList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_normals", "snapshot", "index"), &OpenXRSpatialComponentMesh3DNormalsList::get_normals);
}

void OpenXRSpatialComponentMesh3DNormalsList::set_capacity(uint32_t p_capacity) {
	mesh_normals_data.resize(p_capacity);

	mesh3d_normals_list.meshNormalsCount = uint32_t(mesh_normals_data.size());
	mesh3d_normals_list.meshNormals = mesh_normals_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentMesh3DNormalsList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_MESH_3D_NORMALS_EXT;
}

void *OpenXRSpatialComponentMesh3DNormalsList::get_structure_data(void *p_next) {
	mesh3d_normals_list.next = p_next;
	return &mesh3d_normals_list;
}

XrSpatialBufferIdEXT OpenXRSpatialComponentMesh3DNormalsList::get_normal_buffer_id(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh_normals_data.size(), XR_NULL_SPATIAL_BUFFER_ID_EXT);

	return mesh_normals_data[p_index].bufferId;
}

PackedVector3Array OpenXRSpatialComponentMesh3DNormalsList::get_normals(RID p_snapshot, int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh_normals_data.size(), PackedVector3Array());

	const XrSpatialBufferEXT &buffer = mesh_normals_data[p_index];
	if (buffer.bufferId == XR_NULL_SPATIAL_BUFFER_ID_EXT) {
		// We don't have data (yet).
		return PackedVector3Array();
	}

	ERR_FAIL_COND_V(buffer.bufferType != XR_SPATIAL_BUFFER_TYPE_VECTOR3F_EXT, PackedVector3Array());

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, PackedVector3Array());

	return se_extension->get_vector3_buffer(p_snapshot, buffer.bufferId);
}

////////////////////////////////////////////////////////////////////////////
// OpenXRMeshTracker

void OpenXRMeshTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bounds_size", "bounds_size"), &OpenXRMeshTracker::set_bounds_size);
	ClassDB::bind_method(D_METHOD("get_bounds_size"), &OpenXRMeshTracker::get_bounds_size);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "bounds_size"), "set_bounds_size", "get_bounds_size");

	ClassDB::bind_method(D_METHOD("set_mesh_offset", "offset"), &OpenXRMeshTracker::set_mesh_offset);
	ClassDB::bind_method(D_METHOD("get_mesh_offset"), &OpenXRMeshTracker::get_mesh_offset);
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "mesh_offset"), "set_mesh_offset", "get_mesh_offset");

	ClassDB::bind_method(D_METHOD("get_mesh_adjusted_offset"), &OpenXRMeshTracker::get_mesh_adjusted_offset);

	ClassDB::bind_method(D_METHOD("clear_mesh_data"), &OpenXRMeshTracker::clear_mesh_data);
	ClassDB::bind_method(D_METHOD("get_vertices"), &OpenXRMeshTracker::get_vertices);
	ClassDB::bind_method(D_METHOD("get_normals"), &OpenXRMeshTracker::get_normals);
	ClassDB::bind_method(D_METHOD("get_indices"), &OpenXRMeshTracker::get_indices);
	ClassDB::bind_method(D_METHOD("get_mesh"), &OpenXRMeshTracker::get_mesh);
#ifndef PHYSICS_3D_DISABLED
	ClassDB::bind_method(D_METHOD("get_shape"), &OpenXRMeshTracker::get_shape);
#endif

	ADD_SIGNAL(MethodInfo("mesh_changed"));
}

Transform3D OpenXRMeshTracker::get_mesh_adjusted_offset() const {
	Transform3D offset = mesh.offset;

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server) {
		float ws = xr_server->get_world_scale();

		// Here we also scale the mesh/collision so the size keeps matching reality.
		offset.basis = offset.basis.scaled(Vector3(ws, ws, ws));
		offset.origin *= ws;
	}

	return offset;
}

void OpenXRMeshTracker::clear_mesh_data() {
	vertex_buffer_id = XR_NULL_SPATIAL_BUFFER_ID_EXT;
	normal_buffer_id = XR_NULL_SPATIAL_BUFFER_ID_EXT;
	index_buffer_id = XR_NULL_SPATIAL_BUFFER_ID_EXT;
	mesh.vertices.clear();
	mesh.normals.clear();
	mesh.indices.clear();

	if (mesh.mesh.is_valid()) {
		mesh.mesh.unref();
		emit_signal(SNAME("mesh_changed"));
	}
#ifndef PHYSICS_3D_DISABLED
	mesh.shape3d.unref();
#endif
}

void OpenXRMeshTracker::set_vertex_buffer(XrSpatialBufferIdEXT p_vertex_buffer_id, const PackedVector3Array &p_vertices) {
	if (vertex_buffer_id == p_vertex_buffer_id) {
		return;
	}

	vertex_buffer_id = p_vertex_buffer_id;
	mesh.vertices = p_vertices;

	// Free cashed mesh and shape data.
	if (mesh.mesh.is_valid()) {
		mesh.mesh.unref();
		emit_signal(SNAME("mesh_changed"));
	}
#ifndef PHYSICS_3D_DISABLED
	mesh.shape3d.unref();
#endif
}

void OpenXRMeshTracker::set_normal_buffer(XrSpatialBufferIdEXT p_normal_buffer_id, const PackedVector3Array &p_normals) {
	if (normal_buffer_id == p_normal_buffer_id) {
		return;
	}

	normal_buffer_id = p_normal_buffer_id;
	mesh.normals = p_normals;

	// Free cashed mesh and shape data.
	if (mesh.mesh.is_valid()) {
		mesh.mesh.unref();
		emit_signal(SNAME("mesh_changed"));
	}
#ifndef PHYSICS_3D_DISABLED
	mesh.shape3d.unref();
#endif
}

void OpenXRMeshTracker::set_index_buffer(XrSpatialBufferIdEXT p_index_buffer_id, const PackedInt32Array &p_indices) {
	if (index_buffer_id == p_index_buffer_id) {
		return;
	}

	index_buffer_id = p_index_buffer_id;
	mesh.indices = p_indices;

	// Free cashed mesh and shape data.
	if (mesh.mesh.is_valid()) {
		mesh.mesh.unref();
		emit_signal(SNAME("mesh_changed"));
	}
#ifndef PHYSICS_3D_DISABLED
	mesh.shape3d.unref();
#endif
}

Ref<Mesh> OpenXRMeshTracker::get_mesh() {
	// return cached data if available
	if (mesh.mesh.is_valid()) {
		return mesh.mesh;
	}

	if (mesh.vertices.size() > 3 && mesh.indices.size() > 3) {
		Ref<ArrayMesh> array_mesh;
		Array arr;

		// Build our array with data.
		arr.resize(RSE::ARRAY_MAX);
		arr[RSE::ARRAY_VERTEX] = mesh.vertices;
		if (mesh.normals.size() == mesh.vertices.size()) {
			arr[RSE::ARRAY_NORMAL] = mesh.normals;
		}
		arr[RSE::ARRAY_INDEX] = mesh.indices;

		// Create our array mesh.
		array_mesh.instantiate();
		array_mesh->add_surface_from_arrays(Mesh::PrimitiveType::PRIMITIVE_TRIANGLES, arr);

		// Cache this.
		mesh.mesh = array_mesh;
	}

	return mesh.mesh;
}

#ifndef PHYSICS_3D_DISABLED
Ref<Shape3D> OpenXRMeshTracker::get_shape() {
	// return cached data if available
	if (mesh.shape3d.is_valid()) {
		return mesh.shape3d;
	}

	Ref<ConcavePolygonShape3D> shape;
	Vector<Vector3> faces;

	// Get some direct access to our data.
	int isize = mesh.indices.size();
	const Vector3 *vr = mesh.vertices.ptr();
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
	faces.resize(isize + 6 * outer_edges.size());
	Vector3 *write = faces.ptrw();

	// Add surface.
	for (int i = 0; i < isize; i += 3) {
		*write++ = vr[ir[i]];
		*write++ = vr[ir[i + 1]];
		*write++ = vr[ir[i + 2]];
	}

	// Create our shape.
	shape.instantiate();
	shape->set_faces(faces);

	mesh.shape3d = shape;

	return mesh.shape3d;
}
#endif

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialMeshTrackingCapability

OpenXRSpatialMeshTrackingCapability *OpenXRSpatialMeshTrackingCapability::singleton = nullptr;

void OpenXRSpatialMeshTrackingCapability::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_supported"), &OpenXRSpatialMeshTrackingCapability::is_supported);
	ClassDB::bind_method(D_METHOD("is_watertight_supported"), &OpenXRSpatialMeshTrackingCapability::is_watertight_supported);
	ClassDB::bind_method(D_METHOD("start_entity_discovery", "spatial_context", "component_data", "next_snapshot_create", "next_snapshot_query", "user_callback"), &OpenXRSpatialMeshTrackingCapability::start_entity_discovery, DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Callable()));

	ClassDB::bind_method(D_METHOD("get_built_in_tracking_state"), &OpenXRSpatialMeshTrackingCapability::get_built_in_tracking_state);
	ClassDB::bind_method(D_METHOD("start_built_in_tracking", "watertight_mode"), &OpenXRSpatialMeshTrackingCapability::start_built_in_tracking);
	ClassDB::bind_method(D_METHOD("stop_built_in_tracking", "clear_trackers"), &OpenXRSpatialMeshTrackingCapability::stop_built_in_tracking, DEFVAL(true));
}

OpenXRSpatialMeshTrackingCapability *OpenXRSpatialMeshTrackingCapability::get_singleton() {
	return singleton;
}

OpenXRSpatialMeshTrackingCapability::OpenXRSpatialMeshTrackingCapability() {
	singleton = this;
}

OpenXRSpatialMeshTrackingCapability::~OpenXRSpatialMeshTrackingCapability() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRSpatialMeshTrackingCapability::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enabled") && GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/mesh_tracking/enable")) {
		// Note, this extension is dependent on archetypes, but this is already being requested.
		request_extensions[XR_EXT_SPATIAL_MESH_TRACKING_EXTENSION_NAME] = &spatial_mesh_tracking_ext;
	}

	return request_extensions;
}

void OpenXRSpatialMeshTrackingCapability::on_session_created(const XrSession p_session) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	if (!se_extension->get_active()) {
		spatial_mesh_tracking_ext = false;
		return;
	}

	if (!spatial_mesh_tracking_ext) {
		return;
	}

	if (!se_extension->supports_archetypes()) {
		// This would mean the XR Runtime isn't following spec, but just in case we double check.
		spatial_mesh_tracking_ext = false;
		return;
	}

	spatial_mesh_tracking_supported = se_extension->supports_capability(XR_SPATIAL_CAPABILITY_MESH_TRACKING_EXT);
	if (!spatial_mesh_tracking_supported) {
		// Supported by XR runtime but not by device? We're done.
		return;
	}

	spatial_mesh_watertight_supported = se_extension->supports_capability_feature(XR_SPATIAL_CAPABILITY_MESH_TRACKING_EXT, XR_SPATIAL_CAPABILITY_FEATURE_TRACK_WATERTIGHT_MESH_EXT);

	if (!discovery_signal_connected) {
		discovery_signal_connected = true;
		se_extension->connect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialMeshTrackingCapability::_on_spatial_discovery_recommended));
	}

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/mesh_tracking/enable_builtin_detection")) {
		// Start our built in tracking.
		start_built_in_tracking(GLOBAL_GET_CACHED(int, "xr/openxr/extensions/spatial_entity/mesh_tracking/watertight_mode"));
	}
}

void OpenXRSpatialMeshTrackingCapability::on_session_destroyed() {
	if (!spatial_mesh_tracking_supported) {
		return;
	}
	spatial_mesh_tracking_supported = false;

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	// Stop our built in tracking (if applicable), this will also clean up any meshes previously detected.
	stop_built_in_tracking();

	if (discovery_signal_connected) {
		discovery_signal_connected = false;
		se_extension->disconnect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialMeshTrackingCapability::_on_spatial_discovery_recommended));
	}
}

void OpenXRSpatialMeshTrackingCapability::on_process() {
	if (!spatial_context.is_valid()) {
		return;
	}

	// Protection against mesh discovery happening too often.
	if (discovery_cooldown > 0) {
		discovery_cooldown--;
	}

	// Check if we need to start our discovery.
	if (need_discovery && discovery_cooldown == 0 && !discovery_query_result.is_valid()) {
		need_discovery = false;
		discovery_cooldown = 60; // Set our cooldown to 60 frames, it doesn't need to be an exact science.

		if (mesh_component_data.is_empty()) {
			if (mesh_configuration.is_null()) {
				// This should not happen, start_built_in_tracking not called?
				mesh_configuration.instantiate();
			}

			// We always need a query result data object, and it must be first
			Ref<OpenXRSpatialQueryResultData> query_result_data;
			query_result_data.instantiate();
			mesh_component_data.push_back(query_result_data);

			// Base our mesh_component_data on the component types used in plane_configuration
			XrSpatialCapabilityConfigurationBaseHeaderEXT *config = mesh_configuration->get_configuration();
			for (uint32_t i = 0; i < config->enabledComponentCount; ++i) {
				switch (config->enabledComponents[i]) {
					case XR_SPATIAL_COMPONENT_TYPE_BOUNDED_3D_EXT: {
						Ref<OpenXRSpatialComponentBounded3DList> bounded3d_list;
						bounded3d_list.instantiate();
						mesh_component_data.push_back(bounded3d_list);
					} break;
					case XR_SPATIAL_COMPONENT_TYPE_MESH_3D_EXT: {
						Ref<OpenXRSpatialComponentMesh3DList> mesh3d_list;
						mesh3d_list.instantiate();
						mesh_component_data.push_back(mesh3d_list);
					} break;
					case XR_SPATIAL_COMPONENT_TYPE_MESH_3D_NORMALS_EXT: {
						Ref<OpenXRSpatialComponentMesh3DNormalsList> mesh3d_normals_list;
						mesh3d_normals_list.instantiate();
						mesh_component_data.push_back(mesh3d_normals_list);
					} break;
					default:
						WARN_PRINT("Unexpected mesh component; ignoring");
						break;
				}
			}
		}

		discovery_query_result = start_entity_discovery(spatial_context, mesh_component_data);
	}
}

Ref<OpenXRFutureResult> OpenXRSpatialMeshTrackingCapability::start_entity_discovery(RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_create, Ref<OpenXRStructureBase> p_next_snapshot_query, const Callable &p_user_callback) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);
	return se_extension->discover_spatial_entities_with_component_data(p_spatial_context, p_component_data, p_next_snapshot_create, callable_mp(this, &OpenXRSpatialMeshTrackingCapability::_process_snapshot).bind(p_spatial_context, p_component_data, p_next_snapshot_query, p_user_callback));
}

bool OpenXRSpatialMeshTrackingCapability::is_supported() {
	return spatial_mesh_tracking_ext && spatial_mesh_tracking_supported;
}

bool OpenXRSpatialMeshTrackingCapability::is_watertight_supported() {
	return spatial_mesh_tracking_ext && spatial_mesh_tracking_supported && spatial_mesh_watertight_supported;
}

bool OpenXRSpatialMeshTrackingCapability::start_built_in_tracking(int p_watertight_mode) {
	ERR_FAIL_COND_V(!is_supported(), false);
	ERR_FAIL_COND_V(builtin_tracking_state > 0, false);
	builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_SETTING_UP;

	// Start by creating our spatial context
	built_in_future = _create_spatial_context(p_watertight_mode);
	if (built_in_future.is_null()) {
		builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_SETUP_FAILED;
		return false;
	}

	return true;
}

void OpenXRSpatialMeshTrackingCapability::stop_built_in_tracking(bool p_clear_trackers) {
	// Reset our tracking state
	builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_NOT_ACTIVE;

	// Cancel any discovery query
	if (discovery_query_result.is_valid()) {
		if (discovery_query_result->get_status() == OpenXRFutureResult::RESULT_RUNNING) {
			discovery_query_result->cancel_future();
		}

		discovery_query_result.unref();
	}

	// If we have our future object, clean it up.
	if (built_in_future.is_valid()) {
		if (built_in_future->get_status() == OpenXRFutureResult::RESULT_RUNNING) {
			built_in_future->cancel_future();
		}

		built_in_future.unref();
	}

	// Free our spatial context
	if (spatial_context.is_valid()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL(se_extension);

		if (p_clear_trackers && mesh_trackers.has(spatial_context)) {
			XRServer *xr_server = XRServer::get_singleton();
			ERR_FAIL_NULL(xr_server);

			// Free and unregister our anchors
			HashMap<XrSpatialEntityIdEXT, Ref<OpenXRMeshTracker>> &meshes = mesh_trackers[spatial_context];
			for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRMeshTracker>> &mesh_tracker : meshes) {
				xr_server->remove_tracker(mesh_tracker.value);
			}
			mesh_trackers.erase(spatial_context);
		}

		se_extension->free_spatial_context(spatial_context);
		spatial_context = RID();
	}
}

////////////////////////////////////////////////////////////////////////////
// Discovery logic

Ref<OpenXRFutureResult> OpenXRSpatialMeshTrackingCapability::_create_spatial_context(int p_watertight_mode) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);
	ERR_FAIL_COND_V(p_watertight_mode == 2 && !spatial_mesh_watertight_supported, nullptr);

	TypedArray<OpenXRSpatialCapabilityConfigurationBaseHeader> capability_configurations;

	// Create our configuration objects.
	mesh_configuration.instantiate();

	if (p_watertight_mode > 0 && spatial_mesh_watertight_supported) {
		Ref<OpenXRSpatialContextWaterTightMeshConfig> watertight_config;
		watertight_config.instantiate();

		// Add add_next...
		mesh_configuration->add_next(watertight_config);
	}

	capability_configurations.push_back(mesh_configuration);

	return se_extension->create_spatial_context(capability_configurations, nullptr, callable_mp(this, &OpenXRSpatialMeshTrackingCapability::_on_spatial_context_created), callable_mp(this, &OpenXRSpatialMeshTrackingCapability::_on_spatial_context_creation_failed));
}

void OpenXRSpatialMeshTrackingCapability::_on_spatial_context_created(RID p_spatial_context) {
	builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_ENABLED;
	spatial_context = p_spatial_context;
	need_discovery = true;

	// We don't need our future anymore.
	built_in_future.unref();
}

void OpenXRSpatialMeshTrackingCapability::_on_spatial_context_creation_failed(int p_xr_result, bool p_is_completion_failure) {
	XrResult result = XrResult(p_xr_result);

	switch (result) {
		case XR_ERROR_PERMISSION_INSUFFICIENT:
			builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_NO_PERMISSION;
			break;
		case XR_ERROR_SPATIAL_CAPABILITY_UNSUPPORTED_EXT:
			builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_UNSUPPORTED_CAPABILITY;
			break;
		// We may wish to support additional error codes that are likely here.
		default:
			builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_SETUP_FAILED;
			break;
	}

	// We don't need our future anymore.
	built_in_future.unref();
}

void OpenXRSpatialMeshTrackingCapability::_on_spatial_discovery_recommended(RID p_spatial_context) {
	if (p_spatial_context == spatial_context) {
		// Trigger new discovery.
		need_discovery = true;
	}
}

void OpenXRSpatialMeshTrackingCapability::_process_snapshot(RID p_snapshot, RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_query, const Callable &p_user_callback) {
	if (p_user_callback.is_valid()) {
		p_user_callback.call(p_snapshot, false);
	}

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Make a copy of the meshes we have right now, so we know which ones to clean up.
	LocalVector<XrSpatialEntityIdEXT> current_meshes;
	HashMap<XrSpatialEntityIdEXT, Ref<OpenXRMeshTracker>> &meshes = mesh_trackers[p_spatial_context];
	current_meshes.resize(meshes.size());
	int p = 0;
	for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRMeshTracker>> &mesh : meshes) {
		current_meshes[p++] = mesh.key;
	}

	// The first must be OpenXRSpatialQueryResultData
	Ref<OpenXRSpatialQueryResultData> query_result_data = p_component_data.is_empty() ? Variant() : p_component_data[0];
	ERR_FAIL_COND(query_result_data.is_null());

	// Get the archetypes present in our snapshot.
	// Note: archetypes can be empty if our snapshot is empty!
	LocalVector<OpenXRSpatialEntityExtension::Archetype> archetypes;
	archetypes = se_extension->get_snapshot_archetypes(p_snapshot);

	// Now loop through our archetypes to find all our entities.
	for (const OpenXRSpatialEntityExtension::Archetype &archetype : archetypes) {
		Ref<OpenXRStructureBase> next = p_next_snapshot_query;

		// Add a filter for our archetype so we only select entities that match this profile.
		Ref<OpenXrSpatialFilterArchetype> filter_archetype;
		filter_archetype.instantiate();
		filter_archetype->set_archetype_id(archetype.archetype_id);
		filter_archetype->set_next(next);
		next = filter_archetype;

		// Filter out components that are part of our entities for this archetype.
		TypedArray<OpenXRSpatialComponentData> component_data;
		for (Ref<OpenXRSpatialComponentData> data : p_component_data) {
			if ((data == query_result_data) || (archetype.component_types.has(data->get_component_type()))) {
				component_data.push_back(data);
			}
		}

		if (se_extension->query_snapshot(p_snapshot, component_data, next)) {
			int64_t size = query_result_data->get_capacity();
			for (int64_t i = 0; i < size; i++) {
				XrSpatialEntityIdEXT entity_id = query_result_data->get_entity_id(i);
				XrSpatialEntityTrackingStateEXT entity_state = query_result_data->get_entity_state(i);

				// Erase it from our current meshes (if we have it, else this is ignored).
				current_meshes.erase(entity_id);

				if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT) {
					// We should only get this status on updates as a prelude to needing to remove this mesh.
					// So we just update the status.
					if (meshes.has(entity_id)) {
						Ref<OpenXRMeshTracker> mesh_tracker = meshes[entity_id];
						mesh_tracker->invalidate_pose(SNAME("default"));
						mesh_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT);
					}
				} else {
					// Process our entity
					bool add_to_xr_server = false;
					Ref<OpenXRMeshTracker> mesh_tracker;

					if (meshes.has(entity_id)) {
						// We know about this one already
						mesh_tracker = meshes[entity_id];
					} else {
						// Create a new anchor
						mesh_tracker.instantiate();
						mesh_tracker->set_spatial_context(p_spatial_context);
						mesh_tracker->set_entity(se_extension->make_spatial_entity(se_extension->get_spatial_snapshot_context(p_snapshot), entity_id));
						meshes[entity_id] = mesh_tracker;

						add_to_xr_server = true;
					}

					// Handle component data
					if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT) {
						mesh_tracker->invalidate_pose(SNAME("default"));
						mesh_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT);

						// No further component data will be valid in this state, we need to ignore it!
					} else if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT) {
						bool has_mesh_data = false;
						mesh_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT);
						Transform3D center_pose;
						Transform3D mesh_transform;

						for (Ref<OpenXRSpatialComponentData> data : component_data) {
							switch (data->get_component_type()) {
								case XR_SPATIAL_COMPONENT_TYPE_BOUNDED_3D_EXT: {
									Ref<OpenXRSpatialComponentBounded3DList> bounded3d_list = data;
									if (bounded3d_list.is_valid()) {
										center_pose = bounded3d_list->get_center_pose(i);
										mesh_tracker->set_pose(SNAME("default"), center_pose, Vector3(), Vector3());
										mesh_tracker->set_bounds_size(bounded3d_list->get_size(i));
									}
								} break;
								case XR_SPATIAL_COMPONENT_TYPE_MESH_3D_EXT: {
									Ref<OpenXRSpatialComponentMesh3DList> mesh3d_list = data;
									if (mesh3d_list.is_valid()) {
										mesh_transform = mesh3d_list->get_transform(i);

										XrSpatialBufferIdEXT vertex_buffer_id = mesh3d_list->get_vertex_buffer_id(i);
										if (mesh_tracker->get_vertex_buffer_id() != vertex_buffer_id) {
											mesh_tracker->set_vertex_buffer(vertex_buffer_id, mesh3d_list->get_vertices(p_snapshot, i));
										}
										XrSpatialBufferIdEXT index_buffer_id = mesh3d_list->get_index_buffer_id(i);
										if (mesh_tracker->get_index_buffer_id() != index_buffer_id) {
											mesh_tracker->set_index_buffer(index_buffer_id, mesh3d_list->get_indices(p_snapshot, i));
										}

										has_mesh_data = true;
									}
								} break;
								case XR_SPATIAL_COMPONENT_TYPE_MESH_3D_NORMALS_EXT: {
									Ref<OpenXRSpatialComponentMesh3DNormalsList> mesh3d_normals_list = data;
									if (mesh3d_normals_list.is_valid()) {
										XrSpatialBufferIdEXT normal_buffer_id = mesh3d_normals_list->get_normal_buffer_id(i);
										if (mesh_tracker->get_normal_buffer_id() != normal_buffer_id) {
											mesh_tracker->set_normal_buffer(normal_buffer_id, mesh3d_normals_list->get_normals(p_snapshot, i));
										}
									}
								} break;
								default:
									// Okay, maybe other data types are being queried that we don't know about
									break;
							}
						}

						// Set this as an offset, this will likely end up being an identity transform in this situation.
						mesh_tracker->set_mesh_offset(center_pose.inverse() * mesh_transform);

						if (!has_mesh_data) {
							// Just in case we set this before.
							mesh_tracker->clear_mesh_data();
						}
					}

					if (add_to_xr_server) {
						// Register with XR server
						xr_server->add_tracker(mesh_tracker);
					}
				}
			}
		}
	}

	// Remove any meshes that are no longer there...
	for (const XrSpatialEntityIdEXT &entity_id : current_meshes) {
		if (meshes.has(entity_id)) {
			Ref<OpenXRMeshTracker> mesh_tracker = meshes[entity_id];

			// Just in case there are still references out there to this marker,
			// reset some stuff.
			mesh_tracker->invalidate_pose(SNAME("default"));
			mesh_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT);

			// Remove it from our XRServer
			xr_server->remove_tracker(mesh_tracker);

			// Remove it from our trackers
			meshes.erase(entity_id);
		}
	}
}
