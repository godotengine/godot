/**************************************************************************/
/*  openxr_spatial_mesh_tracking.h                                        */
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

#pragma once

#include "../openxr_extension_wrapper.h"
#include "../openxr_future_extension.h"
#include "openxr_spatial_entities.h"
#include "openxr_spatial_entity_extension.h"

#ifndef PHYSICS_3D_DISABLED
#include "scene/resources/3d/shape_3d.h"
#endif

// Mesh tracking capability configuration
class OpenXRSpatialCapabilityConfigurationMeshTracking : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationMeshTracking, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	bool get_supports_mesh_3d_normals();
	void add_next(Ref<OpenXRStructureBase> p_next);

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return mesh_enabled_components; }

protected:
	static void _bind_methods();

private:
	int supports_mesh_3d_normals = -1;

	Vector<XrSpatialComponentTypeEXT> mesh_enabled_components;
	XrSpatialCapabilityConfigurationMeshTrackingEXT mesh_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_MESH_TRACKING_EXT, nullptr, XR_SPATIAL_CAPABILITY_MESH_TRACKING_EXT, 0, nullptr };
	Ref<OpenXRStructureBase> next_structure;

	PackedInt64Array _get_enabled_components() const;
};

// Watertight configuration
class OpenXRSpatialContextWaterTightMeshConfig : public OpenXRStructureBase {
	GDCLASS(OpenXRSpatialContextWaterTightMeshConfig, OpenXRStructureBase);

public:
	virtual void *get_header(void *p_next) override;
	virtual XrStructureType get_structure_type() override;

	bool get_enable_watertight_mesh() const;
	void set_enable_watertight_mesh(bool p_enabled);

protected:
	static void _bind_methods();

private:
	XrSpatialFeatureTrackWatertightMeshEXT watertight_mesh_config = { XR_TYPE_SPATIAL_FEATURE_TRACK_WATERTIGHT_MESH_EXT, nullptr, true };
};

// Mesh3D normals list
class OpenXRSpatialComponentMesh3DNormalsList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentMesh3DNormalsList, OpenXRSpatialComponentData);

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	XrSpatialBufferIdEXT get_normal_buffer_id(int64_t p_index) const;
	PackedVector3Array get_normals(RID p_snapshot, int64_t p_index) const;

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialBufferEXT> mesh_normals_data;

	XrSpatialComponentMesh3DNormalsListEXT mesh3d_normals_list = { XR_TYPE_SPATIAL_COMPONENT_MESH_3D_NORMALS_LIST_EXT, nullptr, 0, nullptr };
};

// Mesh tracker
class OpenXRMeshTracker : public OpenXRSpatialEntityTracker {
	GDCLASS(OpenXRMeshTracker, OpenXRSpatialEntityTracker);

public:
	void set_bounds_size(const Vector3 &p_size) { bounds_size = p_size; }
	Vector3 get_bounds_size() const { return bounds_size; }

	void set_mesh_offset(const Transform3D &p_offset) { mesh.offset = p_offset; }
	Transform3D get_mesh_offset() const { return mesh.offset; }

	Transform3D get_mesh_adjusted_offset() const;

	void clear_mesh_data();
	void set_vertex_buffer(XrSpatialBufferIdEXT p_vertex_buffer_id, const PackedVector3Array &p_vertices);
	void set_normal_buffer(XrSpatialBufferIdEXT p_normal_buffer_id, const PackedVector3Array &p_normals);
	void set_index_buffer(XrSpatialBufferIdEXT p_index_buffer_id, const PackedInt32Array &p_indices);

	PackedVector3Array get_vertices() const { return mesh.vertices; }
	PackedVector3Array get_normals() const { return mesh.normals; }
	PackedInt32Array get_indices() const { return mesh.indices; }
	Ref<Mesh> get_mesh();
#ifndef PHYSICS_3D_DISABLED
	Ref<Shape3D> get_shape();
#endif

	XrSpatialBufferIdEXT get_vertex_buffer_id() const { return vertex_buffer_id; }
	XrSpatialBufferIdEXT get_normal_buffer_id() const { return normal_buffer_id; }
	XrSpatialBufferIdEXT get_index_buffer_id() const { return index_buffer_id; }

protected:
	static void _bind_methods();

private:
	Vector3 bounds_size;

	// Mesh data (if we have this)
	struct MeshData {
		Transform3D offset;
		PackedVector3Array vertices;
		PackedVector3Array normals;
		PackedInt32Array indices;

		Ref<Mesh> mesh;
#ifndef PHYSICS_3D_DISABLED
		Ref<Shape3D> shape3d;
#endif
	} mesh;

	struct Edge {
		int32_t a;
		int32_t b;
		static _FORCE_INLINE_ uint32_t hash(const Edge &p_edge) {
			uint32_t h = hash_murmur3_one_32(p_edge.a);
			return hash_murmur3_one_32(p_edge.b, h);
		}
		bool operator==(const Edge &p_edge) const {
			return (a == p_edge.a && b == p_edge.b);
		}

		Edge(int32_t p_a = 0, int32_t p_b = 0) {
			a = p_a;
			b = p_b;
			if (a < b) {
				SWAP(a, b);
			}
		}
	};

	// Cache buffer ids so we can check for new data
	XrSpatialBufferIdEXT vertex_buffer_id = XR_NULL_SPATIAL_BUFFER_ID_EXT;
	XrSpatialBufferIdEXT normal_buffer_id = XR_NULL_SPATIAL_BUFFER_ID_EXT;
	XrSpatialBufferIdEXT index_buffer_id = XR_NULL_SPATIAL_BUFFER_ID_EXT;
};

// Mesh tracking logic
class OpenXRSpatialMeshTrackingCapability : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialMeshTrackingCapability, OpenXRExtensionWrapper);

protected:
	static void _bind_methods();

public:
	static OpenXRSpatialMeshTrackingCapability *get_singleton();

	OpenXRSpatialMeshTrackingCapability();
	virtual ~OpenXRSpatialMeshTrackingCapability() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;

	virtual void on_process() override;

	Ref<OpenXRFutureResult> start_entity_discovery(RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_create = nullptr, Ref<OpenXRStructureBase> p_next_snapshot_query = nullptr, const Callable &p_user_callback = Callable());

	bool is_supported();
	bool is_watertight_supported();

	OpenXRSpatialEntityExtension::TrackingState get_built_in_tracking_state() { return builtin_tracking_state; }
	bool start_built_in_tracking(int p_watertight_mode);
	void stop_built_in_tracking(bool p_clear_trackers = true);

private:
	static OpenXRSpatialMeshTrackingCapability *singleton;
	bool spatial_mesh_tracking_ext = false;
	bool spatial_mesh_tracking_supported = false;
	bool spatial_mesh_watertight_supported = false;
	bool discovery_signal_connected = false;

	OpenXRSpatialEntityExtension::TrackingState builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_NOT_ACTIVE;
	Ref<OpenXRFutureResult> built_in_future;
	RID spatial_context;

	bool need_discovery = false;
	int discovery_cooldown = 0;
	Ref<OpenXRFutureResult> discovery_query_result;

	Ref<OpenXRSpatialCapabilityConfigurationMeshTracking> mesh_configuration;
	TypedArray<OpenXRSpatialComponentData> mesh_component_data;

	// Discovery logic
	Ref<OpenXRFutureResult> _create_spatial_context(int p_watertight_mode);
	void _on_spatial_context_created(RID p_spatial_context);
	void _on_spatial_context_creation_failed(int p_xr_result, bool p_is_completion_failure);

	void _on_spatial_discovery_recommended(RID p_spatial_context);

	void _process_snapshot(RID p_snapshot, RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_query, const Callable &p_user_callback);

	// Trackers; maps each Spatial Context RID to their mesh entities and trackers
	HashMap<RID, HashMap<XrSpatialEntityIdEXT, Ref<OpenXRMeshTracker>>> mesh_trackers;
};
