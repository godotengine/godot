/**************************************************************************/
/*  openxr_spatial_entity_extension.h                                     */
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

#include "../../openxr_util.h"
#include "../openxr_extension_wrapper.h"
#include "core/templates/rid_owner.h"
#include "core/variant/typed_array.h"
#include "openxr_spatial_entities.h"

// Spatial entity extension
class OpenXRSpatialEntityExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialEntityExtension, OpenXRExtensionWrapper);

public:
	enum Capability {
		CAPABILITY_PLANE_TRACKING = XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT,
		CAPABILITY_MARKER_TRACKING_QR_CODE = XR_SPATIAL_CAPABILITY_MARKER_TRACKING_QR_CODE_EXT,
		CAPABILITY_MARKER_TRACKING_MICRO_QR_CODE = XR_SPATIAL_CAPABILITY_MARKER_TRACKING_MICRO_QR_CODE_EXT,
		CAPABILITY_MARKER_TRACKING_ARUCO_MARKER = XR_SPATIAL_CAPABILITY_MARKER_TRACKING_ARUCO_MARKER_EXT,
		CAPABILITY_MARKER_TRACKING_APRIL_TAG = XR_SPATIAL_CAPABILITY_MARKER_TRACKING_APRIL_TAG_EXT,
		CAPABILITY_ANCHOR = XR_SPATIAL_CAPABILITY_ANCHOR_EXT,
	};

	enum ComponentType {
		COMPONENT_TYPE_BOUNDED_2D = XR_SPATIAL_COMPONENT_TYPE_BOUNDED_2D_EXT,
		COMPONENT_TYPE_BOUNDED_3D = XR_SPATIAL_COMPONENT_TYPE_BOUNDED_3D_EXT,
		COMPONENT_TYPE_PARENT = XR_SPATIAL_COMPONENT_TYPE_PARENT_EXT,
		COMPONENT_TYPE_MESH_3D = XR_SPATIAL_COMPONENT_TYPE_MESH_3D_EXT,
		COMPONENT_TYPE_PLANE_ALIGNMENT = XR_SPATIAL_COMPONENT_TYPE_PLANE_ALIGNMENT_EXT,
		COMPONENT_TYPE_MESH_2D = XR_SPATIAL_COMPONENT_TYPE_MESH_2D_EXT,
		COMPONENT_TYPE_POLYGON_2D = XR_SPATIAL_COMPONENT_TYPE_POLYGON_2D_EXT,
		COMPONENT_TYPE_PLANE_SEMANTIC_LABEL = XR_SPATIAL_COMPONENT_TYPE_PLANE_SEMANTIC_LABEL_EXT,
		COMPONENT_TYPE_MARKER = XR_SPATIAL_COMPONENT_TYPE_MARKER_EXT,
		COMPONENT_TYPE_ANCHOR = XR_SPATIAL_COMPONENT_TYPE_ANCHOR_EXT,
		COMPONENT_TYPE_PERSISTENCE = XR_SPATIAL_COMPONENT_TYPE_PERSISTENCE_EXT,
	};

	static OpenXRSpatialEntityExtension *get_singleton();

	OpenXRSpatialEntityExtension();
	virtual ~OpenXRSpatialEntityExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;
	virtual void on_session_destroyed() override;

	virtual bool on_event_polled(const XrEventDataBuffer &event) override;

	bool get_active() const;
	bool supports_capability(XrSpatialCapabilityEXT p_capability);
	bool supports_component_type(XrSpatialCapabilityEXT p_capability, XrSpatialComponentTypeEXT p_component_type);

	// Spatial contexts
	Ref<OpenXRFutureResult> create_spatial_context(const TypedArray<OpenXRSpatialCapabilityConfigurationBaseHeader> &p_capability_configurations, Ref<OpenXRStructureBase> p_next, const Callable &p_user_callback);
	bool get_spatial_context_ready(RID p_spatial_context) const;
	void free_spatial_context(RID p_spatial_context);
	XrSpatialContextEXT get_spatial_context_handle(RID p_spatial_context) const;

	// Discovery query
	Ref<OpenXRFutureResult> discover_spatial_entities(RID p_spatial_context, const Vector<XrSpatialComponentTypeEXT> &p_component_types, Ref<OpenXRStructureBase> p_next, const Callable &p_user_callback);

	// Update query
	RID update_spatial_entities(RID p_spatial_context, const LocalVector<RID> &p_entities, const LocalVector<XrSpatialComponentTypeEXT> &p_component_types, Ref<OpenXRStructureBase> p_next);

	// Snapshot data
	void free_spatial_snapshot(RID p_spatial_snapshot);
	XrSpatialSnapshotEXT get_spatial_snapshot_handle(RID p_spatial_snapshot) const;
	RID get_spatial_snapshot_context(RID p_spatial_snapshot) const;

	bool query_snapshot(RID p_spatial_snapshot, const TypedArray<OpenXRSpatialComponentData> &p_component_data, Ref<OpenXRStructureBase> p_next);

	// Buffers from snapshot
	String get_string(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;
	PackedByteArray get_uint8_buffer(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;
	Vector<uint16_t> get_uint16_buffer(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;
	Vector<uint32_t> get_uint32_buffer(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;
	PackedFloat32Array get_float_buffer(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;
	PackedVector2Array get_vector2_buffer(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;
	PackedVector3Array get_vector3_buffer(RID p_spatial_snapshot, XrSpatialBufferIdEXT p_buffer_id) const;

	// Entities
	RID find_spatial_entity(XrSpatialEntityIdEXT p_entity_id) const;
	RID add_spatial_entity(RID p_spatial_context, XrSpatialEntityIdEXT p_entity_id, XrSpatialEntityEXT p_entity);
	RID make_spatial_entity(RID p_spatial_context, XrSpatialEntityIdEXT p_entity_id);
	XrSpatialEntityIdEXT get_spatial_entity_id(RID p_entity) const;
	RID get_spatial_entity_context(RID p_entity) const;
	void free_spatial_entity(RID p_entity);

	static String get_spatial_capability_name(XrSpatialCapabilityEXT p_capability);
	static String get_spatial_component_type_name(XrSpatialComponentTypeEXT p_component_type);
	static String get_spatial_feature_name(XrSpatialCapabilityFeatureEXT p_feature);

protected:
	static void _bind_methods();

private:
	static OpenXRSpatialEntityExtension *singleton;

	bool spatial_entity_ext = false;

	// Capabilities
	struct SpatialEntityCapabality {
		Vector<XrSpatialComponentTypeEXT> component_types;
		Vector<XrSpatialCapabilityFeatureEXT> features;
	};
	HashMap<XrSpatialCapabilityEXT, SpatialEntityCapabality> supported_capabilities;
	int capabilities_load_state = 0; // 0 = no, 1 = yes, 2 = failed
	bool _load_capabilities();

	bool _supports_capability(Capability p_capability);
	bool _supports_component_type(Capability p_capability, ComponentType p_component_type);

	// Spatial context
	struct SpatialContextData {
		XrSpatialContextEXT spatial_context = XR_NULL_HANDLE;
	};
	mutable RID_Owner<SpatialContextData> spatial_context_owner;

	void _on_context_creation_ready(Ref<OpenXRFutureResult> p_future_result, const Callable &p_user_callback);
	uint64_t _get_spatial_context_handle(RID p_spatial_context) const;

	// Spatial query
	Ref<OpenXRFutureResult> _discover_spatial_entities(RID p_spatial_context, const PackedInt64Array &p_component_types, Ref<OpenXRStructureBase> p_next, const Callable &p_callback);
	void _on_discovered_spatial_entities(Ref<OpenXRFutureResult> p_future_result, RID p_discovery_spatial_context, const Callable &p_user_callback);

	// Update query
	RID _update_spatial_entities(RID p_spatial_context, const TypedArray<RID> &p_entities, const PackedInt64Array &p_component_types, Ref<OpenXRStructureBase> p_next);

	// Snapshot data
	struct SpatialSnapshotData {
		RID spatial_context;
		XrSpatialSnapshotEXT spatial_snapshot = XR_NULL_HANDLE;
	};
	mutable RID_Owner<SpatialSnapshotData> spatial_snapshot_owner;

	uint64_t _get_spatial_snapshot_handle(RID p_spatial_snapshot) const;

	// Buffers from snapshot
	String _get_string(RID p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedByteArray _get_uint8_buffer(RID p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedInt32Array _get_uint16_buffer(RID p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedInt32Array _get_uint32_buffer(RID p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedFloat32Array _get_float_buffer(RID p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedVector2Array _get_vector2_buffer(RID p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedVector3Array _get_vector3_buffer(RID p_spatial_snapshot, uint64_t p_buffer_id) const;

	// Entities
	struct SpatialEntityData {
		RID spatial_context;
		XrSpatialEntityIdEXT entity_id = XR_NULL_ENTITY;
		XrSpatialEntityEXT entity = XR_NULL_HANDLE;
	};
	mutable RID_Owner<SpatialEntityData> spatial_entity_owner;

	RID _find_entity(uint64_t p_entity_id);
	RID _add_entity(RID p_spatial_context, uint64_t p_entity_id, uint64_t p_entity);
	RID _make_entity(RID p_spatial_context, uint64_t p_entity_id);
	uint64_t _get_entity_id(RID p_entity) const;

	// OpenXR API call wrappers

	// Spatial entities
	EXT_PROTO_XRRESULT_FUNC5(xrEnumerateSpatialCapabilitiesEXT, (XrInstance), instance, (XrSystemId), system_id, (uint32_t), capability_capacity_input, (uint32_t *), capability_count_output, (XrSpatialCapabilityEXT *), capabilities);
	EXT_PROTO_XRRESULT_FUNC4(xrEnumerateSpatialCapabilityComponentTypesEXT, (XrInstance), instance, (XrSystemId), systemId, (XrSpatialCapabilityEXT), capability, (XrSpatialCapabilityComponentTypesEXT *), capability_components);
	EXT_PROTO_XRRESULT_FUNC6(xrEnumerateSpatialCapabilityFeaturesEXT, (XrInstance), instance, (XrSystemId), systemId, (XrSpatialCapabilityEXT), capability, (uint32_t), capability_feature_capacity_input, (uint32_t *), capability_feature_count_output, (XrSpatialCapabilityFeatureEXT *), capability_features);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialContextAsyncEXT, (XrSession), session, (const XrSpatialContextCreateInfoEXT *), create_info, (XrFutureEXT *), future);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialContextCompleteEXT, (XrSession), session, (XrFutureEXT), future, (XrCreateSpatialContextCompletionEXT *), completion);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpatialContextEXT, (XrSpatialContextEXT), spatial_context);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialDiscoverySnapshotAsyncEXT, (XrSpatialContextEXT), spatial_context, (const XrSpatialDiscoverySnapshotCreateInfoEXT *), create_info, (XrFutureEXT *), future);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialDiscoverySnapshotCompleteEXT, (XrSpatialContextEXT), spatial_context, (const XrCreateSpatialDiscoverySnapshotCompletionInfoEXT *), create_snapshot_completion_info, (XrCreateSpatialDiscoverySnapshotCompletionEXT *), completion);
	EXT_PROTO_XRRESULT_FUNC3(xrQuerySpatialComponentDataEXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialComponentDataQueryConditionEXT *), query_condition, (XrSpatialComponentDataQueryResultEXT *), query_result);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpatialSnapshotEXT, (XrSpatialSnapshotEXT), snapshot);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialEntityFromIdEXT, (XrSpatialContextEXT), spatial_context, (const XrSpatialEntityFromIdCreateInfoEXT *), create_info, (XrSpatialEntityEXT *), spatial_entity);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpatialEntityEXT, (XrSpatialEntityEXT), spatial_entity);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialUpdateSnapshotEXT, (XrSpatialContextEXT), spatial_context, (const XrSpatialUpdateSnapshotCreateInfoEXT *), createInfo, (XrSpatialSnapshotEXT *), snapshot);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferStringEXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (char *), buffer);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferUint8EXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (uint8_t *), buffer);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferUint16EXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (uint16_t *), buffer);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferUint32EXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (uint32_t *), buffer);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferFloatEXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (float *), buffer);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferVector2fEXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (XrVector2f *), buffer);
	EXT_PROTO_XRRESULT_FUNC5(xrGetSpatialBufferVector3fEXT, (XrSpatialSnapshotEXT), snapshot, (const XrSpatialBufferGetInfoEXT *), info, (uint32_t), buffer_capacity_input, (uint32_t *), buffer_count_output, (XrVector3f *), buffer);
};

VARIANT_ENUM_CAST(OpenXRSpatialEntityExtension::Capability);
VARIANT_ENUM_CAST(OpenXRSpatialEntityExtension::ComponentType);
