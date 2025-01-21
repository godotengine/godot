/**************************************************************************/
/*  openxr_spatial_entities.cpp                                           */
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

#include "openxr_spatial_entities.h"

#include "../../openxr_api.h"
#include "core/variant/native_ptr.h"
#include "openxr_spatial_entity_extension.h"

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialCapabilityConfigurationBaseHeader

void OpenXRSpatialCapabilityConfigurationBaseHeader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_valid_configuration"), &OpenXRSpatialCapabilityConfigurationBaseHeader::has_valid_configuration);

	GDVIRTUAL_BIND(_has_valid_configuration);
	GDVIRTUAL_BIND(_get_configuration);
}

bool OpenXRSpatialCapabilityConfigurationBaseHeader::has_valid_configuration() const {
	bool is_valid = false;

	if (GDVIRTUAL_CALL(_has_valid_configuration, is_valid)) {
		return is_valid;
	}

	return false;
}

XrSpatialCapabilityConfigurationBaseHeaderEXT *OpenXRSpatialCapabilityConfigurationBaseHeader::get_configuration() {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_get_configuration, pointer)) {
		return reinterpret_cast<XrSpatialCapabilityConfigurationBaseHeaderEXT *>(pointer);
	}

	return nullptr;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialEntityTracker

void OpenXRSpatialEntityTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_entity", "entity"), &OpenXRSpatialEntityTracker::set_entity);
	ClassDB::bind_method(D_METHOD("get_entity"), &OpenXRSpatialEntityTracker::get_entity);

	ADD_PROPERTY(PropertyInfo(Variant::RID, "entity"), "set_entity", "get_entity");

	ClassDB::bind_method(D_METHOD("set_spatial_tracking_state", "spatial_tracking_state"), &OpenXRSpatialEntityTracker::_set_spatial_tracking_state);
	ClassDB::bind_method(D_METHOD("get_spatial_tracking_state"), &OpenXRSpatialEntityTracker::_get_spatial_tracking_state);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spatial_tracking_state"), "set_spatial_tracking_state", "get_spatial_tracking_state");

	ADD_SIGNAL(MethodInfo("spatial_tracking_state_changed", PropertyInfo(Variant::INT, "spatial_tracking_state")));

	BIND_ENUM_CONSTANT(ENTITY_TRACKING_STATE_STOPPED);
	BIND_ENUM_CONSTANT(ENTITY_TRACKING_STATE_PAUSED);
	BIND_ENUM_CONSTANT(ENTITY_TRACKING_STATE_TRACKING);
}

OpenXRSpatialEntityTracker::OpenXRSpatialEntityTracker() {
	set_tracker_type(XRServer::TrackerType::TRACKER_ANCHOR);
}

OpenXRSpatialEntityTracker::~OpenXRSpatialEntityTracker() {
	if (spatial_entity.is_valid()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		if (se_extension) {
			se_extension->free_spatial_entity(spatial_entity);
			spatial_entity = RID();
		}
	}
}

void OpenXRSpatialEntityTracker::set_entity(const RID &p_entity) {
	if (spatial_entity.is_valid()) {
		if (spatial_entity == p_entity) {
			return;
		}

		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		if (se_extension) {
			se_extension->free_spatial_entity(spatial_entity);
			spatial_entity = RID();
		}
	}

	spatial_entity = p_entity;

	if (p_entity.is_valid()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL(se_extension);

		XrSpatialEntityIdEXT entity_id = se_extension->get_spatial_entity_id(p_entity);

		String tracker_name = String("openxr/spatial_entity/") + String::num_int64(entity_id);
		set_tracker_name(tracker_name);
	} else {
		set_tracker_name("openxr/spatial_entity/null");
	}
}

RID OpenXRSpatialEntityTracker::get_entity() const {
	return spatial_entity;
}

void OpenXRSpatialEntityTracker::set_spatial_tracking_state(const XrSpatialEntityTrackingStateEXT p_state) {
	if (spatial_tracking_state != p_state) {
		spatial_tracking_state = p_state;

		emit_signal(SNAME("spatial_tracking_state_changed"), spatial_tracking_state);
	}
}

void OpenXRSpatialEntityTracker::_set_spatial_tracking_state(const EntityTrackingState p_state) {
	set_spatial_tracking_state((XrSpatialEntityTrackingStateEXT)p_state);
}

XrSpatialEntityTrackingStateEXT OpenXRSpatialEntityTracker::get_spatial_tracking_state() const {
	return spatial_tracking_state;
}

OpenXRSpatialEntityTracker::EntityTrackingState OpenXRSpatialEntityTracker::_get_spatial_tracking_state() const {
	return (EntityTrackingState)get_spatial_tracking_state();
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialComponentData

void OpenXRSpatialComponentData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_capacity", "capacity"), &OpenXRSpatialComponentData::set_capacity);

	GDVIRTUAL_BIND(_set_capacity, "capacity");
	GDVIRTUAL_BIND(_get_component_type);
	GDVIRTUAL_BIND(_get_structure_data, "next");
}

void OpenXRSpatialComponentData::set_capacity(uint32_t p_capacity) {
	GDVIRTUAL_CALL(_set_capacity, p_capacity);
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentData::get_component_type() const {
	uint64_t component_type = XR_SPATIAL_COMPONENT_TYPE_MAX_ENUM_EXT;

	if (GDVIRTUAL_CALL(_get_component_type, component_type)) {
		return (XrSpatialComponentTypeEXT)component_type;
	}

	return XR_SPATIAL_COMPONENT_TYPE_MAX_ENUM_EXT;
}

void *OpenXRSpatialComponentData::get_structure_data(void *p_next) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_get_structure_data, (uint64_t)p_next, pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return p_next;
}

////////////////////////////////////////////////////////////////////////////
// Spatial component bounded2d list

void OpenXRSpatialComponentBounded2DList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_center_pose", "index"), &OpenXRSpatialComponentBounded2DList::get_center_pose);
	ClassDB::bind_method(D_METHOD("get_size", "index"), &OpenXRSpatialComponentBounded2DList::get_size);
}

void OpenXRSpatialComponentBounded2DList::set_capacity(uint32_t p_capacity) {
	bounded2d_data.resize(p_capacity);

	bounded2d_list.boundCount = uint32_t(bounded2d_data.size());
	bounded2d_list.bounds = bounded2d_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentBounded2DList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_BOUNDED_2D_EXT;
}

void *OpenXRSpatialComponentBounded2DList::get_structure_data(void *p_next) {
	bounded2d_list.next = p_next;
	return &bounded2d_list;
}

Transform3D OpenXRSpatialComponentBounded2DList::get_center_pose(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, bounded2d_data.size(), Transform3D());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	return openxr_api->transform_from_pose(bounded2d_data[p_index].center);
}

Vector2 OpenXRSpatialComponentBounded2DList::get_size(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, bounded2d_data.size(), Vector2());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Vector2());

	const XrExtent2Df &extents = bounded2d_data[p_index].extents;
	return Vector2(extents.width, extents.height);
}

////////////////////////////////////////////////////////////////////////////
// Spatial component bounded3d list

void OpenXRSpatialComponentBounded3DList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_center_pose", "index"), &OpenXRSpatialComponentBounded3DList::get_center_pose);
	ClassDB::bind_method(D_METHOD("get_size", "index"), &OpenXRSpatialComponentBounded3DList::get_size);
}

void OpenXRSpatialComponentBounded3DList::set_capacity(uint32_t p_capacity) {
	bounded3d_data.resize(p_capacity);

	bounded3d_list.boundCount = uint32_t(bounded3d_data.size());
	bounded3d_list.bounds = bounded3d_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentBounded3DList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_BOUNDED_3D_EXT;
}

void *OpenXRSpatialComponentBounded3DList::get_structure_data(void *p_next) {
	bounded3d_list.next = p_next;
	return &bounded3d_list;
}

Transform3D OpenXRSpatialComponentBounded3DList::get_center_pose(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, bounded3d_data.size(), Transform3D());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	return openxr_api->transform_from_pose(bounded3d_data[p_index].center);
}

Vector3 OpenXRSpatialComponentBounded3DList::get_size(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, bounded3d_data.size(), Vector3());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Vector3());

	const XrExtent3Df &extents = bounded3d_data[p_index].extents;
	return Vector3(extents.width, extents.height, extents.depth);
}

////////////////////////////////////////////////////////////////////////////
// Spatial component parent list

void OpenXRSpatialComponentParentList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_parent", "index"), &OpenXRSpatialComponentParentList::get_parent);
}

void OpenXRSpatialComponentParentList::set_capacity(uint32_t p_capacity) {
	parent_data.resize(p_capacity);

	parent_list.parentCount = uint32_t(parent_data.size());
	parent_list.parents = parent_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentParentList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_PARENT_EXT;
}

void *OpenXRSpatialComponentParentList::get_structure_data(void *p_next) {
	parent_list.next = p_next;
	return &parent_list;
}

RID OpenXRSpatialComponentParentList::get_parent(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, parent_data.size(), RID());

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, RID());

	return se_extension->find_spatial_entity(parent_data[p_index]);
}

////////////////////////////////////////////////////////////////////////////
// Spatial component mesh2d list

void OpenXRSpatialComponentMesh2DList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_transform", "index"), &OpenXRSpatialComponentMesh2DList::get_transform);
	ClassDB::bind_method(D_METHOD("get_vertices", "snapshot", "index"), &OpenXRSpatialComponentMesh2DList::get_vertices);
	ClassDB::bind_method(D_METHOD("get_indices", "snapshot", "index"), &OpenXRSpatialComponentMesh2DList::get_indices);
}

void OpenXRSpatialComponentMesh2DList::set_capacity(uint32_t p_capacity) {
	mesh2d_data.resize(p_capacity);

	mesh2d_list.meshCount = uint32_t(mesh2d_data.size());
	mesh2d_list.meshes = mesh2d_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentMesh2DList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_MESH_2D_EXT;
}

void *OpenXRSpatialComponentMesh2DList::get_structure_data(void *p_next) {
	mesh2d_list.next = p_next;
	return &mesh2d_list;
}

Transform3D OpenXRSpatialComponentMesh2DList::get_transform(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh2d_data.size(), Transform3D());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	return openxr_api->transform_from_pose(mesh2d_data[p_index].origin);
}

PackedVector2Array OpenXRSpatialComponentMesh2DList::get_vertices(RID p_snapshot, int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh2d_data.size(), PackedVector2Array());

	const XrSpatialBufferEXT &buffer = mesh2d_data[p_index].vertexBuffer;
	if (buffer.bufferId == XR_NULL_SPATIAL_BUFFER_ID_EXT) {
		// We don't have data (yet).
		return PackedVector2Array();
	}

	ERR_FAIL_COND_V(buffer.bufferType != XR_SPATIAL_BUFFER_TYPE_VECTOR2F_EXT, PackedVector2Array());

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, PackedVector2Array());

	return se_extension->get_vector2_buffer(p_snapshot, buffer.bufferId);
}

PackedInt32Array OpenXRSpatialComponentMesh2DList::get_indices(RID p_snapshot, int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh2d_data.size(), PackedInt32Array());

	const XrSpatialBufferEXT &buffer = mesh2d_data[p_index].indexBuffer;
	if (buffer.bufferId == XR_NULL_SPATIAL_BUFFER_ID_EXT) {
		// We don't have data (yet).
		return PackedInt32Array();
	}

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, PackedInt32Array());

	PackedInt32Array ret;

	switch (buffer.bufferType) {
		case XR_SPATIAL_BUFFER_TYPE_UINT8_EXT: {
			PackedByteArray data = se_extension->get_uint8_buffer(p_snapshot, buffer.bufferId);

			ret.resize(data.size());
			int count = ret.size();
			int32_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				ptr[i] = data[i];
			}
		} break;
		case XR_SPATIAL_BUFFER_TYPE_UINT16_EXT: {
			Vector<uint16_t> data = se_extension->get_uint16_buffer(p_snapshot, buffer.bufferId);

			ret.resize(data.size());
			int count = ret.size();
			int32_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				ptr[i] = data[i];
			}
		} break;
		case XR_SPATIAL_BUFFER_TYPE_UINT32_EXT: {
			Vector<uint32_t> data = se_extension->get_uint32_buffer(p_snapshot, buffer.bufferId);

			ret.resize(data.size());
			int count = ret.size();
			int32_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				ptr[i] = data[i];
			}
		} break;
		default: {
			ERR_FAIL_V_MSG(PackedInt32Array(), "OpenXR: Unsupported buffer type for indices.");
		} break;
	}

	return ret;
}

////////////////////////////////////////////////////////////////////////////
// Spatial component mesh3d list

void OpenXRSpatialComponentMesh3DList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_transform", "index"), &OpenXRSpatialComponentMesh3DList::get_transform);
	ClassDB::bind_method(D_METHOD("get_mesh", "index"), &OpenXRSpatialComponentMesh3DList::get_mesh);
}

void OpenXRSpatialComponentMesh3DList::set_capacity(uint32_t p_capacity) {
	mesh3d_data.resize(p_capacity);

	mesh3d_list.meshCount = uint32_t(mesh3d_data.size());
	mesh3d_list.meshes = mesh3d_data.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialComponentMesh3DList::get_component_type() const {
	return XR_SPATIAL_COMPONENT_TYPE_MESH_3D_EXT;
}

void *OpenXRSpatialComponentMesh3DList::get_structure_data(void *p_next) {
	mesh3d_list.next = p_next;
	return &mesh3d_list;
}

Transform3D OpenXRSpatialComponentMesh3DList::get_transform(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh3d_data.size(), Transform3D());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	return openxr_api->transform_from_pose(mesh3d_data[p_index].origin);
}

Ref<Mesh> OpenXRSpatialComponentMesh3DList::get_mesh(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, mesh3d_data.size(), nullptr);

	// TODO implement, need to convert mesh data to Godot mesh resource

	return nullptr;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialQueryResultData

void OpenXRSpatialQueryResultData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_capacity"), &OpenXRSpatialQueryResultData::get_capacity);
	ClassDB::bind_method(D_METHOD("get_entity_id", "index"), &OpenXRSpatialQueryResultData::_get_entity_id);
	ClassDB::bind_method(D_METHOD("get_entity_state", "index"), &OpenXRSpatialQueryResultData::_get_entity_state);
}

void OpenXRSpatialQueryResultData::set_capacity(uint32_t p_capacity) {
	entity_ids.resize(p_capacity);
	entity_states.resize(p_capacity);

	query_result.entityIdCapacityInput = entity_ids.size();
	query_result.entityIds = entity_ids.ptrw();
	query_result.entityStateCapacityInput = entity_states.size();
	query_result.entityStates = entity_states.ptrw();
}

XrSpatialComponentTypeEXT OpenXRSpatialQueryResultData::get_component_type() const {
	// This component is always included and has no type.
	return XR_SPATIAL_COMPONENT_TYPE_MAX_ENUM_EXT;
}

void *OpenXRSpatialQueryResultData::get_structure_data(void *p_next) {
	query_result.next = p_next;
	query_result.entityIdCountOutput = 0;
	query_result.entityStateCountOutput = 0;
	return &query_result;
}

XrSpatialEntityIdEXT OpenXRSpatialQueryResultData::get_entity_id(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, entity_ids.size(), XR_NULL_ENTITY);

	return entity_ids[p_index];
}

uint64_t OpenXRSpatialQueryResultData::_get_entity_id(int64_t p_index) const {
	return (uint64_t)get_entity_id(p_index);
}

XrSpatialEntityTrackingStateEXT OpenXRSpatialQueryResultData::get_entity_state(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, entity_states.size(), XR_SPATIAL_ENTITY_TRACKING_STATE_MAX_ENUM_EXT);

	return entity_states[p_index];
}

OpenXRSpatialEntityTracker::EntityTrackingState OpenXRSpatialQueryResultData::_get_entity_state(int64_t p_index) const {
	return (OpenXRSpatialEntityTracker::EntityTrackingState)get_entity_state(p_index);
}

String OpenXRSpatialQueryResultData::get_entity_tracking_state_name(XrSpatialEntityTrackingStateEXT p_tracking_state) {
	XR_ENUM_SWITCH(XrSpatialEntityTrackingStateEXT, p_tracking_state)
}
