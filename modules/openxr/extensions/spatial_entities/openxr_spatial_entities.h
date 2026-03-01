/**************************************************************************/
/*  openxr_spatial_entities.h                                             */
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

#include "../../openxr_structure.h"
#include "../openxr_future_extension.h"
#include "scene/resources/mesh.h"
#include "servers/xr/xr_positional_tracker.h"

#define XR_NULL_ENTITY 0x7FFFFFFF

// Wrapper class for XrSpatialCapabilityConfigurationBaseHeaderEXT
class OpenXRSpatialCapabilityConfigurationBaseHeader : public RefCounted {
	GDCLASS(OpenXRSpatialCapabilityConfigurationBaseHeader, RefCounted);

protected:
	static void _bind_methods();

public:
	virtual bool has_valid_configuration() const;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration();

	GDVIRTUAL0RC(bool, _has_valid_configuration);
	GDVIRTUAL0R(uint64_t, _get_configuration);
};

// Tracker for our spatial entities
class OpenXRSpatialEntityTracker : public XRPositionalTracker {
	GDCLASS(OpenXRSpatialEntityTracker, XRPositionalTracker);

public:
	enum EntityTrackingState {
		ENTITY_TRACKING_STATE_STOPPED = XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT,
		ENTITY_TRACKING_STATE_PAUSED = XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT,
		ENTITY_TRACKING_STATE_TRACKING = XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT,
	};

	OpenXRSpatialEntityTracker();
	virtual ~OpenXRSpatialEntityTracker();

	void set_entity(const RID &p_entity);
	RID get_entity() const;

	void set_spatial_tracking_state(const XrSpatialEntityTrackingStateEXT p_state);
	XrSpatialEntityTrackingStateEXT get_spatial_tracking_state() const;

protected:
	static void _bind_methods();

private:
	RID spatial_entity;
	XrSpatialEntityTrackingStateEXT spatial_tracking_state = XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT;

	void _set_spatial_tracking_state(const EntityTrackingState p_state);
	EntityTrackingState _get_spatial_tracking_state() const;
};

VARIANT_ENUM_CAST(OpenXRSpatialEntityTracker::EntityTrackingState)

// Wrapper class for our spatial component data returned by discovery queries
class OpenXRSpatialComponentData : public RefCounted {
	GDCLASS(OpenXRSpatialComponentData, RefCounted);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity);
	virtual XrSpatialComponentTypeEXT get_component_type() const;
	virtual void *get_structure_data(void *p_next);

	GDVIRTUAL1(_set_capacity, uint32_t);
	GDVIRTUAL0RC(uint64_t, _get_component_type);
	GDVIRTUAL1RC(uint64_t, _get_structure_data, uint64_t);
};

class OpenXRSpatialComponentBounded2DList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentBounded2DList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	Transform3D get_center_pose(int64_t p_index) const;
	Vector2 get_size(int64_t p_index) const;

private:
	Vector<XrSpatialBounded2DDataEXT> bounded2d_data;

	XrSpatialComponentBounded2DListEXT bounded2d_list = { XR_TYPE_SPATIAL_COMPONENT_BOUNDED_2D_LIST_EXT, nullptr, 0, nullptr };
};

class OpenXRSpatialComponentBounded3DList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentBounded3DList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	Transform3D get_center_pose(int64_t p_index) const;
	Vector3 get_size(int64_t p_index) const;

private:
	Vector<XrBoxf> bounded3d_data;

	XrSpatialComponentBounded3DListEXT bounded3d_list = { XR_TYPE_SPATIAL_COMPONENT_BOUNDED_3D_LIST_EXT, nullptr, 0, nullptr };
};

class OpenXRSpatialComponentParentList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentParentList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	RID get_parent(int64_t p_index) const;

private:
	Vector<XrSpatialEntityIdEXT> parent_data;

	XrSpatialComponentParentListEXT parent_list = { XR_TYPE_SPATIAL_COMPONENT_PARENT_LIST_EXT, nullptr, 0, nullptr };
};

class OpenXRSpatialComponentMesh2DList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentMesh2DList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	Transform3D get_transform(int64_t p_index) const;
	PackedVector2Array get_vertices(RID p_snapshot, int64_t p_index) const;
	PackedInt32Array get_indices(RID p_snapshot, int64_t p_index) const;

private:
	Vector<XrSpatialMeshDataEXT> mesh2d_data;

	XrSpatialComponentMesh2DListEXT mesh2d_list = { XR_TYPE_SPATIAL_COMPONENT_MESH_2D_LIST_EXT, nullptr, 0, nullptr };
};

class OpenXRSpatialComponentMesh3DList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentMesh3DList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	Transform3D get_transform(int64_t p_index) const;
	Ref<Mesh> get_mesh(int64_t p_index) const;

private:
	Vector<XrSpatialMeshDataEXT> mesh3d_data;

	XrSpatialComponentMesh3DListEXT mesh3d_list = { XR_TYPE_SPATIAL_COMPONENT_MESH_3D_LIST_EXT, nullptr, 0, nullptr };
};

class OpenXRSpatialQueryResultData : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialQueryResultData, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	int64_t get_capacity() const { return entity_ids.size(); }
	XrSpatialEntityIdEXT get_entity_id(int64_t p_index) const;
	XrSpatialEntityTrackingStateEXT get_entity_state(int64_t p_index) const;

	static String get_entity_tracking_state_name(XrSpatialEntityTrackingStateEXT p_tracking_state);

private:
	Vector<XrSpatialEntityIdEXT> entity_ids;
	Vector<XrSpatialEntityTrackingStateEXT> entity_states;

	XrSpatialComponentDataQueryResultEXT query_result = { XR_TYPE_SPATIAL_COMPONENT_DATA_QUERY_RESULT_EXT, nullptr, 0, 0, nullptr, 0, 0, nullptr };

	uint64_t _get_entity_id(int64_t p_index) const;
	OpenXRSpatialEntityTracker::EntityTrackingState _get_entity_state(int64_t p_index) const;
};
