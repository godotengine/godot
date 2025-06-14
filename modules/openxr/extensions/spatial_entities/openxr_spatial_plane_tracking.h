/**************************************************************************/
/*  openxr_spatial_plane_tracking.h                                       */
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

#include "openxr_spatial_entities.h"
#include "openxr_spatial_entity_extension.h"

// Plane tracking capability configuration
class OpenXRSpatialCapabilityConfigurationPlaneTracking : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationPlaneTracking, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	bool get_supports_mesh_2d();
	bool get_supports_polygons();
	bool get_supports_labels();

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return plane_enabled_components; }

protected:
	static void _bind_methods();

private:
	int supports_mesh_2d = -1;
	int supports_polygons = -1;
	int supports_labels = -1;

	Vector<XrSpatialComponentTypeEXT> plane_enabled_components;
	XrSpatialCapabilityConfigurationPlaneTrackingEXT plane_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_PLANE_TRACKING_EXT, nullptr, XR_SPATIAL_CAPABILITY_PLANE_TRACKING_EXT, 0, nullptr };

	PackedInt64Array _get_enabled_components() const;
};

// Plane alignment component data
class OpenXRSpatialComponentPlaneAlignmentList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentPlaneAlignmentList, OpenXRSpatialComponentData);

	// TODO add enum that mirrors XrSpatialPlaneAlignmentEXT

public:
	enum OpenXrSpatialPlaneAlignment { // Must mirror XrSpatialPlaneAlignmentEXT
		OPENXR_SPATIAL_PLANE_ALIGNMENT_HORIZONTAL_UPWARD = 0,
		OPENXR_SPATIAL_PLANE_ALIGNMENT_HORIZONTAL_DOWNWARD = 1,
		OPENXR_SPATIAL_PLANE_ALIGNMENT_VERTICAL = 2,
		OPENXR_SPATIAL_PLANE_ALIGNMENT_ARBITRARY = 3,
		OPENXR_SPATIAL_PLANE_ALIGNMENT_MAX_ENUM = 0x7FFFFFFF
	};

	virtual void set_capacity(uint32_t p_entity_id_capacity, uint32_t p_entity_state_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	OpenXrSpatialPlaneAlignment get_plane_alignment(int64_t p_index) const;

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialPlaneAlignmentEXT> plane_alignment_data;

	XrSpatialComponentPlaneAlignmentListEXT plane_alignment_list = { XR_TYPE_SPATIAL_COMPONENT_PLANE_ALIGNMENT_LIST_EXT, nullptr, 0, nullptr };
};

VARIANT_ENUM_CAST(OpenXRSpatialComponentPlaneAlignmentList::OpenXrSpatialPlaneAlignment);

class OpenXRSpatialComponentPolygon2DList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentPolygon2DList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_entity_id_capacity, uint32_t p_entity_state_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	Transform3D get_transform(int64_t p_index) const;
	PackedVector2Array get_vertices(RID p_snapshot, int64_t p_index) const;

private:
	Vector<XrSpatialPolygon2DDataEXT> polygon2d_data;

	XrSpatialComponentPolygon2DListEXT polygon2d_list = { XR_TYPE_SPATIAL_COMPONENT_POLYGON_2D_LIST_EXT, nullptr, 0, nullptr };
};

// Plane semantic label component data.
class OpenXRSpatialComponentPlaneSemanticLabelList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentPlaneSemanticLabelList, OpenXRSpatialComponentData);

public:
	enum OpenXrSpatialPlaneSemanticLabel { // Must mirror XrSpatialPlaneSemanticLabelEXT
		OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_UNCATEGORIZED = 1,
		OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_FLOOR = 2,
		OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_WALL = 3,
		OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_CEILING = 4,
		OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_TABLE = 5,
		OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_MAX_ENUM = 0x7FFFFFFF
	};

	virtual void set_capacity(uint32_t p_entity_id_capacity, uint32_t p_entity_state_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	OpenXrSpatialPlaneSemanticLabel get_plane_semantic_label(int64_t p_index) const;

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialPlaneSemanticLabelEXT> plane_semantic_label_data;

	XrSpatialComponentPlaneSemanticLabelListEXT plane_semantic_label_list = { XR_TYPE_SPATIAL_COMPONENT_PLANE_SEMANTIC_LABEL_LIST_EXT, nullptr, 0, nullptr };
};

VARIANT_ENUM_CAST(OpenXRSpatialComponentPlaneSemanticLabelList::OpenXrSpatialPlaneSemanticLabel);

// Plane tracker
class OpenXRPlaneTracker : public OpenXRSpatialEntityTracker {
	GDCLASS(OpenXRPlaneTracker, OpenXRSpatialEntityTracker);

public:
	void set_bounds_size(Vector2 p_bounds_size);
	Vector2 get_bounds_size() const;

	void set_plane_alignment(OpenXRSpatialComponentPlaneAlignmentList::OpenXrSpatialPlaneAlignment p_plane_alignment);
	OpenXRSpatialComponentPlaneAlignmentList::OpenXrSpatialPlaneAlignment get_plane_alignment() const;

	void set_plane_label(OpenXRSpatialComponentPlaneSemanticLabelList::OpenXrSpatialPlaneSemanticLabel p_plane_label);
	OpenXRSpatialComponentPlaneSemanticLabelList::OpenXrSpatialPlaneSemanticLabel get_plane_label() const;

	void set_mesh_data(const Transform3D &p_origin, const PackedVector2Array &p_vertices, const PackedInt64Array &p_indices = PackedInt64Array());
	void clear_mesh_data();

	Transform3D get_mesh_offset() const;
	Ref<Mesh> get_mesh() const;
	Ref<Shape3D> get_shape(real_t p_thickness = 0.01) const;

protected:
	static void _bind_methods();

private:
	Vector2 bounds_size;
	OpenXRSpatialComponentPlaneAlignmentList::OpenXrSpatialPlaneAlignment plane_alignment = OpenXRSpatialComponentPlaneAlignmentList::OpenXrSpatialPlaneAlignment::OPENXR_SPATIAL_PLANE_ALIGNMENT_MAX_ENUM;
	OpenXRSpatialComponentPlaneSemanticLabelList::OpenXrSpatialPlaneSemanticLabel plane_label = OpenXRSpatialComponentPlaneSemanticLabelList::OpenXrSpatialPlaneSemanticLabel::OPENXR_SPATIAL_PLANE_SEMANTIC_LABEL_MAX_ENUM;

	// Mesh data (if we have this)
	struct MeshData {
		bool has_mesh = false;
		Transform3D origin;
		PackedVector2Array vertices;
		PackedInt64Array indices;
	} mesh;
};

// Plane tracking logic
class OpenXRSpatialPlaneTrackingCapability : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialPlaneTrackingCapability, OpenXRExtensionWrapper);

protected:
	static void _bind_methods();

public:
	static OpenXRSpatialPlaneTrackingCapability *get_singleton();

	OpenXRSpatialPlaneTrackingCapability();
	virtual ~OpenXRSpatialPlaneTrackingCapability() override;

	virtual HashMap<String, bool *> get_requested_extensions() override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;

	bool is_supported();

private:
	static OpenXRSpatialPlaneTrackingCapability *singleton;
	bool spatial_plane_tracking_ext = false;

	RID spatial_context;
	Ref<OpenXRFutureResult> discovery_query_result;

	Ref<OpenXRSpatialCapabilityConfigurationPlaneTracking> plane_configuration;

	// Discovery logic
	Ref<OpenXRFutureResult> _create_spatial_context();
	void _on_spatial_context_created(RID p_spatial_context);

	void _on_spatial_discovery_recommended(RID p_spatial_context);

	Ref<OpenXRFutureResult> _start_entity_discovery();
	void _on_spatial_discovery_completed(RID p_snapshot);

	// Trackers
	HashMap<XrSpatialEntityIdEXT, Ref<OpenXRPlaneTracker>> plane_trackers;
};
