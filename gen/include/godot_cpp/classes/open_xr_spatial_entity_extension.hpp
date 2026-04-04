/**************************************************************************/
/*  open_xr_spatial_entity_extension.hpp                                  */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/open_xr_extension_wrapper.hpp>
#include <godot_cpp/classes/open_xr_structure_base.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRFutureResult;
class OpenXRSpatialCapabilityConfigurationBaseHeader;
class OpenXRSpatialComponentData;
class PackedInt64Array;

class OpenXRSpatialEntityExtension : public OpenXRExtensionWrapper {
	GDEXTENSION_CLASS(OpenXRSpatialEntityExtension, OpenXRExtensionWrapper)

public:
	enum Capability {
		CAPABILITY_PLANE_TRACKING = 1000741000,
		CAPABILITY_MARKER_TRACKING_QR_CODE = 1000743000,
		CAPABILITY_MARKER_TRACKING_MICRO_QR_CODE = 1000743001,
		CAPABILITY_MARKER_TRACKING_ARUCO_MARKER = 1000743002,
		CAPABILITY_MARKER_TRACKING_APRIL_TAG = 1000743003,
		CAPABILITY_ANCHOR = 1000762000,
	};

	enum ComponentType {
		COMPONENT_TYPE_BOUNDED_2D = 1,
		COMPONENT_TYPE_BOUNDED_3D = 2,
		COMPONENT_TYPE_PARENT = 3,
		COMPONENT_TYPE_MESH_3D = 4,
		COMPONENT_TYPE_PLANE_ALIGNMENT = 1000741000,
		COMPONENT_TYPE_MESH_2D = 1000741001,
		COMPONENT_TYPE_POLYGON_2D = 1000741002,
		COMPONENT_TYPE_PLANE_SEMANTIC_LABEL = 1000741003,
		COMPONENT_TYPE_MARKER = 1000743000,
		COMPONENT_TYPE_ANCHOR = 1000762000,
		COMPONENT_TYPE_PERSISTENCE = 1000763000,
	};

	bool supports_capability(OpenXRSpatialEntityExtension::Capability p_capability);
	bool supports_component_type(OpenXRSpatialEntityExtension::Capability p_capability, OpenXRSpatialEntityExtension::ComponentType p_component_type);
	Ref<OpenXRFutureResult> create_spatial_context(const TypedArray<Ref<OpenXRSpatialCapabilityConfigurationBaseHeader>> &p_capability_configurations, const Ref<OpenXRStructureBase> &p_next = nullptr, const Callable &p_user_callback = Callable());
	bool get_spatial_context_ready(const RID &p_spatial_context) const;
	void free_spatial_context(const RID &p_spatial_context);
	uint64_t get_spatial_context_handle(const RID &p_spatial_context) const;
	Ref<OpenXRFutureResult> discover_spatial_entities(const RID &p_spatial_context, const PackedInt64Array &p_component_types, const Ref<OpenXRStructureBase> &p_next = nullptr, const Callable &p_user_callback = Callable());
	RID update_spatial_entities(const RID &p_spatial_context, const TypedArray<RID> &p_entities, const PackedInt64Array &p_component_types, const Ref<OpenXRStructureBase> &p_next = nullptr);
	void free_spatial_snapshot(const RID &p_spatial_snapshot);
	uint64_t get_spatial_snapshot_handle(const RID &p_spatial_snapshot) const;
	RID get_spatial_snapshot_context(const RID &p_spatial_snapshot) const;
	bool query_snapshot(const RID &p_spatial_snapshot, const TypedArray<Ref<OpenXRSpatialComponentData>> &p_component_data, const Ref<OpenXRStructureBase> &p_next = nullptr);
	String get_string(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedByteArray get_uint8_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedInt32Array get_uint16_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedInt32Array get_uint32_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedFloat32Array get_float_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedVector2Array get_vector2_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	PackedVector3Array get_vector3_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const;
	RID find_spatial_entity(uint64_t p_entity_id);
	RID add_spatial_entity(const RID &p_spatial_context, uint64_t p_entity_id, uint64_t p_entity);
	RID make_spatial_entity(const RID &p_spatial_context, uint64_t p_entity_id);
	uint64_t get_spatial_entity_id(const RID &p_entity) const;
	RID get_spatial_entity_context(const RID &p_entity) const;
	void free_spatial_entity(const RID &p_entity);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		OpenXRExtensionWrapper::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRSpatialEntityExtension::Capability);
VARIANT_ENUM_CAST(OpenXRSpatialEntityExtension::ComponentType);

