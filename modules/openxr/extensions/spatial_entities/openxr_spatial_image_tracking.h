/**************************************************************************/
/*  openxr_spatial_image_tracking.h                                       */
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

#include "core/io/image.h"

class OpenXRSpatialCapabilityConfigurationImageTracking : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationImageTracking, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return enabled_components; }

	void add_image_tracking_database(RID p_image_database);
	void remove_image_tracking_database(RID p_image_database);

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialComponentTypeEXT> enabled_components;
	LocalVector<RID> image_database_rids;
	LocalVector<XrSpatialImageTrackingDatabaseEXT> image_tracking_databases;
	XrSpatialCapabilityConfigurationImageTrackingEXT image_tracking_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_IMAGE_TRACKING_EXT, nullptr, XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, 0, nullptr, 0, nullptr };

   	PackedInt64Array _get_enabled_components() const;
};

class OpenXRSpatialComponentImage2DList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentImage2DList, OpenXRSpatialComponentData);

protected:
	static void _bind_methods();

public:
	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override { return XR_SPATIAL_COMPONENT_TYPE_IMAGE_2D_EXT; }
	virtual void *get_structure_data(void *p_next) override;

	XrSpatialImageTrackingDatabaseEXT get_image_tracking_database(int64_t p_index) const;
	uint32_t get_reference_image_index(int64_t p_index) const;

private:
	Vector<XrSpatialImage2DDataEXT> image2d_data;

	XrSpatialComponentImage2DListEXT image2d_list = { XR_TYPE_SPATIAL_COMPONENT_IMAGE_2D_LIST_EXT, nullptr, 0, nullptr };

	RID _get_image_tracking_database(int64_t p_index) const;
};

class OpenXRImageTracker : public OpenXRSpatialEntityTracker {
	GDCLASS(OpenXRImageTracker, OpenXRSpatialEntityTracker);

public:
	void set_bounds_size(const Vector2 &p_bounds_size);
	Vector2 get_bounds_size() const;

	String get_image_name() const;
	void set_image_info(XrSpatialImageTrackingDatabaseEXT p_image_tracking_database, uint32_t p_image_index);

protected:
	static void _bind_methods();

private:
	Vector2 bounds_size;
	XrSpatialImageTrackingDatabaseEXT image_tracking_database = XR_NULL_HANDLE;
	uint32_t image_index = 0;
	String image_name;
};

// Image tracking logic
class OpenXRSpatialImageTrackingCapability : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialImageTrackingCapability, OpenXRExtensionWrapper);

public:
	static OpenXRSpatialImageTrackingCapability *get_singleton();

	OpenXRSpatialImageTrackingCapability();
	virtual ~OpenXRSpatialImageTrackingCapability() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_instance_destroyed() override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;

	virtual void on_process() override;

	bool is_supported() { return spatial_image_tracking_ext && spatial_image_tracking_supported; }

	// Image tracking database
	RID image_database_create();
	void image_database_free(RID p_image_database);
	void image_database_add(RID p_image_database, String p_name, Ref<Image> p_image, float p_physical_width = 0.0, bool p_is_static = false);
	Ref<OpenXRFutureResult> image_database_submit(RID p_image_database, const Callable &p_user_callback = Callable(), const Callable &p_failure_callback = Callable());
	XrSpatialImageTrackingDatabaseEXT image_database_get_handle(RID p_image_database);
	String get_image_name(XrSpatialImageTrackingDatabaseEXT p_image_database, uint32_t p_image_index);

	OpenXRSpatialEntityExtension::TrackingState get_built_in_tracking_state() { return builtin_tracking_state; }
	bool start_built_in_tracking(RID p_image_database = RID());
	void stop_built_in_tracking(bool p_clear_trackers = true);

	Ref<OpenXRFutureResult> start_entity_discovery(RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_create = nullptr, Ref<OpenXRStructureBase> p_next_snapshot_query = nullptr, const Callable &p_user_callback = Callable());

	static String get_reference_image_format_name(XrSpatialReferenceImageFormatEXT p_image_format);

protected:
	static void _bind_methods();

private:
	static OpenXRSpatialImageTrackingCapability *singleton;

	bool spatial_image_tracking_ext = false;
	bool spatial_image_tracking_supported = false;
	bool spatial_image_automatic_size_supported = false;
	bool spatial_image_fixed_size_supported = false;
	bool spatial_image_static_hint_supported = false;
	LocalVector<XrSpatialReferenceImageFormatEXT> supported_reference_image_formats;

	// Image tracking database
	struct TrackingImage {
		String name;
		XrSpatialReferenceImageFormatEXT format;
		Size2i size;
		Vector<uint8_t> image_data;
		float physical_width;
		bool is_static;

		// OpenXR data
		LocalVector<XrSpatialReferenceImagePlaneEXT> image_planes;
		XrSpatialImageStaticOptimizationEXT image_static;
		XrSpatialImageSizeEXT image_size;
	};

	struct TrackingImageDatabase {
		LocalVector<TrackingImage> tracking_images;

		// Submitting database
		Ref<OpenXRFutureResult> future_result;
		XrSpatialImageTrackingDatabaseEXT database = XR_NULL_HANDLE;
	};

	RID_Owner<TrackingImageDatabase, true> image_database_owner;
	void _on_image_database_created_ready(Ref<OpenXRFutureResult> p_future_result, RID p_image_database, const Callable &p_user_callback, const Callable &p_failure_callback);
	
	OpenXRSpatialEntityExtension::TrackingState builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_NOT_ACTIVE;
	Ref<OpenXRFutureResult> built_in_future;
	RID image_db;
	RID spatial_context;

	void _on_image_database_created(RID p_image_database);
	void _on_image_database_creation_failed(RID p_image_database, int p_xr_result, bool p_is_completion_failure);

	bool need_discovery = false;
	int discovery_cooldown = 0;
	Ref<OpenXRFutureResult> discovery_query_result;

	Ref<OpenXRSpatialCapabilityConfigurationImageTracking> image_tracking_configuration;
	TypedArray<OpenXRSpatialComponentData> image_tracking_component_data;

	// Discovery logic
	Ref<OpenXRFutureResult> _create_spatial_context(RID p_image_database);
	void _on_spatial_context_created(RID p_spatial_context);
	void _on_spatial_context_creation_failed(int p_xr_result, bool p_is_completion_failure);

	void _on_spatial_discovery_recommended(RID p_spatial_context);

	void _process_snapshot(RID p_snapshot, RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_query, const Callable &p_user_callback);

	// Trackers; maps each Spatial Context RID to their image entities and trackers
	HashMap<RID, HashMap<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>>> image_trackers;

	// OpenXR API call wrappers
	EXT_PROTO_XRRESULT_FUNC6(xrEnumerateSpatialReferenceImageFormatsEXT, (XrInstance), instance, (XrSystemId), systemId, (XrSpatialCapabilityEXT), capability, (uint32_t), formatCapacityInput, (uint32_t *), formatCountOutput, (XrSpatialReferenceImageFormatEXT *), formats);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialImageTrackingDatabaseAsyncEXT, (XrSession), session, (const XrSpatialImageTrackingDatabaseCreateInfoEXT *), createInfo, (XrFutureEXT *), future);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateSpatialImageTrackingDatabaseCompleteEXT, (XrSession), session, (XrFutureEXT), future, (XrCreateSpatialImageTrackingDatabaseCompletionEXT *), completion);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpatialImageTrackingDatabaseEXT, (XrSpatialImageTrackingDatabaseEXT), database);
};
