/**************************************************************************/
/*  openxr_spatial_image_tracking.cpp                                     */
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

#include "openxr_spatial_image_tracking.h"

#include "../../openxr_api.h"
#include "../../openxr_util.h"
#include "openxr_spatial_entity_extension.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "servers/xr/xr_server.h"

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialCapabilityConfigurationImageTracking

void OpenXRSpatialCapabilityConfigurationImageTracking::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_image_tracking_database", "image_database"), &OpenXRSpatialCapabilityConfigurationImageTracking::add_image_tracking_database);
	ClassDB::bind_method(D_METHOD("remove_image_tracking_database", "image_database"), &OpenXRSpatialCapabilityConfigurationImageTracking::remove_image_tracking_database);

	ClassDB::bind_method(D_METHOD("get_enabled_components"), &OpenXRSpatialCapabilityConfigurationImageTracking::_get_enabled_components);
}

bool OpenXRSpatialCapabilityConfigurationImageTracking::has_valid_configuration() const {
	OpenXRSpatialImageTrackingCapability *capability = OpenXRSpatialImageTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, false);

	if (!capability->is_supported()) {
		return false;
	}

	if (image_database_rids.is_empty()) {
		return false;
	}

	return true;
}

XrSpatialCapabilityConfigurationBaseHeaderEXT *OpenXRSpatialCapabilityConfigurationImageTracking::get_configuration() {
	OpenXRSpatialImageTrackingCapability *capability = OpenXRSpatialImageTrackingCapability::get_singleton();
	ERR_FAIL_NULL_V(capability, nullptr);

	if (capability->is_supported()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL_V(se_extension, nullptr);

		// Copy image databases
		image_tracking_databases.resize(image_database_rids.size());
		for (uint32_t i = 0; i < image_database_rids.size(); i++) {
			image_tracking_databases[i] = capability->image_database_get_handle(image_database_rids[i]);
		}
		image_tracking_config.imageTrackingDatabaseCount = image_tracking_databases.size();
		image_tracking_config.imageTrackingDatabases = image_tracking_databases.ptr();

		enabled_components.clear();

		// Guaranteed components:
		enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_IMAGE_2D_EXT);
		enabled_components.push_back(XR_SPATIAL_COMPONENT_TYPE_BOUNDED_2D_EXT);

		// Set up our enabled components.
		image_tracking_config.enabledComponentCount = enabled_components.size();
		image_tracking_config.enabledComponents = enabled_components.ptr();

		// and return this.
		return (XrSpatialCapabilityConfigurationBaseHeaderEXT *)&image_tracking_config;
	}

	return nullptr;
}

void OpenXRSpatialCapabilityConfigurationImageTracking::add_image_tracking_database(RID p_image_database) {
	if (!image_database_rids.has(p_image_database)) {
		image_database_rids.push_back(p_image_database);
	}
}

void OpenXRSpatialCapabilityConfigurationImageTracking::remove_image_tracking_database(RID p_image_database) {
	if (image_database_rids.has(p_image_database)) {
		image_database_rids.erase(p_image_database);
	}
}

PackedInt64Array OpenXRSpatialCapabilityConfigurationImageTracking::_get_enabled_components() const {
	PackedInt64Array components;

	for (const XrSpatialComponentTypeEXT &component_type : enabled_components) {
		components.push_back((int64_t)component_type);
	}

	return components;
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialCapabilityConfigurationImageTracking

void OpenXRSpatialComponentImage2DList::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_image_tracking_database", "index"), &OpenXRSpatialComponentImage2DList::_get_image_tracking_database);
	ClassDB::bind_method(D_METHOD("get_reference_image_index", "index"), &OpenXRSpatialComponentImage2DList::get_reference_image_index);
}

void OpenXRSpatialComponentImage2DList::set_capacity(uint32_t p_capacity) {
	image2d_data.resize(p_capacity);

	image2d_list.imageCount = uint32_t(image2d_data.size());
	image2d_list.images = image2d_data.ptrw();
}

void *OpenXRSpatialComponentImage2DList::get_structure_data(void *p_next) {
	image2d_list.next = p_next;
	return &image2d_list;
}

XrSpatialImageTrackingDatabaseEXT OpenXRSpatialComponentImage2DList::get_image_tracking_database(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, image2d_data.size(), XR_NULL_HANDLE);
	return image2d_data[p_index].imageTrackingDatabase;
}

uint32_t OpenXRSpatialComponentImage2DList::get_reference_image_index(int64_t p_index) const {
	ERR_FAIL_INDEX_V(p_index, image2d_data.size(), 0);
	return image2d_data[p_index].referenceImageIndex;
}

RID OpenXRSpatialComponentImage2DList::_get_image_tracking_database(int64_t p_index) const {
	// TODO implement, lookup tracking database RID in our image tracking extension


	return RID();
}

////////////////////////////////////////////////////////////////////////////
// OpenXRImageTracker

void OpenXRImageTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bounds_size", "bounds_size"), &OpenXRImageTracker::set_bounds_size);
	ClassDB::bind_method(D_METHOD("get_bounds_size"), &OpenXRImageTracker::get_bounds_size);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bounds_size"), "set_bounds_size", "get_bounds_size");

	ClassDB::bind_method(D_METHOD("get_image_name"), &OpenXRImageTracker::get_image_name);
}

void OpenXRImageTracker::set_bounds_size(const Vector2 &p_bounds_size) {
	bounds_size = p_bounds_size;
}

Vector2 OpenXRImageTracker::get_bounds_size() const {
	return bounds_size;
}

String OpenXRImageTracker::get_image_name() const {
	return image_name;
}

void OpenXRImageTracker::set_image_info(XrSpatialImageTrackingDatabaseEXT p_image_tracking_database, uint32_t p_image_index) {
	if (image_tracking_database != p_image_tracking_database || image_index != p_image_index) {
		image_tracking_database = p_image_tracking_database;
		image_index = p_image_index;

		OpenXRSpatialImageTrackingCapability *image_tracking_capability = OpenXRSpatialImageTrackingCapability::get_singleton();
		ERR_FAIL_NULL(image_tracking_capability);
		
		image_name = image_tracking_capability->get_image_name(image_tracking_database, image_index);
	}
}

////////////////////////////////////////////////////////////////////////////
// OpenXRSpatialImageTrackingCapability

OpenXRSpatialImageTrackingCapability *OpenXRSpatialImageTrackingCapability::singleton = nullptr;

OpenXRSpatialImageTrackingCapability *OpenXRSpatialImageTrackingCapability::get_singleton() {
	return singleton;
}

OpenXRSpatialImageTrackingCapability::OpenXRSpatialImageTrackingCapability() {
	singleton = this;
}

OpenXRSpatialImageTrackingCapability::~OpenXRSpatialImageTrackingCapability() {
	singleton = nullptr;
}

void OpenXRSpatialImageTrackingCapability::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_supported"), &OpenXRSpatialImageTrackingCapability::is_supported);
	ClassDB::bind_method(D_METHOD("start_entity_discovery", "spatial_context", "component_data", "next_snapshot_create", "next_snapshot_query", "user_callback"), &OpenXRSpatialImageTrackingCapability::start_entity_discovery, DEFVAL(Variant()), DEFVAL(Variant()), DEFVAL(Callable()));

	ClassDB::bind_method(D_METHOD("image_database_create"), &OpenXRSpatialImageTrackingCapability::image_database_create);
	ClassDB::bind_method(D_METHOD("image_database_free", "image_database"), &OpenXRSpatialImageTrackingCapability::image_database_free);
	ClassDB::bind_method(D_METHOD("image_database_add", "image_database", "name", "image", "physical_width", "is_static"), &OpenXRSpatialImageTrackingCapability::image_database_add, DEFVAL(0.0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("image_database_submit", "image_database", "user_callback", "fail_callback"), &OpenXRSpatialImageTrackingCapability::image_database_submit, DEFVAL(Callable()), DEFVAL(Callable()));

	ClassDB::bind_method(D_METHOD("get_built_in_tracking_state"), &OpenXRSpatialImageTrackingCapability::get_built_in_tracking_state);
	ClassDB::bind_method(D_METHOD("start_built_in_tracking", "image_database"), &OpenXRSpatialImageTrackingCapability::start_built_in_tracking);
	ClassDB::bind_method(D_METHOD("stop_built_in_tracking", "clear_trackers"), &OpenXRSpatialImageTrackingCapability::stop_built_in_tracking, DEFVAL(true));
}

HashMap<String, bool *> OpenXRSpatialImageTrackingCapability::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/enabled") && GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/spatial_image_tracking/enable")) {
		request_extensions[XR_EXT_SPATIAL_IMAGE_TRACKING_EXTENSION_NAME] = &spatial_image_tracking_ext;
	}

	return request_extensions;
}

void OpenXRSpatialImageTrackingCapability::on_instance_created(const XrInstance p_instance) {
	if (spatial_image_tracking_ext) {
		// Obtain pointers
		EXT_INIT_XR_FUNC(xrEnumerateSpatialReferenceImageFormatsEXT);
		EXT_INIT_XR_FUNC(xrCreateSpatialImageTrackingDatabaseAsyncEXT);
		EXT_INIT_XR_FUNC(xrCreateSpatialImageTrackingDatabaseCompleteEXT);
		EXT_INIT_XR_FUNC(xrDestroySpatialImageTrackingDatabaseEXT);
	}
}

void OpenXRSpatialImageTrackingCapability::on_instance_destroyed() {
	xrEnumerateSpatialReferenceImageFormatsEXT_ptr = nullptr;
	xrCreateSpatialImageTrackingDatabaseAsyncEXT_ptr = nullptr;
	xrCreateSpatialImageTrackingDatabaseCompleteEXT_ptr = nullptr;
	xrDestroySpatialImageTrackingDatabaseEXT_ptr = nullptr;
}

void OpenXRSpatialImageTrackingCapability::on_session_created(const XrSession p_session) {
	if (!spatial_image_tracking_ext) {
		return;
	}

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Enumerate supported formats
	uint32_t capacity = 0;
	XrResult result = xrEnumerateSpatialReferenceImageFormatsEXT(openxr_api->get_instance(), openxr_api->get_system_id(), XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, 0, &capacity, nullptr);
	if (XR_FAILED(result)) {
		ERR_FAIL_MSG("OpenXR: Failed to obtain supported image format count [" + openxr_api->get_error_string(result) + "]");
	}
	if (capacity > 0) {
		supported_reference_image_formats.resize(capacity);

		result = xrEnumerateSpatialReferenceImageFormatsEXT(openxr_api->get_instance(), openxr_api->get_system_id(), XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, supported_reference_image_formats.size(), &capacity, supported_reference_image_formats.ptr());
		if (XR_FAILED(result)) {
			ERR_FAIL_MSG("OpenXR: Failed to obtain supported image formats [" + openxr_api->get_error_string(result) + "]");
		}
	}


	spatial_image_tracking_supported = se_extension->supports_component_type(XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, XR_SPATIAL_COMPONENT_TYPE_IMAGE_2D_EXT);
	if (!spatial_image_tracking_supported) {
		// Supported by XR runtime but not by device? We're done.
		return;
	}

	spatial_image_automatic_size_supported = se_extension->supports_capability_feature(XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, XR_SPATIAL_CAPABILITY_FEATURE_IMAGE_TRACKING_AUTOMATIC_SIZE_IMAGES_EXT);
	spatial_image_fixed_size_supported = se_extension->supports_capability_feature(XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, XR_SPATIAL_CAPABILITY_FEATURE_IMAGE_TRACKING_FIXED_SIZE_IMAGES_EXT);
	spatial_image_static_hint_supported = se_extension->supports_capability_feature(XR_SPATIAL_CAPABILITY_IMAGE_TRACKING_EXT, XR_SPATIAL_CAPABILITY_FEATURE_IMAGE_TRACKING_STATIC_IMAGES_EXT);

	se_extension->connect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_spatial_discovery_recommended));

	bool enable_builtin = GLOBAL_GET_CACHED(bool, "xr/openxr/extensions/spatial_entity/spatial_image_tracking/enable_builtin_detection");
	if (enable_builtin) {
		start_built_in_tracking();
	}
}

void OpenXRSpatialImageTrackingCapability::on_session_destroyed() {
	// Stop our built in tracking (if applicable), this will also clean up any images previously detected.
	stop_built_in_tracking();

	// Destroy image databases
	for (const RID &rid : image_database_owner.get_owned_list()) {
		image_database_free(rid);
	}

	// Free and unregister all our image trackers
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);
	for (const KeyValue<RID, HashMap<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>>> &images : image_trackers) {
		for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>> &image_tracker : images.value) {
			xr_server->remove_tracker(image_tracker.value);
		}
	}
	image_trackers.clear();

	// Disconnect our discovery
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	se_extension->disconnect(SNAME("spatial_discovery_recommended"), callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_spatial_discovery_recommended));

	supported_reference_image_formats.clear();
}

void OpenXRSpatialImageTrackingCapability::on_process() {
	if (!spatial_context.is_valid()) {
		return;
	}

	// Protection against image discovery happening too often.
	if (discovery_cooldown > 0) {
		discovery_cooldown--;
	}

	// Check if we need to start our discovery.
	if (need_discovery && discovery_cooldown == 0 && !discovery_query_result.is_valid()) {
		print_line("OpenXR: trigger image discovery"); // TESTING REMOVE!

		need_discovery = false;
		discovery_cooldown = 60; // Set our cooldown to 60 frames, it doesn't need to be an exact science.

		if (image_tracking_component_data.is_empty()) {
			// We always need a query result data object, and it must be first
			Ref<OpenXRSpatialQueryResultData> query_result_data;
			query_result_data.instantiate();
			image_tracking_component_data.push_back(query_result_data);

			Ref<OpenXRSpatialComponentBounded2DList> bounded2d_list;
			bounded2d_list.instantiate();
			image_tracking_component_data.push_back(bounded2d_list);

			Ref<OpenXRSpatialComponentImage2DList> image2d_list;
			image2d_list.instantiate();
			image_tracking_component_data.push_back(image2d_list);
		}

		discovery_query_result = start_entity_discovery(spatial_context, image_tracking_component_data);
	}
}

RID OpenXRSpatialImageTrackingCapability::image_database_create() {
	ERR_FAIL_COND_V(!spatial_image_tracking_ext, RID());

	return image_database_owner.make_rid();
}

void OpenXRSpatialImageTrackingCapability::image_database_free(RID p_image_database) {
	ERR_FAIL_COND(!spatial_image_tracking_ext);

	TrackingImageDatabase *image_database = image_database_owner.get_or_null(p_image_database);
	ERR_FAIL_NULL(image_database);

	if (image_database->future_result.is_valid()) {
		if (image_database->future_result->get_status() == OpenXRFutureResult::RESULT_RUNNING) {
			image_database->future_result->cancel_future();
		}

		image_database->future_result.unref();
	}

	if (image_database->database != XR_NULL_HANDLE) {
		OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
		ERR_FAIL_NULL(openxr_api);

		XrResult result = xrDestroySpatialImageTrackingDatabaseEXT(image_database->database);
		if (XR_FAILED(result)) {
			WARN_PRINT("OpenXR: Failed to destroy image database [" + openxr_api->get_error_string(result) + "]");
		}

		image_database->database = XR_NULL_HANDLE;
	}

	image_database_owner.free(p_image_database);
}

void OpenXRSpatialImageTrackingCapability::image_database_add(RID p_image_database, String p_name, Ref<Image> p_image, float p_physical_width, bool p_is_static) {
	ERR_FAIL_COND(!spatial_image_tracking_ext);

	TrackingImageDatabase *image_database = image_database_owner.get_or_null(p_image_database);
	ERR_FAIL_NULL(image_database);

	// Already submitted?
	ERR_FAIL_COND_MSG(image_database->database != XR_NULL_HANDLE || image_database->future_result.is_valid(), "Can't add images to a tracking image database after it has been submitted.");

	TrackingImage tracking_image;
	switch (p_image->get_format()) {
		case Image::FORMAT_RGB8: {
			tracking_image.format = XR_SPATIAL_REFERENCE_IMAGE_FORMAT_RGB_888_EXT;
		} break;
		case Image::FORMAT_RGBA8: {
			tracking_image.format = XR_SPATIAL_REFERENCE_IMAGE_FORMAT_RGBA_8888_EXT;
		} break;
		default: {
			// Currently don't support XR_SPATIAL_REFERENCE_IMAGE_FORMAT_YUV_420_888_EXT

			// Q: check if we can add a conversion to RGB(A)8 here if the format is different.
			// Or always fail in this condition?

			ERR_FAIL_MSG("OpenXR: Unsupported image format, must be FORMAT_RGB8 or FORMAT_RGBA8.");
		} break;
	}

	ERR_FAIL_COND(!supported_reference_image_formats.has(tracking_image.format));

	tracking_image.name = p_name;
	tracking_image.size = p_image->get_size();
	tracking_image.image_data = p_image->get_data();
	tracking_image.physical_width = p_physical_width;
	tracking_image.is_static = p_is_static;

	print_verbose("OpenXR: Added image " + p_name + " (" + itos(tracking_image.size.x) + ", " + itos(tracking_image.size.y) + ")")

	image_database->tracking_images.push_back(tracking_image);
}

Ref<OpenXRFutureResult> OpenXRSpatialImageTrackingCapability::image_database_submit(RID p_image_database, const Callable &p_user_callback, const Callable &p_failure_callback) {
	ERR_FAIL_COND_V(!spatial_image_tracking_ext, Ref<OpenXRFutureResult>());

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Ref<OpenXRFutureResult>());

	OpenXRFutureExtension *future_api = OpenXRFutureExtension::get_singleton();
	ERR_FAIL_NULL_V(future_api, Ref<OpenXRFutureResult>());

	TrackingImageDatabase *image_database = image_database_owner.get_or_null(p_image_database);
	ERR_FAIL_NULL_V(image_database, Ref<OpenXRFutureResult>());

	// Already submitted?
	ERR_FAIL_COND_V(image_database->database != XR_NULL_HANDLE || image_database->future_result.is_valid(), Ref<OpenXRFutureResult>());

	thread_local LocalVector<XrSpatialReferenceImageEXT> reference_images;
	reference_images.resize(image_database->tracking_images.size());

	for (uint32_t i = 0; i < image_database->tracking_images.size(); i++) {
		TrackingImage &tracking_image = image_database->tracking_images[i];

		switch (tracking_image.format) {
			case XR_SPATIAL_REFERENCE_IMAGE_FORMAT_RGB_888_EXT: {
				tracking_image.image_planes.resize(3);
				for (uint32_t p = 0; p < 3; p++) {
					tracking_image.image_planes[p].bufferSize = tracking_image.image_data.size() - p;
					tracking_image.image_planes[p].buffer = tracking_image.image_data.ptr() + p;
					tracking_image.image_planes[p].rowStride = 3;
					tracking_image.image_planes[p].pixelStride = tracking_image.size.x * 3;
				};
			} break;
			case XR_SPATIAL_REFERENCE_IMAGE_FORMAT_RGBA_8888_EXT: {
				tracking_image.image_planes.resize(4);
				for (uint32_t p = 0; p < 4; p++) {
					tracking_image.image_planes[p].bufferSize = tracking_image.image_data.size() - p;
					tracking_image.image_planes[p].buffer = tracking_image.image_data.ptr() + p;
					tracking_image.image_planes[p].rowStride = 4;
					tracking_image.image_planes[p].pixelStride = tracking_image.size.x * 4;
				};
			} break;
			default: {
				// Huh?
			} break;
		};	

		void *next = nullptr;

		if (spatial_image_static_hint_supported) {
			tracking_image.image_static.type = XR_TYPE_SPATIAL_IMAGE_STATIC_OPTIMIZATION_EXT;
			tracking_image.image_static.next = next;
			tracking_image.image_static.optimizeForStaticImage = tracking_image.is_static;
			next = &tracking_image.image_static;
		}

		if (spatial_image_fixed_size_supported && tracking_image.physical_width > 0.0) {
			tracking_image.image_size.type = XR_TYPE_SPATIAL_IMAGE_SIZE_EXT;
			tracking_image.image_size.next = next;
			tracking_image.image_size.physicalWidth = tracking_image.physical_width;
			next = &tracking_image.image_size;
		} else if (!spatial_image_automatic_size_supported) {
			WARN_PRINT_ONCE("OpenXR: Image tracking, automatic physical image size is not supported, runtime may not recognise your image without setting a size.");
		}

		reference_images[i].type = XR_TYPE_SPATIAL_REFERENCE_IMAGE_EXT;
		reference_images[i].next = next;
		reference_images[i].width = tracking_image.size.x;
		reference_images[i].height = tracking_image.size.y;
		reference_images[i].format = tracking_image.format;
		reference_images[i].planeCount = tracking_image.image_planes.size();
		reference_images[i].planes = tracking_image.image_planes.ptr();
	}

	XrSpatialImageTrackingDatabaseCreateInfoEXT create_info = {
		XR_TYPE_SPATIAL_IMAGE_TRACKING_DATABASE_CREATE_INFO_EXT,
		nullptr,
		reference_images.size(),
		reference_images.ptr()
	};

	XrFutureEXT future;
	XrResult result = xrCreateSpatialImageTrackingDatabaseAsyncEXT(openxr_api->get_session(), &create_info, &future);
	ERR_FAIL_COND_V_MSG(XR_FAILED(result), Ref<OpenXRFutureResult>(), "OpenXR: Failed to create image tracking database [" + openxr_api->get_error_string(result) + "]");

	// Create our future result
	image_database->future_result = future_api->register_future(future, callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_image_database_created_ready).bind(p_image_database, p_user_callback, p_failure_callback));

	return image_database->future_result;
}

XrSpatialImageTrackingDatabaseEXT OpenXRSpatialImageTrackingCapability::image_database_get_handle(RID p_image_database) {
	TrackingImageDatabase *image_database = image_database_owner.get_or_null(p_image_database);
	ERR_FAIL_NULL_V(image_database, XR_NULL_HANDLE);

	return image_database->database;
}

String OpenXRSpatialImageTrackingCapability::get_image_name(XrSpatialImageTrackingDatabaseEXT p_image_database, uint32_t p_image_index) {
	for (const RID &rid : image_database_owner.get_owned_list()) {
		TrackingImageDatabase *image_database = image_database_owner.get_or_null(rid);
		if (image_database && image_database->database == p_image_database) {
			if (p_image_index >= image_database->tracking_images.size()) {
				// Out of bounds? Just return empty!
				return "";
			}

			return image_database->tracking_images[p_image_index].name;
		}
	}

	// Not found? Just return empty.
	return "";
}

void OpenXRSpatialImageTrackingCapability::_on_image_database_created_ready(Ref<OpenXRFutureResult> p_future_result, RID p_image_database, const Callable &p_user_callback, const Callable &p_failure_callback) {
	// Complete context creation...
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	XrCreateSpatialImageTrackingDatabaseCompletionEXT completion = {
		XR_TYPE_CREATE_SPATIAL_IMAGE_TRACKING_DATABASE_COMPLETION_EXT, // type
		nullptr, // next
		XR_RESULT_MAX_ENUM, // futureResult
		XR_NULL_HANDLE // database
	};
	XrResult result = xrCreateSpatialImageTrackingDatabaseCompleteEXT(openxr_api->get_session(), p_future_result->get_future(), &completion);
	if (XR_FAILED(result)) { // Did our xrCreateSpatialContextCompleteEXT call fail?
		if (p_failure_callback.is_valid()) {
			p_failure_callback.call(p_image_database, int(result), false);
		}

		// Log issue and fail.
		ERR_FAIL_MSG("OpenXR: Failed to complete image database create future [" + openxr_api->get_error_string(result) + "]");
	}
	if (XR_FAILED(completion.futureResult)) { // Did our completion fail?
		if (p_failure_callback.is_valid()) {
			p_failure_callback.call(p_image_database, int(completion.futureResult), true);
		}

		// Log issue and fail.
		ERR_FAIL_MSG("OpenXR: Failed to complete image database creation [" + openxr_api->get_error_string(completion.futureResult) + "]");
	}

	TrackingImageDatabase *image_database = image_database_owner.get_or_null(p_image_database);
	ERR_FAIL_NULL(image_database);

	// Remember our database
	image_database->database = completion.database;

	// Set our RID as our result value on our future.
	p_future_result->set_result_value(p_image_database);

	// And perform our callback if we have one.
	if (p_user_callback.is_valid()) {
		p_user_callback.call(p_image_database);
	}
}

bool OpenXRSpatialImageTrackingCapability::start_built_in_tracking(RID p_image_database) {
	ERR_FAIL_COND_V(builtin_tracking_state > 0, false);

	RID image_database = p_image_database;

	// See if we need to create our default image database
	if (image_database.is_null()) {
		ERR_FAIL_V_MSG(false, "not yet supported!");

		// Create a new image database
		image_database = image_database_create();

		// Load default image database from project settings
		//for (....) {
		//	image_database_add(image_datbase, ...);
		//}

		// If we created it, remember it, we want to free it later.
		image_db = image_database;
	};

	builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_SETTING_UP;

	if (image_database_get_handle(image_database) == XR_NULL_HANDLE) {
		// Need to submit this and start session on success (note, using the same built_in_future here as for sessions is fine, as these actions are sequential).
		built_in_future = image_database_submit(image_database, callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_image_database_created), callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_image_database_creation_failed));
	} else {
		// Start creating session
		built_in_future = _create_spatial_context(image_database);
	}

	if (built_in_future.is_null()) {
		builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_SETUP_FAILED;
		return false;
	}

	return true;
}

void OpenXRSpatialImageTrackingCapability::stop_built_in_tracking(bool p_clear_trackers) {
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

	// If we have a spatial context, clean it up.
	if (spatial_context.is_valid()) {
		OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
		ERR_FAIL_NULL(se_extension);

		if (p_clear_trackers && image_trackers.has(spatial_context)) {
			XRServer *xr_server = XRServer::get_singleton();
			ERR_FAIL_NULL(xr_server);

			// Free and unregister our image trackers
			HashMap<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>> &images = image_trackers[spatial_context];
			for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>> &image : images) {
				xr_server->remove_tracker(image.value);
			}
			image_trackers.erase(spatial_context);
		}

		se_extension->free_spatial_context(spatial_context);
		spatial_context = RID();
	}

	if (image_db.is_valid()) {
		image_database_free(image_db);
		image_db = RID();
	}
}

Ref<OpenXRFutureResult> OpenXRSpatialImageTrackingCapability::start_entity_discovery(RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_create, Ref<OpenXRStructureBase> p_next_snapshot_query, const Callable &p_user_callback) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, nullptr);
	return se_extension->discover_spatial_entities_with_component_data(p_spatial_context, p_component_data, p_next_snapshot_create, callable_mp(this, &OpenXRSpatialImageTrackingCapability::_process_snapshot).bind(p_spatial_context, p_component_data, p_next_snapshot_query, p_user_callback));
}

void OpenXRSpatialImageTrackingCapability::_on_image_database_created(RID p_image_database) {
	// Now create our spatial context...
	built_in_future = _create_spatial_context(p_image_database);
	if (built_in_future.is_null()) {
		builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_SETUP_FAILED;
	}
}

void OpenXRSpatialImageTrackingCapability::_on_image_database_creation_failed(RID p_image_database, int p_xr_result, bool p_is_completion_failure) {
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

	// If this was our database, free it up!
	if (image_db == p_image_database && image_db.is_valid()) {
		image_database_free(image_db);
		image_db = RID();
	}
}

Ref<OpenXRFutureResult> OpenXRSpatialImageTrackingCapability::_create_spatial_context(RID p_image_database) {
	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL_V(se_extension, Ref<OpenXRFutureResult>());

	TypedArray<OpenXRSpatialCapabilityConfigurationBaseHeader> capability_configurations;

	image_tracking_configuration.instantiate();
	capability_configurations.push_back(image_tracking_configuration);

	return se_extension->create_spatial_context(capability_configurations, nullptr, callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_spatial_context_created), callable_mp(this, &OpenXRSpatialImageTrackingCapability::_on_spatial_context_creation_failed));
}

void OpenXRSpatialImageTrackingCapability::_on_spatial_context_created(RID p_spatial_context) {
	builtin_tracking_state = OpenXRSpatialEntityExtension::TrackingState::TRACKING_ENABLED;
	spatial_context = p_spatial_context;
	need_discovery = true;

	// We don't need our future anymore.
	built_in_future.unref();
}

void OpenXRSpatialImageTrackingCapability::_on_spatial_context_creation_failed(int p_xr_result, bool p_is_completion_failure) {
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

	// If we created our image db, then free it.
	if (image_db.is_valid()) {
		image_database_free(image_db);
		image_db = RID();
	}
}

void OpenXRSpatialImageTrackingCapability::_on_spatial_discovery_recommended(RID p_spatial_context) {
	if (p_spatial_context == spatial_context) {
		// Trigger new discovery.
		need_discovery = true;
	}
}

void OpenXRSpatialImageTrackingCapability::_process_snapshot(RID p_snapshot, RID p_spatial_context, TypedArray<OpenXRSpatialComponentData> p_component_data, Ref<OpenXRStructureBase> p_next_snapshot_query, const Callable &p_user_callback) {
	print_line("OpenXR: processing image tracking snapshot"); // TESTING REMOVE!

	if (p_user_callback.is_valid()) {
		p_user_callback.call(p_snapshot, false);
	}

	OpenXRSpatialEntityExtension *se_extension = OpenXRSpatialEntityExtension::get_singleton();
	ERR_FAIL_NULL(se_extension);
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Make a copy of the images we have right now, so we know which ones to clean up.
	LocalVector<XrSpatialEntityIdEXT> current_images;
	HashMap<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>> &images = image_trackers[p_spatial_context];
	current_images.resize(images.size());
	int p = 0;
	for (const KeyValue<XrSpatialEntityIdEXT, Ref<OpenXRImageTracker>> &image : images) {
		current_images[p++] = image.key;
	}

	// The first must be OpenXRSpatialQueryResultData
	Ref<OpenXRSpatialQueryResultData> query_result_data = p_component_data.is_empty() ? Variant() : p_component_data[0];
	ERR_FAIL_COND(query_result_data.is_null());

	Ref<OpenXRSpatialComponentBounded2DList> bounded2d_list;
	Ref<OpenXRSpatialComponentImage2DList> image2d_list;
	for (Ref<OpenXRSpatialComponentData> data : p_component_data) {
		switch (data->get_component_type()) {
			case XR_SPATIAL_COMPONENT_TYPE_BOUNDED_2D_EXT:
				bounded2d_list = data;
				break;
			case XR_SPATIAL_COMPONENT_TYPE_IMAGE_2D_EXT:
				image2d_list = data;
				break;
			default:
				// Okay, maybe other data types are being queried that we don't know about
				break;
		}
	}

	if (se_extension->query_snapshot(p_snapshot, p_component_data, p_next_snapshot_query)) {
		// Now loop through our data and update our images.
		// Q we're assuming entity ID, size and state size are equal, is there ever a situation where they would not be?
		int64_t size = query_result_data->get_capacity();
		for (int64_t i = 0; i < size; i++) {
			XrSpatialEntityIdEXT entity_id = query_result_data->get_entity_id(i);
			XrSpatialEntityTrackingStateEXT entity_state = query_result_data->get_entity_state(i);

			// Erase it from our current images (if we have it, else this is ignored).
			current_images.erase(entity_id);

			if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT) {
				// We should only get this status on updates as a prelude to needing to remove this image.
				// So we just update the status.
				if (images.has(entity_id)) {
					Ref<OpenXRImageTracker> image_tracker = images[entity_id];
					image_tracker->invalidate_pose(SNAME("default"));
					image_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT);
				}
			} else {
				// Process our entity
				bool add_to_xr_server = false;
				Ref<OpenXRImageTracker> image_tracker;

				if (images.has(entity_id)) {
					// We know about this one already
					image_tracker = images[entity_id];
				} else {
					// Create a new anchor
					image_tracker.instantiate();
					image_tracker->set_spatial_context(p_spatial_context);
					image_tracker->set_entity(se_extension->make_spatial_entity(se_extension->get_spatial_snapshot_context(p_snapshot), entity_id));
					images[entity_id] = image_tracker;

					add_to_xr_server = true;
				}

				// Handle component data
				if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT) {
					image_tracker->invalidate_pose(SNAME("default"));
					image_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_PAUSED_EXT);

					// No further component data will be valid in this state, we need to ignore it!
				} else if (entity_state == XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT) {
					if (bounded2d_list.is_valid()) {
						Transform3D transform = bounded2d_list->get_center_pose(i);
						image_tracker->set_pose(SNAME("default"), transform, Vector3(), Vector3());
						image_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_TRACKING_EXT);
						image_tracker->set_bounds_size(bounded2d_list->get_size(i));
					}

					if (image2d_list.is_valid()) {
						// Process our component data.
						XrSpatialImageTrackingDatabaseEXT image_database = image2d_list->get_image_tracking_database(i);
						uint32_t index = image2d_list->get_reference_image_index(i);

						image_tracker->set_image_info(image_database, index);
					}
				}

				if (add_to_xr_server) {
					// Register with XR server
					xr_server->add_tracker(image_tracker);
				}
			}
		}

		// Remove any images that are no longer there...
		for (const XrSpatialEntityIdEXT &entity_id : current_images) {
			if (images.has(entity_id)) {
				Ref<OpenXRImageTracker> image_tracker = images[entity_id];

				// Just in case there are still references out there to this image,
				// reset some stuff.
				image_tracker->invalidate_pose(SNAME("default"));
				image_tracker->set_spatial_tracking_state(XR_SPATIAL_ENTITY_TRACKING_STATE_STOPPED_EXT);

				// Remove it from our XRServer
				xr_server->remove_tracker(image_tracker);

				// Remove it from our trackers
				images.erase(entity_id);
			}
		}
	}

	if (p_user_callback.is_valid()) {
		p_user_callback.call(p_snapshot, true);
	}

	// Now that we're done, clean up our snapshot!
	se_extension->free_spatial_snapshot(p_snapshot);

	// And if this was our discovery snapshot, lets reset it
	if (discovery_query_result.is_valid() && discovery_query_result->get_result_value() == p_snapshot) {
		discovery_query_result.unref();
	}

	// TESTING! REMOVE
	need_discovery = true;
}

String OpenXRSpatialImageTrackingCapability::get_reference_image_format_name(XrSpatialReferenceImageFormatEXT p_image_format) {
	XR_ENUM_SWITCH(XrSpatialReferenceImageFormatEXT, p_image_format)
}
