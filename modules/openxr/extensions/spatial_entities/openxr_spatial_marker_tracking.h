/**************************************************************************/
/*  openxr_spatial_marker_tracking.h                                      */
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

// QrCode marker tracking capability configuration
class OpenXRSpatialCapabilityConfigurationQrCode : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationQrCode, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return enabled_components; }

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialComponentTypeEXT> enabled_components;
	XrSpatialCapabilityConfigurationQrCodeEXT marker_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_QR_CODE_EXT, nullptr, XR_SPATIAL_CAPABILITY_MARKER_TRACKING_QR_CODE_EXT, 0, nullptr };

	PackedInt64Array _get_enabled_components() const;
};

// Micro QrCode marker tracking capability configuration
class OpenXRSpatialCapabilityConfigurationMicroQrCode : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationMicroQrCode, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return enabled_components; }

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialComponentTypeEXT> enabled_components;
	XrSpatialCapabilityConfigurationMicroQrCodeEXT marker_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_MICRO_QR_CODE_EXT, nullptr, XR_SPATIAL_CAPABILITY_MARKER_TRACKING_MICRO_QR_CODE_EXT, 0, nullptr };

	PackedInt64Array _get_enabled_components() const;
};

// Aruco marker tracking capability configuration
class OpenXRSpatialCapabilityConfigurationAruco : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationAruco, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	enum ArucoDict {
		ARUCO_DICT_4X4_50 = XR_SPATIAL_MARKER_ARUCO_DICT_4X4_50_EXT,
		ARUCO_DICT_4X4_100 = XR_SPATIAL_MARKER_ARUCO_DICT_4X4_100_EXT,
		ARUCO_DICT_4X4_250 = XR_SPATIAL_MARKER_ARUCO_DICT_4X4_250_EXT,
		ARUCO_DICT_4X4_1000 = XR_SPATIAL_MARKER_ARUCO_DICT_4X4_1000_EXT,
		ARUCO_DICT_5X5_50 = XR_SPATIAL_MARKER_ARUCO_DICT_5X5_50_EXT,
		ARUCO_DICT_5X5_100 = XR_SPATIAL_MARKER_ARUCO_DICT_5X5_100_EXT,
		ARUCO_DICT_5X5_250 = XR_SPATIAL_MARKER_ARUCO_DICT_5X5_250_EXT,
		ARUCO_DICT_5X5_1000 = XR_SPATIAL_MARKER_ARUCO_DICT_5X5_1000_EXT,
		ARUCO_DICT_6X6_50 = XR_SPATIAL_MARKER_ARUCO_DICT_6X6_50_EXT,
		ARUCO_DICT_6X6_100 = XR_SPATIAL_MARKER_ARUCO_DICT_6X6_100_EXT,
		ARUCO_DICT_6X6_250 = XR_SPATIAL_MARKER_ARUCO_DICT_6X6_250_EXT,
		ARUCO_DICT_6X6_1000 = XR_SPATIAL_MARKER_ARUCO_DICT_6X6_1000_EXT,
		ARUCO_DICT_7X7_50 = XR_SPATIAL_MARKER_ARUCO_DICT_7X7_50_EXT,
		ARUCO_DICT_7X7_100 = XR_SPATIAL_MARKER_ARUCO_DICT_7X7_100_EXT,
		ARUCO_DICT_7X7_250 = XR_SPATIAL_MARKER_ARUCO_DICT_7X7_250_EXT,
		ARUCO_DICT_7X7_1000 = XR_SPATIAL_MARKER_ARUCO_DICT_7X7_1000_EXT,
	};

	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	void set_aruco_dict(XrSpatialMarkerArucoDictEXT p_dict);
	XrSpatialMarkerArucoDictEXT get_aruco_dict() const;

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return enabled_components; }

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialComponentTypeEXT> enabled_components;
	XrSpatialCapabilityConfigurationArucoMarkerEXT marker_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_ARUCO_MARKER_EXT, nullptr, XR_SPATIAL_CAPABILITY_MARKER_TRACKING_ARUCO_MARKER_EXT, 0, nullptr, XR_SPATIAL_MARKER_ARUCO_DICT_7X7_1000_EXT };

	PackedInt64Array _get_enabled_components() const;

	void _set_aruco_dict(ArucoDict p_dict);
	ArucoDict _get_aruco_dict() const;
};

VARIANT_ENUM_CAST(OpenXRSpatialCapabilityConfigurationAruco::ArucoDict);

// April tag marker tracking capability configuration
class OpenXRSpatialCapabilityConfigurationAprilTag : public OpenXRSpatialCapabilityConfigurationBaseHeader {
	GDCLASS(OpenXRSpatialCapabilityConfigurationAprilTag, OpenXRSpatialCapabilityConfigurationBaseHeader);

public:
	enum AprilTagDict {
		APRIL_TAG_DICT_16H5 = XR_SPATIAL_MARKER_APRIL_TAG_DICT_16H5_EXT,
		APRIL_TAG_DICT_25H9 = XR_SPATIAL_MARKER_APRIL_TAG_DICT_25H9_EXT,
		APRIL_TAG_DICT_36H10 = XR_SPATIAL_MARKER_APRIL_TAG_DICT_36H10_EXT,
		APRIL_TAG_DICT_36H11 = XR_SPATIAL_MARKER_APRIL_TAG_DICT_36H11_EXT,
	};

	virtual bool has_valid_configuration() const override;
	virtual XrSpatialCapabilityConfigurationBaseHeaderEXT *get_configuration() override;

	void set_april_dict(XrSpatialMarkerAprilTagDictEXT p_dict);
	XrSpatialMarkerAprilTagDictEXT get_april_dict() const;

	Vector<XrSpatialComponentTypeEXT> get_enabled_components() const { return enabled_components; }

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialComponentTypeEXT> enabled_components;
	XrSpatialCapabilityConfigurationAprilTagEXT marker_config = { XR_TYPE_SPATIAL_CAPABILITY_CONFIGURATION_APRIL_TAG_EXT, nullptr, XR_SPATIAL_CAPABILITY_MARKER_TRACKING_APRIL_TAG_EXT, 0, nullptr, XR_SPATIAL_MARKER_APRIL_TAG_DICT_36H11_EXT };

	PackedInt64Array _get_enabled_components() const;

	void _set_april_dict(AprilTagDict p_dict);
	AprilTagDict _get_april_dict() const;
};

VARIANT_ENUM_CAST(OpenXRSpatialCapabilityConfigurationAprilTag::AprilTagDict);

// Marker component data
class OpenXRSpatialComponentMarkerList : public OpenXRSpatialComponentData {
	GDCLASS(OpenXRSpatialComponentMarkerList, OpenXRSpatialComponentData);

public:
	enum MarkerType {
		MARKER_TYPE_UNKNOWN,
		MARKER_TYPE_QRCODE,
		MARKER_TYPE_MICRO_QRCODE,
		MARKER_TYPE_ARUCO,
		MARKER_TYPE_APRIL_TAG,
		MARKER_TYPE_MAX
	};

	virtual void set_capacity(uint32_t p_capacity) override;
	virtual XrSpatialComponentTypeEXT get_component_type() const override;
	virtual void *get_structure_data(void *p_next) override;

	MarkerType get_marker_type(int64_t p_index) const;
	uint32_t get_marker_id(int64_t p_index) const;
	Variant get_marker_data(RID p_snapshot, int64_t p_index) const;

protected:
	static void _bind_methods();

private:
	Vector<XrSpatialMarkerDataEXT> marker_data;

	XrSpatialComponentMarkerListEXT marker_list = { XR_TYPE_SPATIAL_COMPONENT_MARKER_LIST_EXT, nullptr, 0, nullptr };
};

VARIANT_ENUM_CAST(OpenXRSpatialComponentMarkerList::MarkerType);

// Marker tracker
class OpenXRMarkerTracker : public OpenXRSpatialEntityTracker {
	GDCLASS(OpenXRMarkerTracker, OpenXRSpatialEntityTracker);

public:
	void set_bounds_size(const Vector2 &p_bounds_size);
	Vector2 get_bounds_size() const;

	void set_marker_type(OpenXRSpatialComponentMarkerList::MarkerType p_marker_type);
	OpenXRSpatialComponentMarkerList::MarkerType get_marker_type() const;

	void set_marker_id(uint32_t p_id);
	uint32_t get_marker_id() const;

	void set_marker_data(const Variant &p_data);
	Variant get_marker_data() const;

protected:
	static void _bind_methods();

private:
	Vector2 bounds_size;

	OpenXRSpatialComponentMarkerList::MarkerType marker_type = OpenXRSpatialComponentMarkerList::MarkerType::MARKER_TYPE_UNKNOWN;
	uint32_t marker_id = 0;
	Variant marker_data;
};

// Marker tracking logic
class OpenXRSpatialMarkerTrackingCapability : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRSpatialMarkerTrackingCapability, OpenXRExtensionWrapper);

protected:
	static void _bind_methods();

public:
	static OpenXRSpatialMarkerTrackingCapability *get_singleton();

	OpenXRSpatialMarkerTrackingCapability();
	virtual ~OpenXRSpatialMarkerTrackingCapability() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;

	virtual void on_process() override;

	bool is_qrcode_supported();
	bool is_micro_qrcode_supported();
	bool is_aruco_supported();
	bool is_april_tag_supported();

private:
	static OpenXRSpatialMarkerTrackingCapability *singleton;
	bool spatial_marker_tracking_ext = false;

	RID spatial_context;
	bool need_discovery = false;
	int discovery_cooldown = 0;
	Ref<OpenXRFutureResult> discovery_query_result;

	Ref<OpenXRSpatialCapabilityConfigurationQrCode> qrcode_configuration;
	Ref<OpenXRSpatialCapabilityConfigurationMicroQrCode> micro_qrcode_configuration;
	Ref<OpenXRSpatialCapabilityConfigurationAruco> aruco_configuration;
	Ref<OpenXRSpatialCapabilityConfigurationAprilTag> april_tag_configuration;

	// Discovery logic
	Ref<OpenXRFutureResult> _create_spatial_context();
	void _on_spatial_context_created(RID p_spatial_context);

	void _on_spatial_discovery_recommended(RID p_spatial_context);

	Ref<OpenXRFutureResult> _start_entity_discovery();
	void _process_snapshot(RID p_snapshot, bool p_is_discovery);

	// Trackers
	HashMap<XrSpatialEntityIdEXT, Ref<OpenXRMarkerTracker>> marker_trackers;
};
