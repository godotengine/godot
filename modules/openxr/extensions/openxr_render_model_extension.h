/**************************************************************************/
/*  openxr_render_model_extension.h                                       */
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

#include "modules/modules_enabled.gen.h"

#ifdef MODULE_GLTF_ENABLED
#include "../openxr_uuid.h"
#include "../util.h"
#include "core/templates/rid_owner.h"
#include "modules/gltf/gltf_document.h"
#include "openxr_extension_wrapper.h"
#include "scene/3d/node_3d.h"
#include "servers/xr/xr_pose.h"

#include <openxr/openxr.h>

class OpenXRRenderModelData : public RefCounted {
	GDCLASS(OpenXRRenderModelData, RefCounted);

private:
	Ref<GLTFDocument> gltf_document;
	Ref<GLTFState> gltf_state;
	PackedStringArray node_names;

public:
	Ref<GLTFState> get_gltf_state() { return gltf_state; }

	bool parse_gltf_document(const PackedByteArray &p_bytes);
	Node3D *new_scene_instance();

	void set_node_names(const PackedStringArray &p_node_names);
	PackedStringArray get_node_names() const;
	const String get_node_name(uint32_t p_node_index) const;

	OpenXRRenderModelData();
	~OpenXRRenderModelData();
};

class OpenXRRenderModelExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRRenderModelExtension, OpenXRExtensionWrapper);

protected:
	static void _bind_methods();

public:
	static OpenXRRenderModelExtension *get_singleton();

	OpenXRRenderModelExtension();
	virtual ~OpenXRRenderModelExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_instance_destroyed() override;
	virtual void on_session_destroyed() override;

	virtual bool on_event_polled(const XrEventDataBuffer &event) override;
	virtual void on_sync_actions() override;

	bool is_active() const;

	// Render model.
	bool has_render_model(RID p_render_model) const;
	RID render_model_create(XrRenderModelIdEXT p_render_model_id);
	void render_model_destroy(RID p_render_model);

	TypedArray<RID> render_model_get_all();
	Node3D *render_model_new_scene_instance(RID p_render_model) const;
	PackedStringArray render_model_get_subaction_paths(RID p_render_model);
	XrPath render_model_get_top_level_path(RID p_render_model) const;
	String render_model_get_top_level_path_as_string(RID p_render_model) const;
	XRPose::TrackingConfidence render_model_get_confidence(RID p_render_model) const;
	Transform3D render_model_get_root_transform(RID p_render_model) const;
	uint32_t render_model_get_animatable_node_count(RID p_render_model) const;
	String render_model_get_animatable_node_name(RID p_render_model, uint32_t p_index) const;
	bool render_model_is_animatable_node_visible(RID p_render_model, uint32_t p_index) const;
	Transform3D render_model_get_animatable_node_transform(RID p_render_model, uint32_t p_index) const;

private:
	static OpenXRRenderModelExtension *singleton;

	// Related extensions.
	bool uuid_ext = false;
	bool render_model_ext = false;
	bool interaction_render_model_ext = false;

	// XrSync status
	bool xr_sync_has_run = false;

	// Interaction data.
	bool _interaction_data_dirty = true;
	HashMap<XrRenderModelIdEXT, RID> interaction_render_models;

	void _clear_interaction_data();
	bool _update_interaction_data();

	// Render model.
	Vector<XrPath> toplevel_paths;

	struct RenderModel {
		XrRenderModelIdEXT xr_render_model_id = XR_NULL_RENDER_MODEL_ID_EXT;
		XrRenderModelEXT xr_render_model = XR_NULL_HANDLE;
		uint32_t animatable_node_count = 0;
		Ref<OpenXRRenderModelData> render_model_data;
		XrSpace xr_space = XR_NULL_HANDLE;
		XRPose::TrackingConfidence confidence = XRPose::TrackingConfidence::XR_TRACKING_CONFIDENCE_NONE;
		Transform3D root_transform;
		LocalVector<XrRenderModelNodeStateEXT> node_states;
		XrPath top_level_path = XR_NULL_PATH;
	};

	mutable RID_Owner<RenderModel, true> render_model_owner;

	// GLTF asset cache
	HashMap<XrUuidEXT, Ref<OpenXRRenderModelData>, HashMapHasherXrUuidEXT> render_model_data_cache;

	Ref<OpenXRRenderModelData> _get_render_model_data(XrUuidEXT p_cache_id, uint32_t p_animatable_node_count);
	Ref<OpenXRRenderModelData> _load_asset(XrRenderModelAssetEXT p_asset, uint32_t p_animatable_node_count);
	void _clear_render_model_data();

	// GDScript/GDExtension passthroughs
	RID _render_model_create(uint64_t p_render_model_id);

	// OpenXR API call wrappers
	EXT_PROTO_XRRESULT_FUNC3(xrCreateRenderModelEXT, (XrSession), session, (const XrRenderModelCreateInfoEXT *), createInfo, (XrRenderModelEXT *), renderModel);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroyRenderModelEXT, (XrRenderModelEXT), renderModel);
	EXT_PROTO_XRRESULT_FUNC3(xrGetRenderModelPropertiesEXT, (XrRenderModelEXT), renderModel, (const XrRenderModelPropertiesGetInfoEXT *), getInfo, (XrRenderModelPropertiesEXT *), properties);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateRenderModelSpaceEXT, (XrSession), session, (const XrRenderModelSpaceCreateInfoEXT *), createInfo, (XrSpace *), space);
	EXT_PROTO_XRRESULT_FUNC3(xrCreateRenderModelAssetEXT, (XrSession), session, (const XrRenderModelAssetCreateInfoEXT *), createInfo, (XrRenderModelAssetEXT *), asset);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroyRenderModelAssetEXT, (XrRenderModelAssetEXT), asset);
	EXT_PROTO_XRRESULT_FUNC3(xrGetRenderModelAssetDataEXT, (XrRenderModelAssetEXT), asset, (const XrRenderModelAssetDataGetInfoEXT *), getInfo, (XrRenderModelAssetDataEXT *), buffer);
	EXT_PROTO_XRRESULT_FUNC3(xrGetRenderModelAssetPropertiesEXT, (XrRenderModelAssetEXT), asset, (const XrRenderModelAssetPropertiesGetInfoEXT *), getInfo, (XrRenderModelAssetPropertiesEXT *), properties);
	EXT_PROTO_XRRESULT_FUNC3(xrGetRenderModelStateEXT, (XrRenderModelEXT), renderModel, (const XrRenderModelStateGetInfoEXT *), getInfo, (XrRenderModelStateEXT *), state);
	EXT_PROTO_XRRESULT_FUNC5(xrEnumerateInteractionRenderModelIdsEXT, (XrSession), session, (const XrInteractionRenderModelIdsEnumerateInfoEXT *), getInfo, (uint32_t), renderModelIdCapacityInput, (uint32_t *), renderModelIdCountOutput, (XrRenderModelIdEXT *), renderModelIds);
	EXT_PROTO_XRRESULT_FUNC5(xrEnumerateRenderModelSubactionPathsEXT, (XrRenderModelEXT), renderModel, (const XrInteractionRenderModelSubactionPathInfoEXT *), info, (uint32_t), pathCapacityInput, (uint32_t *), pathCountOutput, (XrPath *), paths);
	EXT_PROTO_XRRESULT_FUNC3(xrGetRenderModelPoseTopLevelUserPathEXT, (XrRenderModelEXT), renderModel, (const XrInteractionRenderModelTopLevelUserPathGetInfoEXT *), info, (XrPath *), topLevelUserPath);

	EXT_PROTO_XRRESULT_FUNC4(xrLocateSpace, (XrSpace), space, (XrSpace), baseSpace, (XrTime), time, (XrSpaceLocation *), location);
	EXT_PROTO_XRRESULT_FUNC1(xrDestroySpace, (XrSpace), space);
	EXT_PROTO_XRRESULT_FUNC5(xrPathToString, (XrInstance), instance, (XrPath), path, (uint32_t), bufferCapacityInput, (uint32_t *), bufferCountOutput, (char *), buffer);
};
#endif // MODULE_GLTF_ENABLED
