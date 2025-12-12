/**************************************************************************/
/*  openxr_render_model_extension.cpp                                     */
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

#include "openxr_render_model_extension.h"

#ifdef MODULE_GLTF_ENABLED
#include "../openxr_api.h"
#include "../openxr_interface.h"

#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "servers/xr/xr_server.h"

OpenXRRenderModelExtension *OpenXRRenderModelExtension::singleton = nullptr;

OpenXRRenderModelExtension *OpenXRRenderModelExtension::get_singleton() {
	return singleton;
}

void OpenXRRenderModelExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_active"), &OpenXRRenderModelExtension::is_active);
	ClassDB::bind_method(D_METHOD("render_model_create", "render_model_id"), &OpenXRRenderModelExtension::render_model_create);
	ClassDB::bind_method(D_METHOD("render_model_destroy", "render_model"), &OpenXRRenderModelExtension::render_model_destroy);
	ClassDB::bind_method(D_METHOD("render_model_get_all"), &OpenXRRenderModelExtension::render_model_get_all);
	ClassDB::bind_method(D_METHOD("render_model_new_scene_instance", "render_model"), &OpenXRRenderModelExtension::render_model_new_scene_instance);
	ClassDB::bind_method(D_METHOD("render_model_get_subaction_paths", "render_model"), &OpenXRRenderModelExtension::render_model_get_subaction_paths);
	ClassDB::bind_method(D_METHOD("render_model_get_top_level_path", "render_model"), &OpenXRRenderModelExtension::render_model_get_top_level_path_as_string);
	ClassDB::bind_method(D_METHOD("render_model_get_confidence", "render_model"), &OpenXRRenderModelExtension::render_model_get_confidence);
	ClassDB::bind_method(D_METHOD("render_model_get_root_transform", "render_model"), &OpenXRRenderModelExtension::render_model_get_root_transform);
	ClassDB::bind_method(D_METHOD("render_model_get_animatable_node_count", "render_model"), &OpenXRRenderModelExtension::render_model_get_animatable_node_count);
	ClassDB::bind_method(D_METHOD("render_model_get_animatable_node_name", "render_model", "index"), &OpenXRRenderModelExtension::render_model_get_animatable_node_name);
	ClassDB::bind_method(D_METHOD("render_model_is_animatable_node_visible", "render_model", "index"), &OpenXRRenderModelExtension::render_model_is_animatable_node_visible);
	ClassDB::bind_method(D_METHOD("render_model_get_animatable_node_transform", "render_model", "index"), &OpenXRRenderModelExtension::render_model_get_animatable_node_transform);

	ADD_SIGNAL(MethodInfo("render_model_added", PropertyInfo(Variant::RID, "render_model")));
	ADD_SIGNAL(MethodInfo("render_model_removed", PropertyInfo(Variant::RID, "render_model")));
	ADD_SIGNAL(MethodInfo("render_model_top_level_path_changed", PropertyInfo(Variant::RID, "render_model")));
}

OpenXRRenderModelExtension::OpenXRRenderModelExtension() {
	singleton = this;
}

OpenXRRenderModelExtension::~OpenXRRenderModelExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRRenderModelExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	if (GLOBAL_GET("xr/openxr/extensions/render_model")) {
		if (p_version < XR_API_VERSION_1_1_0) {
			// Extension was promoted in OpenXR 1.1, only include it in OpenXR 1.0.
			request_extensions[XR_EXT_UUID_EXTENSION_NAME] = &uuid_ext;
		}
		request_extensions[XR_EXT_RENDER_MODEL_EXTENSION_NAME] = &render_model_ext;
		request_extensions[XR_EXT_INTERACTION_RENDER_MODEL_EXTENSION_NAME] = &interaction_render_model_ext;
	}

	return request_extensions;
}

void OpenXRRenderModelExtension::on_instance_created(const XrInstance p_instance) {
	// Standard entry points we use.
	EXT_INIT_XR_FUNC(xrLocateSpace);
	EXT_INIT_XR_FUNC(xrDestroySpace);
	EXT_INIT_XR_FUNC(xrPathToString);

	if (render_model_ext) {
		EXT_INIT_XR_FUNC(xrCreateRenderModelEXT);
		EXT_INIT_XR_FUNC(xrDestroyRenderModelEXT);
		EXT_INIT_XR_FUNC(xrGetRenderModelPropertiesEXT);
		EXT_INIT_XR_FUNC(xrCreateRenderModelSpaceEXT);
		EXT_INIT_XR_FUNC(xrCreateRenderModelAssetEXT);
		EXT_INIT_XR_FUNC(xrDestroyRenderModelAssetEXT);
		EXT_INIT_XR_FUNC(xrGetRenderModelAssetDataEXT);
		EXT_INIT_XR_FUNC(xrGetRenderModelAssetPropertiesEXT);
		EXT_INIT_XR_FUNC(xrGetRenderModelStateEXT);
	}

	if (interaction_render_model_ext) {
		EXT_INIT_XR_FUNC(xrEnumerateInteractionRenderModelIdsEXT);
		EXT_INIT_XR_FUNC(xrEnumerateRenderModelSubactionPathsEXT);
		EXT_INIT_XR_FUNC(xrGetRenderModelPoseTopLevelUserPathEXT);
	}
}

void OpenXRRenderModelExtension::on_session_created(const XrSession p_session) {
	_interaction_data_dirty = true;
}

void OpenXRRenderModelExtension::on_instance_destroyed() {
	xrCreateRenderModelEXT_ptr = nullptr;
	xrDestroyRenderModelEXT_ptr = nullptr;
	xrGetRenderModelPropertiesEXT_ptr = nullptr;
	xrCreateRenderModelSpaceEXT_ptr = nullptr;
	xrCreateRenderModelAssetEXT_ptr = nullptr;
	xrDestroyRenderModelAssetEXT_ptr = nullptr;
	xrGetRenderModelAssetDataEXT_ptr = nullptr;
	xrGetRenderModelAssetPropertiesEXT_ptr = nullptr;
	xrGetRenderModelStateEXT_ptr = nullptr;
	xrEnumerateInteractionRenderModelIdsEXT_ptr = nullptr;
	xrEnumerateRenderModelSubactionPathsEXT_ptr = nullptr;
	xrGetRenderModelPoseTopLevelUserPathEXT_ptr = nullptr;

	uuid_ext = false;
	render_model_ext = false;
	interaction_render_model_ext = false;
}

void OpenXRRenderModelExtension::on_session_destroyed() {
	_clear_interaction_data();
	_clear_render_model_data();

	// We no longer have valid sync data.
	xr_sync_has_run = false;
}

bool OpenXRRenderModelExtension::on_event_polled(const XrEventDataBuffer &event) {
	if (event.type == XR_TYPE_EVENT_DATA_INTERACTION_RENDER_MODELS_CHANGED_EXT) {
		// Mark interaction data as dirty so that we update it on sync.
		_interaction_data_dirty = true;

		return true;
	} else if (event.type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) {
		// If our controller bindings changed, its likely our render models change too.
		// We should be getting a XR_TYPE_EVENT_DATA_INTERACTION_RENDER_MODELS_CHANGED_EXT
		// but checking for this scenario just in case.
		_interaction_data_dirty = true;

		// Do not consider this handled, we simply do additional logic.
		return false;
	}

	return false;
}

void OpenXRRenderModelExtension::on_sync_actions() {
	if (!is_active()) {
		return;
	}

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL(openxr_api);

	// Mark sync as run
	xr_sync_has_run = true;

	// Update our interaction data if needed
	if (_interaction_data_dirty) {
		_update_interaction_data();
	}

	// Loop through all of our render models to update our space and state info
	LocalVector<RID> owned = render_model_owner.get_owned_list();

	for (const RID &rid : owned) {
		RenderModel *render_model = render_model_owner.get_or_null(rid);
		if (render_model && render_model->xr_space != XR_NULL_HANDLE) {
			XrSpaceLocation render_model_location = {
				XR_TYPE_SPACE_LOCATION, // type
				nullptr, // next
				0, // locationFlags
				{ { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 0.0 } }, // pose
			};

			XrResult result = xrLocateSpace(render_model->xr_space, openxr_api->get_play_space(), openxr_api->get_predicted_display_time(), &render_model_location);
			ERR_CONTINUE_MSG(XR_FAILED(result), "OpenXR: Failed to locate render model space [" + openxr_api->get_error_string(result) + "]");

			render_model->confidence = openxr_api->transform_from_location(render_model_location, render_model->root_transform);

			if (!render_model->node_states.is_empty()) {
				// Get node states.
				XrRenderModelStateGetInfoEXT get_state_info = {
					XR_TYPE_RENDER_MODEL_STATE_GET_INFO_EXT, // type
					nullptr, // next
					openxr_api->get_predicted_display_time() // displayTime
				};

				XrRenderModelStateEXT state = {
					XR_TYPE_RENDER_MODEL_STATE_EXT, // type
					nullptr, // next
					render_model->animatable_node_count, // nodeStateCount
					render_model->node_states.ptr(), // nodeStates
				};

				result = xrGetRenderModelStateEXT(render_model->xr_render_model, &get_state_info, &state);
				if (XR_FAILED(result)) {
					ERR_PRINT("OpenXR: Failed to update node states [" + openxr_api->get_error_string(result) + "]");
				}
			}

			XrPath new_path = XR_NULL_PATH;

			if (toplevel_paths.is_empty()) {
				// Set this up just once with paths we support here.
				toplevel_paths.push_back(openxr_api->get_xr_path("/user/hand/left"));
				toplevel_paths.push_back(openxr_api->get_xr_path("/user/hand/right"));
			}

			XrInteractionRenderModelTopLevelUserPathGetInfoEXT info = {
				XR_TYPE_INTERACTION_RENDER_MODEL_TOP_LEVEL_USER_PATH_GET_INFO_EXT, // type
				nullptr, // next
				(uint32_t)toplevel_paths.size(), // topLevelUserPathCount
				toplevel_paths.ptr() // topLevelUserPaths
			};
			result = xrGetRenderModelPoseTopLevelUserPathEXT(render_model->xr_render_model, &info, &new_path);
			if (XR_FAILED(result)) {
				ERR_PRINT("OpenXR: Failed to update the top level path for render models [" + openxr_api->get_error_string(result) + "]");
			} else if (new_path != render_model->top_level_path) {
				print_verbose("OpenXR: Render model top level path changed to " + openxr_api->get_xr_path_name(new_path));

				// Set the new path
				render_model->top_level_path = new_path;

				// And broadcast it
				// Note, converting an XrPath to a String has overhead, so we won't do this automatically.
				emit_signal(SNAME("render_model_top_level_path_changed"), rid);
			}
		}
	}
}

bool OpenXRRenderModelExtension::is_active() const {
	return render_model_ext && interaction_render_model_ext;
}

void OpenXRRenderModelExtension::_clear_interaction_data() {
	for (const KeyValue<XrRenderModelIdEXT, RID> &e : interaction_render_models) {
		render_model_destroy(e.value);
	}
	interaction_render_models.clear();
}

bool OpenXRRenderModelExtension::_update_interaction_data() {
	ERR_FAIL_COND_V_MSG(!interaction_render_model_ext, false, "Interaction render model extension hasn't been enabled.");

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, false);

	XrSession session = openxr_api->get_session();
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, false);

	// Check if syncActions has been run at least once or there is no point in getting data.
	if (!xr_sync_has_run) {
		// Do not treat this as an error.
		return true;
	}

	// If we get this far, no longer mark as dirty.
	// Else we just repeat the same error over and over again.
	_interaction_data_dirty = false;

	// Obtain interaction info.
	XrInteractionRenderModelIdsEnumerateInfoEXT interaction_info = {
		XR_TYPE_INTERACTION_RENDER_MODEL_IDS_ENUMERATE_INFO_EXT, // type
		nullptr, // next
	};

	// Obtain count.
	uint32_t interaction_count = 0;
	XrResult result = xrEnumerateInteractionRenderModelIdsEXT(session, &interaction_info, 0, &interaction_count, nullptr);
	if (XR_FAILED(result)) {
		// not successful? then we do nothing.
		ERR_FAIL_V_MSG(false, "OpenXR: Failed to obtain render model interaction id count [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	// Create some storage
	LocalVector<XrRenderModelIdEXT> render_model_interaction_ids;
	render_model_interaction_ids.resize(interaction_count);

	// Only need to fetch data if there is something to fetch (/we've got storage).
	if (!render_model_interaction_ids.is_empty()) {
		// Obtain interaction ids
		result = xrEnumerateInteractionRenderModelIdsEXT(session, &interaction_info, render_model_interaction_ids.size(), &interaction_count, render_model_interaction_ids.ptr());
		if (XR_FAILED(result)) {
			ERR_FAIL_V_MSG(false, "OpenXR: Failed to obtain render model interaction ids [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
		}
	}

	// Remove render models that are no longer tracked
	LocalVector<XrRenderModelIdEXT> erase_ids;
	for (const KeyValue<XrRenderModelIdEXT, RID> &e : interaction_render_models) {
		if (!render_model_interaction_ids.has(e.key)) {
			if (e.value.is_valid()) {
				render_model_destroy(e.value);
			}

			erase_ids.push_back(e.key);
		}
	}

	// Remove these from our hashmap
	for (const XrRenderModelIdEXT &id : erase_ids) {
		interaction_render_models.erase(id);
	}

	// Now update our models
	for (const XrRenderModelIdEXT &id : render_model_interaction_ids) {
		if (!interaction_render_models.has(id)) {
			// Even if this fails we add it so we don't repeat trying to create it
			interaction_render_models[id] = render_model_create(id);
		}
	}

	return true;
}

bool OpenXRRenderModelExtension::has_render_model(RID p_render_model) const {
	return render_model_owner.owns(p_render_model);
}

RID OpenXRRenderModelExtension::render_model_create(XrRenderModelIdEXT p_render_model_id) {
	ERR_FAIL_COND_V_MSG(!render_model_ext, RID(), "Render model extension hasn't been enabled.");

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, RID());

	XrSession session = openxr_api->get_session();
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, RID());

	RenderModel render_model;
	render_model.xr_render_model_id = p_render_model_id;

	// Get a list of supported glTF extensions.
	const HashSet<String> supported_gltf_extensions_hash_set = GLTFDocument::get_supported_gltf_extensions_hashset();
	Vector<CharString> supported_gltf_extensions_char_string; // Just for temp storage of our c-strings.
	supported_gltf_extensions_char_string.resize(supported_gltf_extensions_hash_set.size());
	int64_t supported_gltf_extension_index = 0;
	for (const String &ext : supported_gltf_extensions_hash_set) {
		supported_gltf_extensions_char_string.set(supported_gltf_extension_index, ext.utf8());
		supported_gltf_extension_index++;
	}
	// Now we can convert them to the `const char *` format.
	Vector<const char *> supported_gltf_extensions;
	supported_gltf_extensions.resize(supported_gltf_extensions_char_string.size());
	for (int64_t i = 0; i < supported_gltf_extensions_char_string.size(); i++) {
		supported_gltf_extensions.write[i] = supported_gltf_extensions_char_string[i].get_data();
	}

	XrRenderModelCreateInfoEXT create_info = {
		XR_TYPE_RENDER_MODEL_CREATE_INFO_EXT, // type
		nullptr, // next
		p_render_model_id, // renderModelId
		uint32_t(supported_gltf_extensions.size()), // gltfExtensionCount
		supported_gltf_extensions.ptr(), // gltfExtensions
	};

	XrResult result = xrCreateRenderModelEXT(session, &create_info, &render_model.xr_render_model);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(RID(), "OpenXR: Failed to create render model [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	XrRenderModelPropertiesGetInfoEXT properties_info = {
		XR_TYPE_RENDER_MODEL_PROPERTIES_GET_INFO_EXT, // type
		nullptr, // next
	};

	XrRenderModelPropertiesEXT properties = {
		XR_TYPE_RENDER_MODEL_PROPERTIES_EXT, // type
		nullptr, // next
		{}, // cacheId
		0, // animatableNodeCount
	};

	result = xrGetRenderModelPropertiesEXT(render_model.xr_render_model, &properties_info, &properties);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to get render model properties [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	} else {
		render_model.animatable_node_count = properties.animatableNodeCount;
		render_model.render_model_data = _get_render_model_data(properties.cacheId, properties.animatableNodeCount);
	}

	// Create space for positioning our asset.
	XrRenderModelSpaceCreateInfoEXT space_create_info = {
		XR_TYPE_RENDER_MODEL_SPACE_CREATE_INFO_EXT, // type
		nullptr, // next
		render_model.xr_render_model // renderModel
	};

	result = xrCreateRenderModelSpaceEXT(session, &space_create_info, &render_model.xr_space);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to create render model space [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	if (render_model.animatable_node_count > 0) {
		render_model.node_states.resize(render_model.animatable_node_count);
	}

	RID new_rid = render_model_owner.make_rid(render_model);

	emit_signal(SNAME("render_model_added"), new_rid);

	return new_rid;
}

RID OpenXRRenderModelExtension::_render_model_create(uint64_t p_render_model_id) {
	RID ret;

	ERR_FAIL_COND_V(p_render_model_id == XR_NULL_RENDER_MODEL_ID_EXT, ret);

	if (is_active()) {
		ret = render_model_create(XrRenderModelIdEXT(p_render_model_id));
	}

	return ret;
}

void OpenXRRenderModelExtension::render_model_destroy(RID p_render_model) {
	ERR_FAIL_COND_MSG(!render_model_ext, "Render model extension hasn't been enabled.");

	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL(render_model);

	emit_signal(SNAME("render_model_removed"), p_render_model);

	// Clean up.
	if (render_model->xr_space != XR_NULL_HANDLE) {
		xrDestroySpace(render_model->xr_space);
	}

	render_model->node_states.clear();

	// And destroy our model.
	XrResult result = xrDestroyRenderModelEXT(render_model->xr_render_model);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to destroy render model [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	render_model_owner.free(p_render_model);
}

TypedArray<RID> OpenXRRenderModelExtension::render_model_get_all() {
	TypedArray<RID> ret;

	LocalVector<RID> rids = render_model_owner.get_owned_list();

	for (const RID &rid : rids) {
		ret.push_back(rid);
	}

	return ret;
}

Node3D *OpenXRRenderModelExtension::render_model_new_scene_instance(RID p_render_model) const {
	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, nullptr);

	if (render_model->render_model_data.is_null()) {
		// We never loaded it (don't spam errors here).
		return nullptr;
	}

	return render_model->render_model_data->new_scene_instance();
}

PackedStringArray OpenXRRenderModelExtension::render_model_get_subaction_paths(RID p_render_model) {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, PackedStringArray());

	XrInstance instance = openxr_api->get_instance();
	ERR_FAIL_COND_V(instance == XR_NULL_HANDLE, PackedStringArray());

	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, PackedStringArray());

	PackedStringArray subaction_paths;

	XrInteractionRenderModelSubactionPathInfoEXT subaction_info = {
		XR_TYPE_INTERACTION_RENDER_MODEL_SUBACTION_PATH_INFO_EXT, // type
		nullptr, // next
	};

	uint32_t capacity;

	XrResult result = xrEnumerateRenderModelSubactionPathsEXT(render_model->xr_render_model, &subaction_info, 0, &capacity, nullptr);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(PackedStringArray(), "OpenXR: Failed to obtain render model subaction path count [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	if (capacity > 0) {
		LocalVector<XrPath> paths;

		paths.resize(capacity);

		result = xrEnumerateRenderModelSubactionPathsEXT(render_model->xr_render_model, &subaction_info, capacity, &capacity, paths.ptr());
		if (XR_FAILED(result)) {
			ERR_FAIL_V_MSG(PackedStringArray(), "OpenXR: Failed to obtain render model subaction paths [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
		}

		for (uint32_t i = 0; i < capacity; i++) {
			char buffer[1024];
			uint32_t size = 0;
			xrPathToString(instance, paths[i], 1024, &size, buffer);
			if (size > 0) {
				subaction_paths.push_back(String(buffer));
			}
		}
	}

	return subaction_paths;
}

XrPath OpenXRRenderModelExtension::render_model_get_top_level_path(RID p_render_model) const {
	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, XRPose::TrackingConfidence::XR_TRACKING_CONFIDENCE_NONE);

	return render_model->top_level_path;
}

String OpenXRRenderModelExtension::render_model_get_top_level_path_as_string(RID p_render_model) const {
	String ret;

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, ret);

	if (is_active() && has_render_model(p_render_model)) {
		XrPath path = render_model_get_top_level_path(p_render_model);
		if (path == XR_NULL_PATH) {
			return "None";
		} else {
			return openxr_api->get_xr_path_name(path);
		}
	}

	return ret;
}

XRPose::TrackingConfidence OpenXRRenderModelExtension::render_model_get_confidence(RID p_render_model) const {
	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, XRPose::TrackingConfidence::XR_TRACKING_CONFIDENCE_NONE);

	return render_model->confidence;
}

Transform3D OpenXRRenderModelExtension::render_model_get_root_transform(RID p_render_model) const {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, Transform3D());

	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, Transform3D());

	// Scale our root transform
	real_t world_scale = xr_server->get_world_scale();
	Transform3D root_transform = render_model->root_transform.scaled(Vector3(world_scale, world_scale, world_scale));

	return xr_server->get_reference_frame() * root_transform;
}

uint32_t OpenXRRenderModelExtension::render_model_get_animatable_node_count(RID p_render_model) const {
	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, 0);

	return render_model->animatable_node_count;
}

String OpenXRRenderModelExtension::render_model_get_animatable_node_name(RID p_render_model, uint32_t p_index) const {
	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, String());

	if (render_model->render_model_data.is_null()) {
		// We never loaded it (don't spam errors here).
		return String();
	}

	return render_model->render_model_data->get_node_name(p_index);
}

bool OpenXRRenderModelExtension::render_model_is_animatable_node_visible(RID p_render_model, uint32_t p_index) const {
	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, false);

	ERR_FAIL_UNSIGNED_INDEX_V(p_index, render_model->animatable_node_count, false);

	if (render_model->node_states.is_empty()) {
		// Never allocated (don't spam errors here).
		return false;
	}

	return render_model->node_states[p_index].isVisible;
}

Transform3D OpenXRRenderModelExtension::render_model_get_animatable_node_transform(RID p_render_model, uint32_t p_index) const {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, Transform3D());

	RenderModel *render_model = render_model_owner.get_or_null(p_render_model);
	ERR_FAIL_NULL_V(render_model, Transform3D());

	ERR_FAIL_UNSIGNED_INDEX_V(p_index, render_model->animatable_node_count, Transform3D());

	if (render_model->node_states.is_empty()) {
		// Never allocated (don't spam errors here).
		return Transform3D();
	}

	return openxr_api->transform_from_pose(render_model->node_states[p_index].nodePose);
}

Ref<OpenXRRenderModelData> OpenXRRenderModelExtension::_get_render_model_data(XrUuidEXT p_cache_id, uint32_t p_animatable_node_count) {
	if (render_model_data_cache.has(p_cache_id)) {
		return render_model_data_cache[p_cache_id];
	}

	// We don't have this cached, lets load it up

	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	ERR_FAIL_NULL_V(openxr_api, nullptr);

	XrSession session = openxr_api->get_session();
	ERR_FAIL_COND_V(session == XR_NULL_HANDLE, nullptr);

	XrRenderModelAssetEXT asset;

	XrRenderModelAssetCreateInfoEXT create_info = {
		XR_TYPE_RENDER_MODEL_ASSET_CREATE_INFO_EXT, // type
		nullptr, // next
		p_cache_id // cacheId
	};

	XrResult result = xrCreateRenderModelAssetEXT(session, &create_info, &asset);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(nullptr, "OpenXR: Failed to create render model asset [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	Ref<OpenXRRenderModelData> render_model_data = _load_asset(asset, p_animatable_node_count);

	// We're done with this :)
	result = xrDestroyRenderModelAssetEXT(asset);
	if (XR_FAILED(result)) {
		ERR_PRINT("OpenXR: Failed to destroy render model asset [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	// And cache it
	render_model_data_cache[p_cache_id] = render_model_data;

	return render_model_data;
}

Ref<OpenXRRenderModelData> OpenXRRenderModelExtension::_load_asset(XrRenderModelAssetEXT p_asset, uint32_t p_animatable_node_count) {
	XrRenderModelAssetDataGetInfoEXT get_info = {
		XR_TYPE_RENDER_MODEL_ASSET_DATA_GET_INFO_EXT, // type
		nullptr, // next
	};

	XrRenderModelAssetDataEXT asset_data = {
		XR_TYPE_RENDER_MODEL_ASSET_DATA_EXT, // type
		nullptr, // next
		0, // bufferCapacityInput;
		0, // bufferCountOutput;
		nullptr // buffer;
	};

	// Obtain required size for the buffer.
	XrResult result = xrGetRenderModelAssetDataEXT(p_asset, &get_info, &asset_data);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(nullptr, "OpenXR: Failed to get render model buffer size [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}
	ERR_FAIL_COND_V(asset_data.bufferCountOutput == 0, nullptr);

	// Allocate data
	PackedByteArray buffer;
	buffer.resize(asset_data.bufferCountOutput);
	asset_data.buffer = buffer.ptrw();
	asset_data.bufferCapacityInput = asset_data.bufferCountOutput;

	// Now get our actual data.
	result = xrGetRenderModelAssetDataEXT(p_asset, &get_info, &asset_data);
	if (XR_FAILED(result)) {
		ERR_FAIL_V_MSG(nullptr, "OpenXR: Failed to get render model buffer [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
	}

	// Get the names of any animatable nodes
	PackedStringArray node_names;
	if (p_animatable_node_count > 0) {
		Vector<XrRenderModelAssetNodePropertiesEXT> node_properties;
		node_properties.resize(p_animatable_node_count);

		XrRenderModelAssetPropertiesGetInfoEXT properties_info = {
			XR_TYPE_RENDER_MODEL_ASSET_PROPERTIES_GET_INFO_EXT, // type
			nullptr, // next
		};

		XrRenderModelAssetPropertiesEXT asset_properties = {
			XR_TYPE_RENDER_MODEL_ASSET_PROPERTIES_EXT, // type
			nullptr, // next
			uint32_t(node_properties.size()), // nodePropertyCount
			node_properties.ptrw(), // nodeProperties
		};

		result = xrGetRenderModelAssetPropertiesEXT(p_asset, &properties_info, &asset_properties);
		if (XR_FAILED(result)) {
			ERR_FAIL_V_MSG(nullptr, "OpenXR: Failed to get render model property info [" + OpenXRAPI::get_singleton()->get_error_string(result) + "]");
		}

		node_names.resize(p_animatable_node_count);
		String *node_names_ptrw = node_names.ptrw();
		for (uint32_t i = 0; i < p_animatable_node_count; i++) {
			node_names_ptrw[i] = String(node_properties[i].uniqueName);
		}
	}

	Ref<OpenXRRenderModelData> render_model_data;
	render_model_data.instantiate();

	render_model_data->parse_gltf_document(buffer);
	render_model_data->set_node_names(node_names);

	return render_model_data;
}

void OpenXRRenderModelExtension::_clear_render_model_data() {
	// Clear our toplevel paths filter.
	toplevel_paths.clear();

	// Clear our render model cache.
	render_model_data_cache.clear();

	// Loop through all of our render models and destroy them.
	LocalVector<RID> owned = render_model_owner.get_owned_list();
	for (const RID &rid : owned) {
		render_model_destroy(rid);
	}
}

bool OpenXRRenderModelData::parse_gltf_document(const PackedByteArray &p_bytes) {
	// State holds our data, document parses GLTF
	Ref<GLTFState> new_state;
	new_state.instantiate();
	Ref<GLTFDocument> new_gltf_document;
	new_gltf_document.instantiate();

	Error err = new_gltf_document->append_from_buffer(p_bytes, "", new_state);
	if (err != OK) {
		ERR_FAIL_V_MSG(false, "OpenXR: Failed to parse GLTF data.");
	}

	gltf_document = new_gltf_document;
	gltf_state = new_state;
	return true;
}

Node3D *OpenXRRenderModelData::new_scene_instance() {
	ERR_FAIL_COND_V(gltf_document.is_null(), nullptr);
	ERR_FAIL_COND_V(gltf_state.is_null(), nullptr);

	return Object::cast_to<Node3D>(gltf_document->generate_scene(gltf_state));
}

void OpenXRRenderModelData::set_node_names(const PackedStringArray &p_node_names) {
	node_names = p_node_names;
}

PackedStringArray OpenXRRenderModelData::get_node_names() const {
	return node_names;
}

const String OpenXRRenderModelData::get_node_name(uint32_t p_node_index) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node_index, node_names.size(), String());

	return node_names[p_node_index];
}

OpenXRRenderModelData::OpenXRRenderModelData() {
}

OpenXRRenderModelData::~OpenXRRenderModelData() {
}
#endif // MODULE_GLTF_ENABLED
