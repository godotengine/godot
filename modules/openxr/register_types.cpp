/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "action_map/openxr_action.h"
#include "action_map/openxr_action_map.h"
#include "action_map/openxr_action_set.h"
#include "action_map/openxr_interaction_profile.h"
#include "action_map/openxr_interaction_profile_metadata.h"
#include "openxr_interface.h"

#include "extensions/openxr_extension_wrapper_extension.h"

#include "scene/openxr_hand.h"

#include "extensions/openxr_composition_layer_depth_extension.h"
#include "extensions/openxr_eye_gaze_interaction.h"
#include "extensions/openxr_fb_display_refresh_rate_extension.h"
#include "extensions/openxr_fb_passthrough_extension_wrapper.h"
#include "extensions/openxr_hand_tracking_extension.h"
#include "extensions/openxr_htc_controller_extension.h"
#include "extensions/openxr_htc_vive_tracker_extension.h"
#include "extensions/openxr_huawei_controller_extension.h"
#include "extensions/openxr_ml2_controller_extension.h"
#include "extensions/openxr_palm_pose_extension.h"
#include "extensions/openxr_pico_controller_extension.h"
#include "extensions/openxr_wmr_controller_extension.h"

#ifdef TOOLS_ENABLED
#include "editor/openxr_editor_plugin.h"
#endif

#ifdef ANDROID_ENABLED
#include "extensions/openxr_android_extension.h"
#endif

#include "core/config/project_settings.h"
#include "main/main.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

static OpenXRAPI *openxr_api = nullptr;
static OpenXRInteractionProfileMetadata *openxr_interaction_profile_metadata = nullptr;
static Ref<OpenXRInterface> openxr_interface;

#ifdef TOOLS_ENABLED
static void _editor_init() {
	if (OpenXRAPI::openxr_is_enabled(false)) {
		// Only add our OpenXR action map editor if OpenXR is enabled for our project

		if (openxr_interaction_profile_metadata == nullptr) {
			// If we didn't initialize our actionmap metadata at startup, we initialize it now.
			openxr_interaction_profile_metadata = memnew(OpenXRInteractionProfileMetadata);
			ERR_FAIL_NULL(openxr_interaction_profile_metadata);
		}

		OpenXREditorPlugin *openxr_plugin = memnew(OpenXREditorPlugin());
		EditorNode::get_singleton()->add_editor_plugin(openxr_plugin);
	}
}
#endif

void initialize_openxr_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
		GDREGISTER_CLASS(OpenXRExtensionWrapperExtension);
		GDREGISTER_CLASS(OpenXRAPIExtension);
	}

	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		if (OpenXRAPI::openxr_is_enabled(false)) {
			// Always register our extension wrappers even if we don't initialize OpenXR.
			// Some of these wrappers will add functionality to our editor.
#ifdef ANDROID_ENABLED
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRAndroidExtension));
#endif

			// register our other extensions
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRPalmPoseExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRPicoControllerExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRCompositionLayerDepthExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRHTCControllerExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRHTCViveTrackerExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRHuaweiControllerExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRFbPassthroughExtensionWrapper));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRDisplayRefreshRateExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRWMRControllerExtension));
			OpenXRAPI::register_extension_wrapper(memnew(OpenXRML2ControllerExtension));

			// register gated extensions
			if (GLOBAL_GET("xr/openxr/extensions/eye_gaze_interaction") && (!OS::get_singleton()->has_feature("mobile") || OS::get_singleton()->has_feature(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME))) {
				OpenXRAPI::register_extension_wrapper(memnew(OpenXREyeGazeInteractionExtension));
			}
			if (GLOBAL_GET("xr/openxr/extensions/hand_tracking")) {
				OpenXRAPI::register_extension_wrapper(memnew(OpenXRHandTrackingExtension));
			}
		}

		if (OpenXRAPI::openxr_is_enabled()) {
			openxr_interaction_profile_metadata = memnew(OpenXRInteractionProfileMetadata);
			ERR_FAIL_NULL(openxr_interaction_profile_metadata);
			openxr_api = memnew(OpenXRAPI);
			ERR_FAIL_NULL(openxr_api);

			if (!openxr_api->initialize(Main::get_rendering_driver_name())) {
				const char *init_error_message =
						"OpenXR was requested but failed to start.\n"
						"Please check if your HMD is connected.\n"
						"When using Windows MR please note that WMR only has DirectX support, make sure SteamVR is your default OpenXR runtime.\n"
						"Godot will start in normal mode.\n";

				WARN_PRINT(init_error_message);

				bool init_show_startup_alert = GLOBAL_GET("xr/openxr/startup_alert");
				if (init_show_startup_alert) {
					OS::get_singleton()->alert(init_error_message);
				}

				memdelete(openxr_api);
				openxr_api = nullptr;
				return;
			}
		}
	}

	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(OpenXRInterface);

		GDREGISTER_CLASS(OpenXRAction);
		GDREGISTER_CLASS(OpenXRActionSet);
		GDREGISTER_CLASS(OpenXRActionMap);
		GDREGISTER_CLASS(OpenXRInteractionProfileMetadata);
		GDREGISTER_CLASS(OpenXRIPBinding);
		GDREGISTER_CLASS(OpenXRInteractionProfile);

		GDREGISTER_CLASS(OpenXRHand);

		XRServer *xr_server = XRServer::get_singleton();
		if (xr_server) {
			openxr_interface.instantiate();
			xr_server->add_interface(openxr_interface);

			if (openxr_interface->initialize_on_startup()) {
				openxr_interface->initialize();
			}
		}

#ifdef TOOLS_ENABLED
		EditorNode::add_init_callback(_editor_init);
#endif
	}
}

void uninitialize_openxr_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (openxr_interface.is_valid()) {
		// uninitialize just in case
		if (openxr_interface->is_initialized()) {
			openxr_interface->uninitialize();
		}

		// unregister our interface from the XR server
		XRServer *xr_server = XRServer::get_singleton();
		if (xr_server) {
			if (xr_server->get_primary_interface() == openxr_interface) {
				xr_server->set_primary_interface(Ref<XRInterface>());
			}
			xr_server->remove_interface(openxr_interface);
		}

		// and release
		openxr_interface.unref();
	}

	if (openxr_api) {
		openxr_api->finish();

		memdelete(openxr_api);
		openxr_api = nullptr;
	}

	if (openxr_interaction_profile_metadata) {
		memdelete(openxr_interaction_profile_metadata);
		openxr_interaction_profile_metadata = nullptr;
	}

	// cleanup our extension wrappers
	OpenXRAPI::cleanup_extension_wrappers();
}
