/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "register_types.h"
#include "main/main.h"

#include "openxr_interface.h"

#include "action_map/openxr_action.h"
#include "action_map/openxr_action_map.h"
#include "action_map/openxr_action_set.h"
#include "action_map/openxr_interaction_profile.h"
#include "action_map/openxr_interaction_profile_meta_data.h"

#include "scene/openxr_hand.h"

static OpenXRAPI *openxr_api = nullptr;
static OpenXRInteractionProfileMetaData *openxr_interaction_profile_meta_data = nullptr;
static Ref<OpenXRInterface> openxr_interface;

#ifdef TOOLS_ENABLED

#include "editor/editor_node.h"
#include "editor/openxr_editor_plugin.h"

static void _editor_init() {
	if (OpenXRAPI::openxr_is_enabled(false)) {
		// Only add our OpenXR action map editor if OpenXR is enabled for our project

		if (openxr_interaction_profile_meta_data == nullptr) {
			// If we didn't initialize our actionmap meta data at startup, we initialise it now.
			openxr_interaction_profile_meta_data = memnew(OpenXRInteractionProfileMetaData);
			ERR_FAIL_NULL(openxr_interaction_profile_meta_data);
		}

		OpenXREditorPlugin *openxr_plugin = memnew(OpenXREditorPlugin());
		EditorNode::get_singleton()->add_editor_plugin(openxr_plugin);
	}
}

#endif

void initialize_openxr_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		// For now we create our openxr device here. If we merge it with openxr_interface we'll create that here soon.

		if (OpenXRAPI::openxr_is_enabled()) {
			openxr_interaction_profile_meta_data = memnew(OpenXRInteractionProfileMetaData);
			ERR_FAIL_NULL(openxr_interaction_profile_meta_data);
			openxr_api = memnew(OpenXRAPI);
			ERR_FAIL_NULL(openxr_api);

			if (!openxr_api->initialize(Main::get_rendering_driver_name())) {
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
		GDREGISTER_CLASS(OpenXRInteractionProfileMetaData);
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

	if (openxr_interaction_profile_meta_data) {
		memdelete(openxr_interaction_profile_meta_data);
		openxr_interaction_profile_meta_data = nullptr;
	}
}
