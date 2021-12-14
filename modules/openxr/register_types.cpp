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

OpenXRAPI *openxr_api = nullptr;
Ref<OpenXRInterface> openxr_interface;

void preregister_openxr_types() {
	// For now we create our openxr device here. If we merge it with openxr_interface we'll create that here soon.

	OpenXRAPI::setup_global_defs();
	openxr_api = OpenXRAPI::get_singleton();
	if (openxr_api) {
		if (!openxr_api->initialise(Main::get_rendering_driver_name())) {
			return;
		}
	}
}

void register_openxr_types() {
	GDREGISTER_CLASS(OpenXRInterface);

	GDREGISTER_CLASS(OpenXRAction);
	GDREGISTER_CLASS(OpenXRActionSet);
	GDREGISTER_CLASS(OpenXRActionMap);
	GDREGISTER_CLASS(OpenXRIPBinding);
	GDREGISTER_CLASS(OpenXRInteractionProfile);

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server) {
		openxr_interface.instantiate();
		xr_server->add_interface(openxr_interface);

		if (openxr_interface->initialise_on_startup()) {
			openxr_interface->initialize();
		}
	}
}

void unregister_openxr_types() {
	if (openxr_interface.is_valid()) {
		// unregister our interface from the XR server
		if (XRServer::get_singleton()) {
			XRServer::get_singleton()->remove_interface(openxr_interface);
		}

		// and release
		openxr_interface.unref();
	}

	if (openxr_api) {
		openxr_api->finish();
		memdelete(openxr_api);
	}
}
