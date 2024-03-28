/**************************************************************************/
/*  openxr_composition_layer.cpp                                          */
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

#include "openxr_composition_layer.h"

#include "../extensions/openxr_composition_layer_extension.h"
#include "../openxr_api.h"
#include "../openxr_interface.h"

#include "scene/3d/mesh_instance_3d.h"
#include "scene/main/viewport.h"

OpenXRCompositionLayer::OpenXRCompositionLayer() {
	openxr_api = OpenXRAPI::get_singleton();

	Ref<OpenXRInterface> openxr_interface = XRServer::get_singleton()->find_interface("OpenXR");
	if (openxr_interface.is_valid()) {
		openxr_interface->connect("session_begun", callable_mp(this, &OpenXRCompositionLayer::_on_openxr_session_begun));
		openxr_interface->connect("session_stopping", callable_mp(this, &OpenXRCompositionLayer::_on_openxr_session_stopping));
	}

	set_process_internal(true);

	if (Engine::get_singleton()->is_editor_hint()) {
		// In the editor, create the fallback right away.
		_create_fallback_node();
	}
}

OpenXRCompositionLayer::~OpenXRCompositionLayer() {
	Ref<OpenXRInterface> openxr_interface = XRServer::get_singleton()->find_interface("OpenXR");
	if (openxr_interface.is_valid()) {
		openxr_interface->disconnect("session_begun", callable_mp(this, &OpenXRCompositionLayer::_on_openxr_session_begun));
		openxr_interface->disconnect("session_stopping", callable_mp(this, &OpenXRCompositionLayer::_on_openxr_session_stopping));
	}

	if (openxr_layer_provider != nullptr) {
		memdelete(openxr_layer_provider);
		openxr_layer_provider = nullptr;
	}
}

void OpenXRCompositionLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_layer_viewport", "viewport"), &OpenXRCompositionLayer::set_layer_viewport);
	ClassDB::bind_method(D_METHOD("get_layer_viewport"), &OpenXRCompositionLayer::get_layer_viewport);

	ClassDB::bind_method(D_METHOD("set_sort_order", "order"), &OpenXRCompositionLayer::set_sort_order);
	ClassDB::bind_method(D_METHOD("get_sort_order"), &OpenXRCompositionLayer::get_sort_order);

	ClassDB::bind_method(D_METHOD("is_natively_supported"), &OpenXRCompositionLayer::is_natively_supported);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "layer_viewport", PROPERTY_HINT_NODE_TYPE, "SubViewport"), "set_layer_viewport", "get_layer_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sort_order", PROPERTY_HINT_NONE, ""), "set_sort_order", "get_sort_order");
}

void OpenXRCompositionLayer::_create_fallback_node() {
	ERR_FAIL_COND(fallback);
	fallback = memnew(MeshInstance3D);
	add_child(fallback, false, INTERNAL_MODE_FRONT);
	should_update_fallback_mesh = true;
}

void OpenXRCompositionLayer::_on_openxr_session_begun() {
	if (!is_natively_supported()) {
		if (!fallback) {
			_create_fallback_node();
		}
	} else if (layer_viewport && is_visible() && is_inside_tree()) {
		openxr_layer_provider->set_viewport(layer_viewport);
	}
}

void OpenXRCompositionLayer::_on_openxr_session_stopping() {
	if (fallback && !Engine::get_singleton()->is_editor_hint()) {
		fallback->queue_free();
		remove_child(fallback);
		fallback = nullptr;
	} else {
		openxr_layer_provider->set_viewport(nullptr);
	}
}

void OpenXRCompositionLayer::update_fallback_mesh() {
	should_update_fallback_mesh = true;
}

void OpenXRCompositionLayer::set_layer_viewport(SubViewport *p_viewport) {
	layer_viewport = p_viewport;
	if (fallback) {
		_reset_fallback_material();
	} else if (openxr_api && openxr_api->is_running() && is_visible() && is_inside_tree()) {
		openxr_layer_provider->set_viewport(layer_viewport);
	}
}

SubViewport *OpenXRCompositionLayer::get_layer_viewport() const {
	return layer_viewport;
}

void OpenXRCompositionLayer::set_sort_order(int p_order) {
	if (openxr_layer_provider) {
		openxr_layer_provider->set_sort_order(p_order);
	}
}

int OpenXRCompositionLayer::get_sort_order() const {
	if (openxr_layer_provider) {
		return openxr_layer_provider->get_sort_order();
	}
	return 1;
}

bool OpenXRCompositionLayer::is_natively_supported() const {
	return OpenXRCompositionLayerExtension::get_singleton()->is_available(openxr_layer_provider->get_openxr_type());
}

void OpenXRCompositionLayer::_reset_fallback_material() {
	ERR_FAIL_NULL(fallback);

	if (fallback->get_mesh().is_null()) {
		return;
	}

	if (layer_viewport) {
		Ref<StandardMaterial3D> material = fallback->get_surface_override_material(0);
		if (material.is_null()) {
			material.instantiate();
			material->set_local_to_scene(true);
			fallback->set_surface_override_material(0, material);
		}

		Ref<ViewportTexture> texture = material->get_texture(StandardMaterial3D::TEXTURE_ALBEDO);
		if (texture.is_null()) {
			texture.instantiate();
			// ViewportTexture can't be configured without a local scene, so use this hack to set it.
			HashMap<Ref<Resource>, Ref<Resource>> remap_cache;
			texture->configure_for_local_scene(this, remap_cache);
		}

		Node *loc_scene = texture->get_local_scene();
		NodePath viewport_path = loc_scene->get_path_to(layer_viewport);
		texture->set_viewport_path_in_scene(viewport_path);
		material->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, texture);
	} else {
		fallback->set_surface_override_material(0, Ref<Material>());
	}
}

void OpenXRCompositionLayer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (fallback) {
				if (should_update_fallback_mesh) {
					fallback->set_mesh(_create_fallback_mesh());
					_reset_fallback_material();
					should_update_fallback_mesh = false;
				}
			} else if (is_visible()) {
				openxr_layer_provider->process();
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!fallback && openxr_api && openxr_api->is_running() && is_inside_tree()) {
				openxr_layer_provider->set_viewport(is_visible() ? layer_viewport : nullptr);
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (openxr_api) {
				// Register our composition layer provider to our OpenXR API.
				openxr_api->register_composition_layer_provider(openxr_layer_provider);
			}
			if (!fallback && layer_viewport && openxr_api && openxr_api->is_running() && is_visible()) {
				openxr_layer_provider->set_viewport(layer_viewport);
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (openxr_api) {
				// Unregister our composition layer provider.
				openxr_api->unregister_composition_layer_provider(openxr_layer_provider);
			}
			if (!fallback) {
				// This will clean up existing resources.
				openxr_layer_provider->set_viewport(nullptr);
			}
		} break;
	}
}
