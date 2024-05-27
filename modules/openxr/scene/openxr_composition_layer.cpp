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
#include "scene/3d/xr_nodes.h"
#include "scene/main/viewport.h"

Vector<OpenXRCompositionLayer *> OpenXRCompositionLayer::composition_layer_nodes;

static const char *HOLE_PUNCH_SHADER_CODE =
		"shader_type spatial;\n"
		"render_mode blend_mix, depth_draw_opaque, cull_back, shadow_to_opacity, shadows_disabled;\n"
		"void fragment() {\n"
		"\tALBEDO = vec3(0.0, 0.0, 0.0);\n"
		"}\n";

OpenXRCompositionLayer::OpenXRCompositionLayer() {
	openxr_api = OpenXRAPI::get_singleton();
	composition_layer_extension = OpenXRCompositionLayerExtension::get_singleton();

	Ref<OpenXRInterface> openxr_interface = XRServer::get_singleton()->find_interface("OpenXR");
	if (openxr_interface.is_valid()) {
		openxr_interface->connect("session_begun", callable_mp(this, &OpenXRCompositionLayer::_on_openxr_session_begun));
		openxr_interface->connect("session_stopping", callable_mp(this, &OpenXRCompositionLayer::_on_openxr_session_stopping));
	}

	set_process_internal(true);
	set_notify_local_transform(true);

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

	composition_layer_nodes.erase(this);

	if (openxr_layer_provider != nullptr) {
		memdelete(openxr_layer_provider);
		openxr_layer_provider = nullptr;
	}
}

void OpenXRCompositionLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_layer_viewport", "viewport"), &OpenXRCompositionLayer::set_layer_viewport);
	ClassDB::bind_method(D_METHOD("get_layer_viewport"), &OpenXRCompositionLayer::get_layer_viewport);

	ClassDB::bind_method(D_METHOD("set_enable_hole_punch", "enable"), &OpenXRCompositionLayer::set_enable_hole_punch);
	ClassDB::bind_method(D_METHOD("get_enable_hole_punch"), &OpenXRCompositionLayer::get_enable_hole_punch);

	ClassDB::bind_method(D_METHOD("set_sort_order", "order"), &OpenXRCompositionLayer::set_sort_order);
	ClassDB::bind_method(D_METHOD("get_sort_order"), &OpenXRCompositionLayer::get_sort_order);

	ClassDB::bind_method(D_METHOD("set_alpha_blend", "enabled"), &OpenXRCompositionLayer::set_alpha_blend);
	ClassDB::bind_method(D_METHOD("get_alpha_blend"), &OpenXRCompositionLayer::get_alpha_blend);

	ClassDB::bind_method(D_METHOD("is_natively_supported"), &OpenXRCompositionLayer::is_natively_supported);

	ClassDB::bind_method(D_METHOD("intersects_ray", "origin", "direction"), &OpenXRCompositionLayer::intersects_ray);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "layer_viewport", PROPERTY_HINT_NODE_TYPE, "SubViewport"), "set_layer_viewport", "get_layer_viewport");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sort_order", PROPERTY_HINT_NONE, ""), "set_sort_order", "get_sort_order");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_blend", PROPERTY_HINT_NONE, ""), "set_alpha_blend", "get_alpha_blend");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_hole_punch", PROPERTY_HINT_NONE, ""), "set_enable_hole_punch", "get_enable_hole_punch");
}

bool OpenXRCompositionLayer::_should_use_fallback_node() {
	if (Engine::get_singleton()->is_editor_hint()) {
		return true;
	} else if (openxr_session_running) {
		return enable_hole_punch || !is_natively_supported();
	}
	return false;
}

void OpenXRCompositionLayer::_create_fallback_node() {
	ERR_FAIL_COND(fallback);
	fallback = memnew(MeshInstance3D);
	fallback->set_cast_shadows_setting(GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
	add_child(fallback, false, INTERNAL_MODE_FRONT);
	should_update_fallback_mesh = true;
}

void OpenXRCompositionLayer::_remove_fallback_node() {
	ERR_FAIL_COND(fallback != nullptr);
	remove_child(fallback);
	fallback->queue_free();
	fallback = nullptr;
}

void OpenXRCompositionLayer::_on_openxr_session_begun() {
	openxr_session_running = true;
	if (layer_viewport && is_natively_supported() && is_visible() && is_inside_tree()) {
		openxr_layer_provider->set_viewport(layer_viewport->get_viewport_rid(), layer_viewport->get_size());
	}
	if (!fallback && _should_use_fallback_node()) {
		_create_fallback_node();
	}
}

void OpenXRCompositionLayer::_on_openxr_session_stopping() {
	openxr_session_running = false;
	if (fallback && !_should_use_fallback_node()) {
		_remove_fallback_node();
	} else {
		openxr_layer_provider->set_viewport(RID(), Size2i());
	}
}

void OpenXRCompositionLayer::update_fallback_mesh() {
	should_update_fallback_mesh = true;
}

bool OpenXRCompositionLayer::is_viewport_in_use(SubViewport *p_viewport) {
	for (const OpenXRCompositionLayer *other_composition_layer : composition_layer_nodes) {
		if (other_composition_layer != this && other_composition_layer->is_inside_tree() && other_composition_layer->get_layer_viewport() == p_viewport) {
			return true;
		}
	}
	return false;
}

void OpenXRCompositionLayer::set_layer_viewport(SubViewport *p_viewport) {
	if (layer_viewport == p_viewport) {
		return;
	}

	if (p_viewport != nullptr) {
		ERR_FAIL_COND_EDMSG(is_viewport_in_use(p_viewport), RTR("Cannot use the same SubViewport with multiple OpenXR composition layers. Clear it from its current layer first."));
	}

	layer_viewport = p_viewport;

	if (layer_viewport) {
		SubViewport::UpdateMode update_mode = layer_viewport->get_update_mode();
		if (update_mode == SubViewport::UPDATE_WHEN_VISIBLE || update_mode == SubViewport::UPDATE_WHEN_PARENT_VISIBLE) {
			WARN_PRINT_ONCE("OpenXR composition layers cannot use SubViewports with UPDATE_WHEN_VISIBLE or UPDATE_WHEN_PARENT_VISIBLE. Switching to UPDATE_ALWAYS.");
			layer_viewport->set_update_mode(SubViewport::UPDATE_ALWAYS);
		}
	}

	if (fallback) {
		_reset_fallback_material();
	} else if (openxr_session_running && is_visible() && is_inside_tree()) {
		if (layer_viewport) {
			openxr_layer_provider->set_viewport(layer_viewport->get_viewport_rid(), layer_viewport->get_size());
		} else {
			openxr_layer_provider->set_viewport(RID(), Size2i());
		}
	}
}

SubViewport *OpenXRCompositionLayer::get_layer_viewport() const {
	return layer_viewport;
}

void OpenXRCompositionLayer::set_enable_hole_punch(bool p_enable) {
	if (enable_hole_punch == p_enable) {
		return;
	}

	enable_hole_punch = p_enable;
	if (_should_use_fallback_node()) {
		if (fallback) {
			_reset_fallback_material();
		} else {
			_create_fallback_node();
		}
	} else if (fallback) {
		_remove_fallback_node();
	}

	update_configuration_warnings();
}

bool OpenXRCompositionLayer::get_enable_hole_punch() const {
	return enable_hole_punch;
}

void OpenXRCompositionLayer::set_sort_order(int p_order) {
	if (openxr_layer_provider) {
		openxr_layer_provider->set_sort_order(p_order);
		update_configuration_warnings();
	}
}

int OpenXRCompositionLayer::get_sort_order() const {
	if (openxr_layer_provider) {
		return openxr_layer_provider->get_sort_order();
	}
	return 1;
}

void OpenXRCompositionLayer::set_alpha_blend(bool p_alpha_blend) {
	if (openxr_layer_provider) {
		openxr_layer_provider->set_alpha_blend(p_alpha_blend);
		if (fallback) {
			_reset_fallback_material();
		}
	}
}

bool OpenXRCompositionLayer::get_alpha_blend() const {
	if (openxr_layer_provider) {
		return openxr_layer_provider->get_alpha_blend();
	}
	return false;
}

bool OpenXRCompositionLayer::is_natively_supported() const {
	if (composition_layer_extension) {
		return composition_layer_extension->is_available(openxr_layer_provider->get_openxr_type());
	}
	return false;
}

Vector2 OpenXRCompositionLayer::intersects_ray(const Vector3 &p_origin, const Vector3 &p_direction) const {
	return Vector2(-1.0, -1.0);
}

void OpenXRCompositionLayer::_reset_fallback_material() {
	ERR_FAIL_NULL(fallback);

	if (fallback->get_mesh().is_null()) {
		return;
	}

	if (enable_hole_punch && !Engine::get_singleton()->is_editor_hint() && is_natively_supported()) {
		Ref<ShaderMaterial> material = fallback->get_surface_override_material(0);
		if (material.is_null()) {
			Ref<Shader> shader;
			shader.instantiate();
			shader->set_code(HOLE_PUNCH_SHADER_CODE);

			material.instantiate();
			material->set_shader(shader);

			fallback->set_surface_override_material(0, material);
		}
	} else if (layer_viewport) {
		Ref<StandardMaterial3D> material = fallback->get_surface_override_material(0);
		if (material.is_null()) {
			material.instantiate();
			material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
			material->set_local_to_scene(true);
			fallback->set_surface_override_material(0, material);
		}

		material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, !enable_hole_punch);
		material->set_transparency(get_alpha_blend() ? StandardMaterial3D::TRANSPARENCY_ALPHA : StandardMaterial3D::TRANSPARENCY_DISABLED);

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
		case NOTIFICATION_POSTINITIALIZE: {
			composition_layer_nodes.push_back(this);

			if (openxr_layer_provider) {
				for (OpenXRExtensionWrapper *extension : OpenXRAPI::get_registered_extension_wrappers()) {
					extension_property_values.merge(extension->get_viewport_composition_layer_extension_property_defaults());
				}
				openxr_layer_provider->set_extension_property_values(extension_property_values);
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (fallback) {
				if (should_update_fallback_mesh) {
					fallback->set_mesh(_create_fallback_mesh());
					_reset_fallback_material();
					should_update_fallback_mesh = false;
				}
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!fallback && openxr_session_running && is_inside_tree()) {
				if (layer_viewport && is_visible()) {
					openxr_layer_provider->set_viewport(layer_viewport->get_viewport_rid(), layer_viewport->get_size());
				} else {
					openxr_layer_provider->set_viewport(RID(), Size2i());
				}
			}
			update_configuration_warnings();
		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			update_configuration_warnings();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (composition_layer_extension) {
				composition_layer_extension->register_viewport_composition_layer_provider(openxr_layer_provider);
			}

			if (is_viewport_in_use(layer_viewport)) {
				set_layer_viewport(nullptr);
			} else if (!fallback && layer_viewport && openxr_session_running && is_visible()) {
				openxr_layer_provider->set_viewport(layer_viewport->get_viewport_rid(), layer_viewport->get_size());
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (composition_layer_extension) {
				composition_layer_extension->unregister_viewport_composition_layer_provider(openxr_layer_provider);
			}

			if (!fallback) {
				// This will clean up existing resources.
				openxr_layer_provider->set_viewport(RID(), Size2i());
			}
		} break;
	}
}

void OpenXRCompositionLayer::_get_property_list(List<PropertyInfo> *p_property_list) const {
	List<PropertyInfo> extension_properties;
	for (OpenXRExtensionWrapper *extension : OpenXRAPI::get_registered_extension_wrappers()) {
		extension->get_viewport_composition_layer_extension_properties(&extension_properties);
	}

	for (const PropertyInfo &pinfo : extension_properties) {
		StringName prop_name = pinfo.name;
		if (!String(prop_name).contains("/")) {
			WARN_PRINT_ONCE(vformat("Discarding OpenXRCompositionLayer property name '%s' from extension because it doesn't contain a '/'."));
			continue;
		}
		p_property_list->push_back(pinfo);
	}
}

bool OpenXRCompositionLayer::_get(const StringName &p_property, Variant &r_value) const {
	if (extension_property_values.has(p_property)) {
		r_value = extension_property_values[p_property];
	}

	return true;
}

bool OpenXRCompositionLayer::_set(const StringName &p_property, const Variant &p_value) {
	extension_property_values[p_property] = p_value;

	if (openxr_layer_provider) {
		openxr_layer_provider->set_extension_property_values(extension_property_values);
	}

	return true;
}

PackedStringArray OpenXRCompositionLayer::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();

	if (is_visible() && is_inside_tree()) {
		XROrigin3D *origin = Object::cast_to<XROrigin3D>(get_parent());
		if (origin == nullptr) {
			warnings.push_back(RTR("OpenXR composition layers must have an XROrigin3D node as their parent."));
		}
	}

	if (!get_transform().basis.is_orthonormal()) {
		warnings.push_back(RTR("OpenXR composition layers must have orthonormalized transforms (ie. no scale or shearing)."));
	}

	if (enable_hole_punch && get_sort_order() >= 0) {
		warnings.push_back(RTR("Hole punching won't work as expected unless the sort order is less than zero."));
	}

	return warnings;
}
