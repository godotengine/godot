/**************************************************************************/
/*  openxr_render_model.cpp                                               */
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

#include "openxr_render_model.h"

#ifdef MODULE_GLTF_ENABLED

#include "../extensions/openxr_render_model_extension.h"
#include "core/config/project_settings.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/xr/xr_nodes.h"
#include "scene/resources/3d/primitive_meshes.h"

void OpenXRRenderModel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_top_level_path"), &OpenXRRenderModel::get_top_level_path);

	ClassDB::bind_method(D_METHOD("get_render_model"), &OpenXRRenderModel::get_render_model);
	ClassDB::bind_method(D_METHOD("set_render_model", "render_model"), &OpenXRRenderModel::set_render_model);
	ADD_PROPERTY(PropertyInfo(Variant::RID, "render_model"), "set_render_model", "get_render_model");

	ADD_SIGNAL(MethodInfo("render_model_top_level_path_changed"));
}

void OpenXRRenderModel::_load_render_model_scene() {
	OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
	ERR_FAIL_NULL(render_model_extension);
	ERR_FAIL_COND(render_model.is_null());

	scene = render_model_extension->render_model_new_scene_instance(render_model);
	if (scene) {
		// Get and cache our animatable nodes.
		animatable_nodes.clear();
		uint32_t count = render_model_extension->render_model_get_animatable_node_count(render_model);
		for (uint32_t i = 0; i < count; i++) {
			String node_name = render_model_extension->render_model_get_animatable_node_name(render_model, i);
			if (!node_name.is_empty()) {
				Node3D *child = Object::cast_to<Node3D>(scene->find_child(node_name));
				if (child) {
					animatable_nodes[node_name] = child;
				}
			}
		}

		// Now add to scene.
		add_child(scene);
	}
}

void OpenXRRenderModel::_on_render_model_top_level_path_changed(RID p_render_model) {
	if (render_model == p_render_model) {
		emit_signal("render_model_top_level_path_changed");
	}
}

void OpenXRRenderModel::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
			ERR_FAIL_NULL(render_model_extension);
			if (render_model.is_valid()) {
				_load_render_model_scene();
			}

			set_process_internal(true);
			render_model_extension->connect("render_model_top_level_path_changed", callable_mp(this, &OpenXRRenderModel::_on_render_model_top_level_path_changed));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);

			if (scene) {
				animatable_nodes.clear();

				remove_child(scene);
				scene->queue_free();
				scene = nullptr;
			}

			OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
			if (render_model_extension) {
				render_model_extension->disconnect("render_model_top_level_path_changed", callable_mp(this, &OpenXRRenderModel::_on_render_model_top_level_path_changed));
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (render_model.is_valid()) {
				OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
				ERR_FAIL_NULL(render_model_extension);

				if (render_model_extension->render_model_get_confidence(render_model) != XRPose::TrackingConfidence::XR_TRACKING_CONFIDENCE_NONE) {
					set_transform(render_model_extension->render_model_get_root_transform(render_model));

					if (scene) {
						uint32_t count = render_model_extension->render_model_get_animatable_node_count(render_model);
						for (uint32_t i = 0; i < count; i++) {
							String node_name = render_model_extension->render_model_get_animatable_node_name(render_model, i);
							if (!node_name.is_empty() && animatable_nodes.has(node_name)) {
								Node3D *child = animatable_nodes[node_name];
								child->set_visible(render_model_extension->render_model_is_animatable_node_visible(render_model, i));
								child->set_transform(render_model_extension->render_model_get_animatable_node_transform(render_model, i));
							}
						}
					}
				}
			}
		} break;
	}
}

String OpenXRRenderModel::get_top_level_path() const {
	String ret;

	OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
	if (render_model.is_valid() && render_model_extension) {
		ret = render_model_extension->render_model_get_top_level_path_as_string(render_model);
	}

	return ret;
}

PackedStringArray OpenXRRenderModel::get_configuration_warnings() const {
	PackedStringArray warnings;

	Node *parent = get_parent();
	if (!parent->is_class("XROrigin3D") && !parent->is_class("OpenXRRenderModelManager")) {
		warnings.push_back("This node must be a child of either a XROrigin3D or OpenXRRenderModelManager node!");
	}

	if (!GLOBAL_GET("xr/openxr/extensions/render_model")) {
		warnings.push_back("The render model extension is not enabled in project settings!");
	}

	return warnings;
}

RID OpenXRRenderModel::get_render_model() const {
	return render_model;
}

void OpenXRRenderModel::set_render_model(RID p_render_model) {
	render_model = p_render_model;
	if (is_inside_tree() && render_model.is_valid()) {
		_load_render_model_scene();
	}
}
#endif // MODULE_GLTF_ENABLED
