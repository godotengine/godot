/**************************************************************************/
/*  openxr_render_model_manager.cpp                                       */
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

#include "openxr_render_model_manager.h"

#ifdef MODULE_GLTF_ENABLED
#include "../extensions/openxr_render_model_extension.h"

#include "../openxr_api.h"
#include "core/config/project_settings.h"
#include "scene/3d/xr/xr_nodes.h"
#include "servers/xr/xr_server.h"

void OpenXRRenderModelManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_tracker"), &OpenXRRenderModelManager::get_tracker);
	ClassDB::bind_method(D_METHOD("set_tracker", "tracker"), &OpenXRRenderModelManager::set_tracker);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tracker", PROPERTY_HINT_ENUM, "Any,None set,Left Hand,Right Hand"), "set_tracker", "get_tracker");

	ClassDB::bind_method(D_METHOD("get_make_local_to_pose"), &OpenXRRenderModelManager::get_make_local_to_pose);
	ClassDB::bind_method(D_METHOD("set_make_local_to_pose", "make_local_to_pose"), &OpenXRRenderModelManager::set_make_local_to_pose);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "make_local_to_pose", PROPERTY_HINT_ENUM_SUGGESTION, "aim,grip"), "set_make_local_to_pose", "get_make_local_to_pose");

	ADD_SIGNAL(MethodInfo("render_model_added", PropertyInfo(Variant::OBJECT, "render_model", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRRenderModel")));
	ADD_SIGNAL(MethodInfo("render_model_removed", PropertyInfo(Variant::OBJECT, "render_model", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRRenderModel")));

	BIND_ENUM_CONSTANT(RENDER_MODEL_TRACKER_ANY);
	BIND_ENUM_CONSTANT(RENDER_MODEL_TRACKER_NONE_SET);
	BIND_ENUM_CONSTANT(RENDER_MODEL_TRACKER_LEFT_HAND);
	BIND_ENUM_CONSTANT(RENDER_MODEL_TRACKER_RIGHT_HAND);
}

bool OpenXRRenderModelManager::_has_filters() {
	return tracker != 0;
}

void OpenXRRenderModelManager::_update_models() {
	OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
	ERR_FAIL_NULL(render_model_extension);

	// Make a copy of our current models.
	HashMap<RID, Node3D *> org_render_models = HashMap<RID, Node3D *>(render_models);

	// Loop through our interaction data so we add new entries.
	TypedArray<RID> render_model_rids = render_model_extension->render_model_get_all();
	for (const RID rid : render_model_rids) {
		bool filter = false;

		if (tracker != 0) {
			XrPath model_path = render_model_extension->render_model_get_top_level_path(rid);
			if (model_path != xr_path) {
				// ignore this.
				filter = true;
			}
		}

		if (!filter) {
			if (render_models.has(rid)) {
				org_render_models.erase(rid);
			} else {
				// Create our container node before adding our first render model.
				if (container == nullptr) {
					container = memnew(Node3D);
					add_child(container);
				}

				OpenXRRenderModel *render_model = memnew(OpenXRRenderModel);
				render_model->set_render_model(rid);
				container->add_child(render_model);
				render_models[rid] = render_model;

				emit_signal(SNAME("render_model_added"), render_model);
			}
		}
	}

	// Remove models we no longer need.
	for (const KeyValue<RID, Node3D *> &e : org_render_models) {
		// We sent this just before removing.
		emit_signal(SNAME("render_model_removed"), e.value);

		if (container) {
			container->remove_child(e.value);
		}
		e.value->queue_free();
		render_models.erase(e.key);
	}

	is_dirty = false;
}

void OpenXRRenderModelManager::_on_render_model_added(RID p_render_model) {
	if (_has_filters()) {
		// We'll update this in internal process.
		is_dirty = true;
	} else {
		// No filters? Do this right away.
		_update_models();
	}
}

void OpenXRRenderModelManager::_on_render_model_removed(RID p_render_model) {
	if (_has_filters()) {
		// We'll update this in internal process.
		is_dirty = true;
	} else {
		// No filters? Do this right away.
		_update_models();
	}
}

void OpenXRRenderModelManager::_on_render_model_top_level_path_changed(RID p_path) {
	if (_has_filters()) {
		// We'll update this in internal process.
		is_dirty = true;
	}
}

void OpenXRRenderModelManager::_notification(int p_what) {
	// Do not run in editor!
	if (Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	OpenXRRenderModelExtension *render_model_extension = OpenXRRenderModelExtension::get_singleton();
	ERR_FAIL_NULL(render_model_extension);
	if (!render_model_extension->is_active()) {
		return;
	}

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_models();

			render_model_extension->connect(SNAME("render_model_added"), callable_mp(this, &OpenXRRenderModelManager::_on_render_model_added));
			render_model_extension->connect(SNAME("render_model_removed"), callable_mp(this, &OpenXRRenderModelManager::_on_render_model_removed));
			render_model_extension->connect(SNAME("render_model_top_level_path_changed"), callable_mp(this, &OpenXRRenderModelManager::_on_render_model_top_level_path_changed));

			if (_has_filters()) {
				set_process_internal(true);
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			render_model_extension->disconnect(SNAME("render_model_added"), callable_mp(this, &OpenXRRenderModelManager::_on_render_model_added));
			render_model_extension->disconnect(SNAME("render_model_removed"), callable_mp(this, &OpenXRRenderModelManager::_on_render_model_removed));
			render_model_extension->disconnect(SNAME("render_model_top_level_path_changed"), callable_mp(this, &OpenXRRenderModelManager::_on_render_model_top_level_path_changed));

			set_process_internal(false);
			is_dirty = false;
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (is_dirty) {
				_update_models();
			}

			if (positional_tracker.is_valid() && !make_local_to_pose.is_empty() && container) {
				Ref<XRPose> pose = positional_tracker->get_pose(make_local_to_pose);
				if (pose.is_valid()) {
					container->set_transform(pose->get_adjusted_transform().affine_inverse());
				} else {
					container->set_transform(Transform3D());
				}
			}

			if (!_has_filters()) {
				// No need to keep calling this.
				set_process_internal(false);
			}
		}
	}
}

PackedStringArray OpenXRRenderModelManager::get_configuration_warnings() const {
	PackedStringArray warnings;

	XROrigin3D *parent = nullptr;
	if (tracker == 0 || tracker == 1) {
		if (!make_local_to_pose.is_empty()) {
			warnings.push_back("Must specify a tracker to make node local to pose.");
		}

		parent = Object::cast_to<XROrigin3D>(get_parent());
	} else {
		Node *node = get_parent();
		while (!parent && node) {
			parent = Object::cast_to<XROrigin3D>(node);

			node = node->get_parent();
		}
	}
	if (!parent) {
		warnings.push_back("This node must be a child of an XROrigin3D node!");
	}

	if (!GLOBAL_GET("xr/openxr/extensions/render_model")) {
		warnings.push_back("The render model extension is not enabled in project settings!");
	}

	return warnings;
}

void OpenXRRenderModelManager::set_tracker(RenderModelTracker p_tracker) {
	if (tracker != p_tracker) {
		tracker = p_tracker;
		is_dirty = true;

		if (tracker == RENDER_MODEL_TRACKER_ANY || tracker == RENDER_MODEL_TRACKER_NONE_SET) {
			xr_path = XR_NULL_PATH;
		} else if (!Engine::get_singleton()->is_editor_hint()) {
			XRServer *xr_server = XRServer::get_singleton();
			OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
			if (openxr_api && xr_server) {
				String toplevel_path;
				String tracker_name;
				if (tracker == RENDER_MODEL_TRACKER_LEFT_HAND) {
					tracker_name = "left_hand";
					toplevel_path = "/user/hand/left";
				} else if (tracker == RENDER_MODEL_TRACKER_RIGHT_HAND) {
					tracker_name = "right_hand";
					toplevel_path = "/user/hand/right";
				} else {
					ERR_FAIL_MSG("Unsupported tracker value set.");
				}

				positional_tracker = xr_server->get_tracker(tracker_name);
				if (positional_tracker.is_null()) {
					WARN_PRINT("OpenXR: Can't find tracker " + tracker_name);
				}

				xr_path = openxr_api->get_xr_path(toplevel_path);
				if (xr_path == XR_NULL_PATH) {
					WARN_PRINT("OpenXR: Can't find path for " + toplevel_path);
				}
			}
		}

		// Even if we now no longer have filters, we must update at least once.
		set_process_internal(true);
	}
}

OpenXRRenderModelManager::RenderModelTracker OpenXRRenderModelManager::get_tracker() const {
	return tracker;
}

void OpenXRRenderModelManager::set_make_local_to_pose(const String &p_action) {
	if (make_local_to_pose != p_action) {
		make_local_to_pose = p_action;

		if (container) {
			// Reset just in case. It'll be set to the correct transform
			// in our process if required.
			container->set_transform(Transform3D());
		}
	}
}

String OpenXRRenderModelManager::get_make_local_to_pose() const {
	return make_local_to_pose;
}
#endif // MODULE_GLTF_ENABLED
