/**************************************************************************/
/*  xr_debugger.cpp                                                       */
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

#ifdef DEBUG_ENABLED

#include "xr_debugger.h"

#include "core/debugger/engine_debugger.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/xr/xr_nodes.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/resources/3d/primitive_meshes.h"

XRDebuggerRuntimeEditor *XRDebugger::get_runtime_editor() {
	return runtime_editor_id.is_valid() ? ObjectDB::get_instance<XRDebuggerRuntimeEditor>(runtime_editor_id) : nullptr;
}

void XRDebugger::initialize() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	ERR_FAIL_NULL(scene_tree);

	XRDebuggerRuntimeEditor *runtime_editor = get_runtime_editor();
	if (runtime_editor) {
		return;
	}

	runtime_editor = memnew(XRDebuggerRuntimeEditor);
	runtime_editor_id = runtime_editor->get_instance_id();

	scene_tree->get_root()->add_child(runtime_editor);
}

void XRDebugger::deinitialize() {
	SceneTree *scene_tree = SceneTree::get_singleton();
	ERR_FAIL_NULL(scene_tree);

	XRDebuggerRuntimeEditor *runtime_editor = get_runtime_editor();
	if (!runtime_editor) {
		return;
	}

	runtime_editor_id = ObjectID();
	runtime_editor->queue_free();
	scene_tree->get_root()->remove_child(runtime_editor);
}

XRDebugger::XRDebugger() {
}

XRDebugger::~XRDebugger() {
	deinitialize();
}

void XRDebuggerRuntimeEditor::_on_controller_button_pressed(const String &p_name, XRController3D *p_controller) {
	// @todo We need to have an action set for OpenXR, and it needs to know to switch to it - not sure how to handle that yet.

	if (p_controller == xr_controller_left && p_name == "menu_button") {
		set_enabled(!enabled);
	} else if (p_controller == xr_controller_right && p_name == "trigger_click") {
		select_pressed = true;
	} else if (p_controller == xr_controller_right && p_name == "grip_click") {
		grab_pressed = true;
		if (selected_node != ObjectID()) {
			Node3D *n = Object::cast_to<Node3D>(ObjectDB::get_instance(selected_node));
			if (n) {
				selected_node_orig_gt = n->get_global_transform();
				grab_orig_t = xr_controller_right->get_transform();
			}
		}
	}
}

void XRDebuggerRuntimeEditor::_on_controller_button_released(const String &p_name, XRController3D *p_controller) {
	// @todo We need to have an action set for OpenXR, and it needs to know to switch to it - not sure how to handle that yet.

	if (p_controller == xr_controller_right && p_name == "grip_click") {
		grab_pressed = false;

		if (selected_node != ObjectID()) {
			Node3D *n = Object::cast_to<Node3D>(ObjectDB::get_instance(selected_node));
			if (n) {
				Node3D *scene_node = _find_nearest_scene_node(n);
				if (scene_node) {
					NodePath scene_path = scene_node->get_owner()->get_path_to(scene_node);
					Transform3D gt = (xr_controller_right->get_transform() * grab_orig_t.affine_inverse()) * selected_node_orig_gt;

					// Make the position relative to the parent.
					Node3D *parent = Object::cast_to<Node3D>(scene_node->get_parent());
					if (parent) {
						gt = parent->get_global_transform().affine_inverse() * gt;
					}

					Array message;
					message.push_back(LiveEditor::get_singleton()->get_live_edit_scene());
					message.push_back(scene_path.operator String());
					message.push_back("transform");
					message.push_back(gt);
					EngineDebugger::get_singleton()->send_message("scene:set_object_property", message);
				}
			}
		}
	}
}

bool XRDebuggerRuntimeEditor::_is_descendent(Node *p_node) {
	Node *parent = p_node->get_parent();
	while (parent) {
		if (parent == this) {
			return true;
		}
		parent = parent->get_parent();
	}
	return false;
}

void XRDebuggerRuntimeEditor::_select_with_ray() {
	Window *root = SceneTree::get_singleton()->get_root();

	Transform3D controller_transform = xr_controller_right->get_global_transform();
	Vector3 pos = controller_transform.origin;
	Vector3 ray = -controller_transform.get_basis().get_column(2);
	// @todo This is 10m ahead - how far should this be?
	Vector3 to = pos + (ray * 10.0);

	Node3D *closest_node = nullptr;
	float closest_distance = 0.0;

#ifndef PHYSICS_3D_DISABLED
	// @todo Try with physics objects first...
#endif

	Vector<ObjectID> items = RS::get_singleton()->instances_cull_ray(pos, to, root->get_world_3d()->get_scenario());
	for (int i = 0; i < items.size(); i++) {
		Object *obj = ObjectDB::get_instance(items[i]);

		GeometryInstance3D *geo_instance = Object::cast_to<GeometryInstance3D>(obj);
		if (geo_instance) {
			if (_is_descendent(geo_instance)) {
				// Discard our child nodes, those are part of the UI.
				continue;
			}

			Ref<TriangleMesh> mesh_collision = geo_instance->generate_triangle_mesh();
			if (mesh_collision.is_valid()) {
				Transform3D gt = geo_instance->get_global_transform();
				Transform3D ai = gt.affine_inverse();
				Vector3 point, normal;
				if (mesh_collision->intersect_ray(ai.xform(pos), ai.basis.xform(ray).normalized(), point, normal)) {
					float distance = pos.distance_to(point);
					if (!closest_node || distance < closest_distance) {
						closest_node = geo_instance;
						closest_distance = distance;
					}
				}
			}
		}
	}

	Vector<Node *> selected_nodes;

	if (closest_node) {
		Node3D *scene_node = _find_nearest_scene_node(closest_node);
		NodePath scene_path;
		if (scene_node) {
			scene_path = scene_node->get_owner()->get_path_to(scene_node);
			closest_node = scene_node;
		}
		selected_node = closest_node->get_instance_id();
		selected_nodes.push_back(closest_node);

		if (!scene_path.is_empty()) {
			Array message2;
			message2.push_back(LiveEditor::get_singleton()->get_live_edit_scene());
			message2.push_back(scene_path.operator String());
			EngineDebugger::get_singleton()->send_message("scene:select_path", message2);
		}

		SceneDebuggerObject obj(selected_node);
		Array arr;
		obj.serialize(arr);

		Array message;
		message.append(arr);
		EngineDebugger::get_singleton()->send_message("remote_objects_selected", message);
	} else {
		selected_node = ObjectID();
		EngineDebugger::get_singleton()->send_message("remote_nothing_selected", Array());
	}

	RuntimeNodeSelect::get_singleton()->_set_selected_nodes(selected_nodes);
}

Node3D *XRDebuggerRuntimeEditor::_find_nearest_scene_node(Node3D *p_node) {
	String live_edit_scene = LiveEditor::get_singleton()->get_live_edit_scene();
	if (live_edit_scene.is_empty()) {
		return nullptr;
	}

	Node *cur_node = p_node;
	while (cur_node) {
		Node *owner = cur_node->get_owner();
		if (owner && owner->get_scene_file_path() == live_edit_scene) {
			return Object::cast_to<Node3D>(cur_node);
		}

		cur_node = cur_node->get_parent();
	}

	return nullptr;
}

void XRDebuggerRuntimeEditor::set_enabled(bool p_enable) {
	if (enabled == p_enable) {
		return;
	}
	enabled = p_enable;

	set_physics_process(enabled);
	xr_controller_left->set_visible(enabled);
	xr_controller_right->set_visible(enabled);

	if (enabled) {
		SceneTree *scene_tree = SceneTree::get_singleton();
		Window *root = scene_tree->get_root();
		XROrigin3D *current_xr_origin = nullptr;

		TypedArray<Node> potential_origins = root->find_children("*", "XROrigin3D", true, false);
		for (const Variant &item : potential_origins) {
			XROrigin3D *potential_origin = Object::cast_to<XROrigin3D>(item);
			if (!potential_origin || potential_origin == this) {
				continue;
			}

			if (potential_origin->is_current()) {
				current_xr_origin = potential_origin;
				break;
			}
		}

		if (current_xr_origin) {
			original_xr_origin_id = current_xr_origin->get_instance_id();
			set_global_transform(current_xr_origin->get_global_transform());

			XRCamera3D *current_xr_camera = nullptr;
			TypedArray<XRCamera3D> potential_cameras = current_xr_origin->find_children("*", "XRCamera3D", true, false);
			for (const Variant &item : potential_cameras) {
				XRCamera3D *potential_camera = Object::cast_to<XRCamera3D>(item);
				if (!potential_camera || _is_descendent(potential_camera)) {
					continue;
				}

				if (potential_camera->is_current()) {
					current_xr_camera = potential_camera;
					break;
				}
			}
			if (current_xr_camera) {
				original_xr_camera_id = current_xr_camera->get_instance_id();
			}

			TypedArray<XRController3D> controllers = current_xr_origin->find_children("*", "XRController3D", true, false);
			for (const Variant &item : controllers) {
				XRController3D *controller = Object::cast_to<XRController3D>(item);
				if (!controller || _is_descendent(controller)) {
					continue;
				}

				original_xr_controllers.push_back({
						controller->get_tracker(), // tracker
						controller->get_instance_id(), // controller_id
				});

				controller->set_tracker("/disabled/by/xr/debugger");
			}
		}

		set_current(true);
		xr_camera->set_current(true);
	} else {
		XROrigin3D *original_xr_origin = ObjectDB::get_instance<XROrigin3D>(original_xr_origin_id);
		if (original_xr_origin) {
			original_xr_origin->set_current(true);
		} else {
			set_current(false);
		}
		original_xr_origin_id = ObjectID();

		XRCamera3D *original_xr_camera = ObjectDB::get_instance<XRCamera3D>(original_xr_camera_id);
		if (original_xr_camera) {
			original_xr_camera->set_current(true);
		} else {
			xr_camera->set_current(false);
		}
		original_xr_camera_id = ObjectID();

		for (const ControllerInfo &ci : original_xr_controllers) {
			XRController3D *xr_controller = ObjectDB::get_instance<XRController3D>(ci.controller_id);
			if (xr_controller) {
				xr_controller->set_tracker(ci.tracker);
			}
		}
		original_xr_controllers.clear();
	}
}

void XRDebuggerRuntimeEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Ensure that we don't take over from the user's XROrigin3D.
			callable_mp(static_cast<XROrigin3D *>(this), &XROrigin3D::set_current).bind(false);
		} break;

		case NOTIFICATION_PHYSICS_PROCESS: {
			if (enabled) {
				if (grab_pressed) {
					if (selected_node != ObjectID()) {
						Node3D *n = Object::cast_to<Node3D>(ObjectDB::get_instance(selected_node));
						if (n) {
							Transform3D gt = (xr_controller_right->get_transform() * grab_orig_t.affine_inverse()) * selected_node_orig_gt;
							n->set_global_transform(gt);
						}
					}
				} else if (select_pressed) {
					select_pressed = false;
					_select_with_ray();
				}
			}
		} break;
	}
}

XRDebuggerRuntimeEditor::XRDebuggerRuntimeEditor() {
	set_name("XRDebuggerRuntimeEditor");
	set_current(false);
	set_process_mode(Node::PROCESS_MODE_ALWAYS);

	xr_camera = memnew(XRCamera3D);
	xr_camera->set_current(false);
	add_child(xr_camera);

	xr_controller_left = memnew(XRController3D);
	xr_controller_left->set_tracker("left_hand");
	xr_controller_left->set_visible(false);
	xr_controller_left->connect("button_pressed", callable_mp(this, &XRDebuggerRuntimeEditor::_on_controller_button_pressed).bind(xr_controller_left));
	xr_controller_left->connect("button_released", callable_mp(this, &XRDebuggerRuntimeEditor::_on_controller_button_released).bind(xr_controller_left));
	add_child(xr_controller_left);

	Ref<BoxMesh> box_mesh;
	box_mesh.instantiate();
	box_mesh->set_size(Vector3(0.4, 0.4, 0.05));

	MeshInstance3D *left_mesh_instance = memnew(MeshInstance3D);
	left_mesh_instance->set_mesh(box_mesh);
	xr_controller_left->add_child(left_mesh_instance);

	xr_controller_right = memnew(XRController3D);
	xr_controller_right->set_tracker("right_hand");
	xr_controller_right->set_visible(false);
	xr_controller_right->connect("button_pressed", callable_mp(this, &XRDebuggerRuntimeEditor::_on_controller_button_pressed).bind(xr_controller_right));
	xr_controller_right->connect("button_released", callable_mp(this, &XRDebuggerRuntimeEditor::_on_controller_button_released).bind(xr_controller_right));
	add_child(xr_controller_right);

	Ref<BoxMesh> ray_mesh;
	ray_mesh.instantiate();
	ray_mesh->set_size(Vector3(0.01, 0.01, 10.0));

	MeshInstance3D *right_mesh_instance = memnew(MeshInstance3D);
	right_mesh_instance->set_mesh(ray_mesh);
	right_mesh_instance->set_position(Vector3(0.0, 0.0, -5.0));
	xr_controller_right->add_child(right_mesh_instance);
}

XRDebuggerRuntimeEditor::~XRDebuggerRuntimeEditor() {
}

#endif
