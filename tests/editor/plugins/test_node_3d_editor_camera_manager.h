/**************************************************************************/
/*  test_node_3d_editor_camera_manager.h                                  */
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

#ifndef TEST_NODE_3D_EDITOR_CAMERA_MANAGER_H
#define TEST_NODE_3D_EDITOR_CAMERA_MANAGER_H

#include "editor/plugins/node_3d_editor_camera_manager.h"

#include "editor/editor_settings.h"
#include "editor/plugins/node_3d_editor_camera_cursor.h"
#include "scene/3d/camera_3d.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"
#include "tests/test_macros.h"

namespace TestNode3DEditorCameraManager {

	TEST_CASE("[TestNode3DEditorCameraManager][SceneTree] Camera manager") {

		Node3DEditorCameraManager* camera_manager = memnew(Node3DEditorCameraManager);
		EditorSettings* editor_settings = memnew(EditorSettings);
		Window* root = SceneTree::get_singleton()->get_root();
		Camera3D* editor_camera = memnew(Camera3D);
		Camera3D* previewing_camera = memnew(Camera3D);
		Camera3D* cinematic_camera = memnew(Camera3D);
		Node3D* some_node = memnew(Node3D);
		Node3D* some_another_node = memnew(Node3D);
		camera_manager->setup(editor_camera, root, root, editor_settings);
		root->add_child(editor_camera);
		root->add_child(cinematic_camera);
		root->add_child(previewing_camera);
		root->add_child(some_node);
		root->add_child(some_another_node);
		editor_settings->set("editors/3d/freelook/freelook_inertia", 1.0);
		editor_settings->set("editors/3d/navigation_feel/orbit_inertia", 1.0);
		editor_settings->set("editors/3d/navigation_feel/translation_inertia", 1.0);
		editor_settings->set("editors/3d/navigation_feel/zoom_inertia", 1.0);

		SUBCASE("[TestNode3DEditorCameraManager] Camera settings") {
			camera_manager->set_camera_settings(0.5, 1.0, 100.0);
			CHECK(editor_camera->get_fov() == 0.5);
			CHECK(editor_camera->get_near() == 1.0);
			CHECK(editor_camera->get_far() == 100.0);
		}

		SUBCASE("[TestNode3DEditorCameraManager] Reset") {

			SUBCASE("Should reset cursor") {
				camera_manager->navigation_move(10.0, 20.0, 50.0);
				camera_manager->navigation_orbit(Vector2(0.5, 0.8));
				camera_manager->set_fov_scale(10.0);
				camera_manager->reset();
				CHECK(camera_manager->get_cursor().get_current_values() == Node3DEditorCameraCursor().get_current_values());
				CHECK(camera_manager->get_cursor().get_target_values() == Node3DEditorCameraCursor().get_target_values());
			}

			SUBCASE("Should stop piloting") {
				camera_manager->pilot(some_node);
				camera_manager->reset();
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
			}

			SUBCASE("Should stop previewing camera") {
				camera_manager->preview_camera(previewing_camera);
				camera_manager->reset();
				CHECK(camera_manager->get_previewing_camera() == nullptr);
			}

			SUBCASE("Should leave cinematic preview mode") {
				camera_manager->set_cinematic_preview_mode(true);
				camera_manager->reset();
				CHECK(!camera_manager->is_in_cinematic_preview_mode());
			}

			SUBCASE("Should set camera to perspective") {
				camera_manager->set_orthogonal(true);
				camera_manager->reset();
				CHECK(!camera_manager->is_orthogonal());
			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Set cursor state") {
			camera_manager->set_cursor_state(Vector3(1.0, 2.0, 3.0), 0.5, 1.5, 10.0);
			CHECK(camera_manager->get_cursor().get_target_values().position == Vector3(1.0, 2.0, 3.0));
			CHECK(camera_manager->get_cursor().get_target_values().x_rot == 0.5);
			CHECK(camera_manager->get_cursor().get_target_values().y_rot == 1.5);
			CHECK(camera_manager->get_cursor().get_target_values().distance == 10.0);
			CHECK(camera_manager->get_cursor().get_target_values() == camera_manager->get_cursor().get_current_values());
		}

		SUBCASE("[TestNode3DEditorCameraManager] Get current camera") {
			cinematic_camera->make_current();

			SUBCASE("Editor's camera") {
				CHECK(camera_manager->get_current_camera() == editor_camera);
			}

			SUBCASE("Previewing camera") {
				camera_manager->preview_camera(previewing_camera);
				CHECK(camera_manager->get_current_camera() == previewing_camera);
			}

			SUBCASE("Cinematic previewing camera") {
				camera_manager->set_cinematic_preview_mode(true);
				CHECK(camera_manager->get_current_camera() == cinematic_camera);
			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Get previewing or cinematic camera") {
			cinematic_camera->make_current();

			SUBCASE("Previewing camera") {
				camera_manager->preview_camera(previewing_camera);
				CHECK(camera_manager->get_previewing_or_cinematic_camera() == previewing_camera);
			}

			SUBCASE("Cinematic previewing camera") {
				camera_manager->set_cinematic_preview_mode(true);
				CHECK(camera_manager->get_previewing_or_cinematic_camera() == cinematic_camera);
			}

			SUBCASE("No previewing camera") {
				CHECK(camera_manager->get_previewing_or_cinematic_camera() == nullptr);
			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Pilot") {

			SUBCASE("Should return the node being piloted") {
				camera_manager->pilot(some_node);
				CHECK(camera_manager->get_node_being_piloted() == some_node);
			}

			SUBCASE("Should move the node when move the camera") {
				some_node->set_global_position(Vector3(100.0, 200.0, 300.0));
				camera_manager->pilot(some_node);
				camera_manager->navigation_pan(Vector2(10.0, 20.0), 1.0);
				camera_manager->navigation_look(Vector2(5.0, -5.0), 1.0);
				camera_manager->update(100.0); // force the interpolation to end
				CHECK(some_node->get_global_position().is_equal_approx(Vector3(90.0, 220.0, 300.0)));
				CHECK(some_node->get_global_rotation_degrees() != Vector3(0.0, 0.0, 0.0));
			}

			SUBCASE("Should move the node in world space") {
				Node3D* child_node = memnew(Node3D);
				some_node->add_child(child_node);
				some_node->set_global_position(Vector3(100.0, 200.0, 300.0));
				child_node->set_position(Vector3(1000.0, 1000.0, 1000.0));
				camera_manager->pilot(child_node);
				camera_manager->navigation_pan(Vector2(10.0, 20.0), 1.0);
				camera_manager->update(100.0); // force the interpolation to end
				CHECK(child_node->get_global_position().is_equal_approx(Vector3(1090.0, 1220.0, 1300.0)));
				memdelete(child_node);
			}

			SUBCASE("Should not mess with objects scale") {
				some_node->set_scale(Vector3(2.0, 2.0, 2.0));
				camera_manager->pilot(some_node);
				camera_manager->navigation_pan(Vector2(10.0, 20.0), 1.0);
				camera_manager->navigation_look(Vector2(5.0, -5.0), 1.0);
				camera_manager->update(100.0); // force the interpolation to end
				CHECK(some_node->get_scale() == Vector3(2.0, 2.0, 2.0));
				CHECK(editor_camera->get_scale() == Vector3(1.0, 1.0, 1.0));
			}

			SUBCASE("Pilot while in cinematic preview mode should turn it off") {
				camera_manager->set_cinematic_preview_mode(true);
				camera_manager->pilot(some_node);
				CHECK(!camera_manager->is_in_cinematic_preview_mode());
			}

			SUBCASE("Pilot while in camera preview mode should turn it off") {
				camera_manager->preview_camera(previewing_camera);
				camera_manager->pilot(some_node);
				CHECK(camera_manager->get_previewing_camera() == nullptr);
			}

			SUBCASE("Pilot while in camera preview mode should keep the preview if the node is the previewing camera") {
				camera_manager->preview_camera(previewing_camera);
				camera_manager->pilot(previewing_camera);
				CHECK(camera_manager->get_previewing_camera() == previewing_camera);
				CHECK(camera_manager->get_node_being_piloted() == previewing_camera);
			}

			SUBCASE("Should change to perspective if in orthogonal mode") {
				camera_manager->set_orthogonal(true);
				camera_manager->pilot(some_node);
				CHECK(!camera_manager->is_orthogonal());
			}

			SUBCASE("Should change to perspective if piloting a non-orthogonal camera") {
				camera_manager->set_orthogonal(true);
				camera_manager->pilot(previewing_camera);
				CHECK(!camera_manager->is_orthogonal());
			}

			SUBCASE("Should change to orthogonal if piloting an orthogonal camera") {
				previewing_camera->set_orthogonal(1.0, 10.0, 100.0);
				camera_manager->pilot(previewing_camera);
				CHECK(camera_manager->is_orthogonal());
			}

			SUBCASE("Should update the camera and cursor to the node's transform") {
				some_node->set_global_position(Vector3(100.0, 200.0, 300.0));
				camera_manager->pilot(some_node);
				CHECK(editor_camera->get_global_position() == Vector3(100.0, 200.0, 300.0));
				CHECK(editor_camera->get_global_rotation_degrees() == Vector3(0.0, 0.0, 0.0));
				CHECK(camera_manager->get_cursor().get_current_values().eye_position == Vector3(100.0, 200.0, 300.0));
				CHECK(camera_manager->get_cursor().get_current_values().position == Vector3(100.0, 200.0, 296.0));
				CHECK(camera_manager->get_cursor().get_current_values().x_rot == 0.0);
				CHECK(camera_manager->get_cursor().get_current_values().y_rot == 0.0);
			}

			SUBCASE("Should stop cursor interpolation") {
				camera_manager->navigation_pan(Vector2(10.0, 20.0), 1.0);
				camera_manager->navigation_look(Vector2(5.0, -5.0), 1.0);
				some_node->set_global_position(Vector3(100.0, 200.0, 300.0));
				camera_manager->pilot(some_node);
				CHECK(camera_manager->get_cursor().get_current_values() == camera_manager->get_cursor().get_target_values());
			}

			SUBCASE("Should stop piloting if the node is destroyed") {
				Node3D* node_to_be_deleted = memnew(Node3D);
				root->add_child(node_to_be_deleted);
				camera_manager->pilot(node_to_be_deleted);
				memdelete(node_to_be_deleted);
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
			}

			SUBCASE("Should stop piloting if the node changes the parenting") {
				Node3D* child_node = memnew(Node3D);
				some_node->add_child(child_node);
				camera_manager->pilot(child_node);
				some_node->remove_child(child_node);
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
				memdelete(child_node);
			}

			SUBCASE("Should update camera and cursor if the node moves") {

			}

			SUBCASE("Should emit signal 'camera_mode_changed'") {
				SIGNAL_WATCH(camera_manager, "camera_mode_changed");
				camera_manager->pilot(some_node);
				SIGNAL_CHECK_TRUE("camera_mode_changed");
				SIGNAL_UNWATCH(camera_manager, "camera_mode_changed");
			}

			SUBCASE("Should do nothing if node is null") {

				SUBCASE("Should not emit signal") {
					SIGNAL_WATCH(camera_manager, "camera_mode_changed");
					camera_manager->pilot(nullptr);
					SIGNAL_CHECK_FALSE("camera_mode_changed");
					SIGNAL_UNWATCH(camera_manager, "camera_mode_changed");
				}

				SUBCASE("Should not stop previewing") {
					camera_manager->preview_camera(previewing_camera);
					camera_manager->pilot(nullptr);
					CHECK(camera_manager->get_previewing_camera() == previewing_camera);
				}

				SUBCASE("Should not leave cinematic preview mode") {
					camera_manager->set_cinematic_preview_mode(true);
					camera_manager->pilot(nullptr);
					CHECK(camera_manager->is_in_cinematic_preview_mode());
				}

				SUBCASE("Should not change to perspective") {
					camera_manager->set_orthogonal(true);
					camera_manager->pilot(nullptr);
					CHECK(camera_manager->is_orthogonal());
				}

				SUBCASE("Should not stop piloting other node") {
					camera_manager->pilot(some_node);
					camera_manager->pilot(nullptr);
					CHECK(camera_manager->get_node_being_piloted() == some_node);
				}
			}

			SUBCASE("Should do nothing if node is already being in pilot mode") {

				SUBCASE("Should not emit signal") {
					camera_manager->pilot(some_node);
					SIGNAL_WATCH(camera_manager, "camera_mode_changed");
					camera_manager->pilot(some_node);
					SIGNAL_CHECK_FALSE("camera_mode_changed");
					SIGNAL_UNWATCH(camera_manager, "camera_mode_changed");
				}

				SUBCASE("Should keep piloting") {
					camera_manager->pilot(some_node);
					camera_manager->pilot(some_node);
					CHECK(camera_manager->get_node_being_piloted() == some_node);
				}
			}

			SUBCASE("Should change the pilot mode from one node to another") {

				SUBCASE("Should return the new node being piloted") {
					camera_manager->pilot(some_another_node);
					camera_manager->pilot(some_node);
					CHECK(camera_manager->get_node_being_piloted() == some_node);
				}

				SUBCASE("Should move the new node when move the camera") {
					camera_manager->pilot(some_another_node);
					camera_manager->pilot(some_node);
					camera_manager->navigation_pan(Vector2(10.0, 20.0), 1.0);
					camera_manager->update(100.0); // force the interpolation to end
					CHECK(some_node->get_global_position().is_equal_approx(Vector3(-10.0, 20.0, 0.0)));
					CHECK(some_another_node->get_global_position() == Vector3(0.0, 0.0, 0.0));
				}

				SUBCASE("Should turn camera preview off if it was piloting the preview camera before") {
					camera_manager->preview_camera(previewing_camera);
					camera_manager->pilot(previewing_camera);
					CHECK(camera_manager->get_previewing_camera() == previewing_camera);
					camera_manager->pilot(some_node);
					CHECK(camera_manager->get_previewing_camera() == nullptr);
				}

				SUBCASE("Should update the camera and cursor to the new node's transform") {
					camera_manager->pilot(some_another_node);
					some_node->set_global_position(Vector3(100.0, 200.0, 300.0));
					camera_manager->pilot(some_node);
					CHECK(editor_camera->get_global_position() == Vector3(100.0, 200.0, 300.0));
				}

				SUBCASE("Should emit signal 'camera_mode_changed'") {
					camera_manager->pilot(some_another_node);
					SIGNAL_WATCH(camera_manager, "camera_mode_changed");
					camera_manager->pilot(some_node);
					SIGNAL_CHECK_TRUE("camera_mode_changed");
					SIGNAL_UNWATCH(camera_manager, "camera_mode_changed");
				}
			}

			SUBCASE("Undo / redo") {

			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Stop piloting") {

			SUBCASE("The node being piloted should be null") {
				camera_manager->pilot(some_node);
				camera_manager->stop_piloting();
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
			}

			SUBCASE("Should stop moving the node with the camera") {
				some_node->set_global_position(Vector3(100.0, 200.0, 300.0));
				camera_manager->pilot(some_node);
				camera_manager->stop_piloting();
				camera_manager->navigation_pan(Vector2(10.0, 20.0), 1.0);
				camera_manager->navigation_look(Vector2(5.0, -5.0), 1.0);
				camera_manager->update(100.0); // force the interpolation to end
				CHECK(some_node->get_global_position() == Vector3(100.0, 200.0, 300.0));
				CHECK(some_node->get_global_rotation_degrees() == Vector3(0.0, 0.0, 0.0));
			}

			SUBCASE("Should not leave pilot mode if a previously piloted node is deleted") {
				Node3D* node_to_be_deleted = memnew(Node3D);
				root->add_child(node_to_be_deleted);
				camera_manager->pilot(node_to_be_deleted);
				camera_manager->stop_piloting();
				camera_manager->pilot(some_node);
				memdelete(node_to_be_deleted);
				CHECK(camera_manager->get_node_being_piloted() == some_node);
			}

			SUBCASE("Should emit signal 'camera_mode_changed'") {
				camera_manager->pilot(some_node);
				SIGNAL_WATCH(camera_manager, "camera_mode_changed");
				camera_manager->stop_piloting();
				SIGNAL_CHECK_TRUE("camera_mode_changed");
				SIGNAL_UNWATCH(camera_manager, "camera_mode_changed");
			}

			SUBCASE("Should not emit signal 'camera_mode_changed' if not piloting") {
				SIGNAL_WATCH(camera_manager, "camera_mode_changed");
				camera_manager->stop_piloting();
				SIGNAL_CHECK_FALSE("camera_mode_changed");
				SIGNAL_UNWATCH(camera_manager, "camera_mode_changed");
			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Allow pilot previewing camera") {

			SUBCASE("Turn on while previewing a camera") {
				camera_manager->preview_camera(previewing_camera);
				camera_manager->set_allow_pilot_previewing_camera(true);
				CHECK(camera_manager->get_previewing_camera() == previewing_camera);
				CHECK(camera_manager->get_node_being_piloted() == previewing_camera);
			}

			SUBCASE("Turn off while previewing a camera") {
				camera_manager->preview_camera(previewing_camera);
				camera_manager->set_allow_pilot_previewing_camera(true);
				camera_manager->set_allow_pilot_previewing_camera(false);
				CHECK(camera_manager->get_previewing_camera() == previewing_camera);
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
			}

			SUBCASE("Turn on without previewing a camera") {
				camera_manager->set_allow_pilot_previewing_camera(true);
				CHECK(camera_manager->get_previewing_camera() == nullptr);
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
			}

			SUBCASE("Turn off while piloting another node") {
				camera_manager->set_allow_pilot_previewing_camera(true);
				camera_manager->pilot(some_node);
				camera_manager->set_allow_pilot_previewing_camera(false);
				CHECK(camera_manager->get_previewing_camera() == nullptr);
				CHECK(camera_manager->get_node_being_piloted() == some_node);
			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Preview camera") {

			SUBCASE("Should set the camera as viewport's camera") {

			}

			SUBCASE("Should stop pilot mode") {

			}

			SUBCASE("Should pilot camera if set_allow_pilot_previewing_camera was set to true before") {
				camera_manager->set_allow_pilot_previewing_camera(true);
				camera_manager->preview_camera(previewing_camera);
				CHECK(camera_manager->get_previewing_camera() == previewing_camera);
				CHECK(camera_manager->get_node_being_piloted() == previewing_camera);
			}

			SUBCASE("Should not pilot camera if set_allow_pilot_previewing_camera was not set to true before") {
				camera_manager->set_allow_pilot_previewing_camera(false);
				camera_manager->preview_camera(previewing_camera);
				CHECK(camera_manager->get_previewing_camera() == previewing_camera);
				CHECK(camera_manager->get_node_being_piloted() == nullptr);
			}

			SUBCASE("Should keep piloting the camera if it already was before preview") {

			}

			SUBCASE("Should leave previewing mode when it is deleted") {

			}

			SUBCASE("Should do nothing if camera is null") {

			}

			SUBCASE("Should do nothing if already previewing the same camera") {

			}

			SUBCASE("Should do nothing if in cinematic previewing mode") {

			}

			SUBCASE("Should replace the camera if was previewing another one") {

			}

			SUBCASE("Should emit signal 'camera_mode_changed'") {

			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Stop previewing camera") {

			SUBCASE("Should set the editor's camera in the viewport") {

			}

			SUBCASE("Should stop pilot mode") {

			}

			SUBCASE("Should not stop previewing camera if a previously previewed camera is deleted") {

			}

			SUBCASE("Should emit signal 'camera_mode_changed'") {

			}

			SUBCASE("Should do nothing if not previewing any camera") {

			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Set cinematic preview mode") {

			SUBCASE("Should stop pilot and normal camera preview modes if setting to true") {

			}

			SUBCASE("Should not stop pilot and normal camera preview modes if setting to false") {

			}

			SUBCASE("Should set the editor's camera in the viewport when leaving the cinematic preview mode") {

			}

			SUBCASE("Should do nothing if set to true and it already is true") {

			}

			SUBCASE("Should do nothing if set to false and it already is false") {

			}

			SUBCASE("Should emit signal 'camera_mode_changed'") {

			}
		}

		SUBCASE("[TestNode3DEditorCameraManager] Toggle orthogonal / perspective") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Set FOV scale") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Navigation move") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Navigation freelook move") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Navigation look") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Navigation pan") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Navigation zoom") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Navigation orbit") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Orbit view") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Change view") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Center to origin") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Focus selection") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Update") {
		}

		SUBCASE("[TestNode3DEditorCameraManager] Update camera") {
		}

		memdelete(some_node);
		memdelete(some_another_node);
		memdelete(cinematic_camera);
		memdelete(previewing_camera);
		memdelete(editor_camera);
		memdelete(camera_manager);
		memdelete(editor_settings);
	}

} // namespace TestNode3DEditorCameraManager

#endif // TEST_NODE_3D_EDITOR_CAMERA_MANAGER_H
