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
#include "tests/test_macros.h"

namespace TestNode3DEditorCameraManager {

TEST_CASE("[TestNode3DEditorCameraManager] Camera settings") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Reset") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Set cursor state") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Get current camera") {
	SUBCASE("Editor's camera") {

	}

	SUBCASE("Previewing camera") {

	}

	SUBCASE("Cinematic previewing camera") {

	}
}

TEST_CASE("[TestNode3DEditorCameraManager] Get previewing or cinematic camera") {
	SUBCASE("Previewing camera") {

	}

	SUBCASE("Cinematic previewing camera") {

	}

	SUBCASE("No previewing camera") {

	}
}

TEST_CASE("[TestNode3DEditorCameraManager] Pilot selection") {

	SUBCASE("Should do nothing of no selection") {

	}

	SUBCASE("Should do nothing if more than one selection") {

	}

	SUBCASE("Should only pilot if it is the only selected node") {

	}
}

TEST_CASE("[TestNode3DEditorCameraManager] Pilot") {

	SUBCASE("Should move the node when move the camera") {

	}

	SUBCASE("Pilot while in cinematic preview mode should turn it off") {

	}

	SUBCASE("Pilot while in camera preview mode should turn it off") {

	}

	SUBCASE("Pilot while in camera preview mode should keep the preview if the node is the previewing camera") {

	}

	SUBCASE("Should change to perspective if in orthogonal mode") {

	}

	SUBCASE("Should change to perspective if piloting a non-orthogonal camera") {

	}

	SUBCASE("Should change to orthogonal if piloting an orthogonal camera") {

	}

	SUBCASE("Should update the camera and cursor to the node's transform") {

	}

	SUBCASE("Should stop cursor interpolation") {

	}

	SUBCASE("Should stop piloting if the node is destroyed") {

	}

	SUBCASE("Should stop piloting if the node changes the parenting") {

	}

	SUBCASE("Should update camera and cursor if the node moves") {

	}

	SUBCASE("Should emit signal 'camera_mode_changed'") {

	}

	SUBCASE("Should do nothing if node is null") {

	}

	SUBCASE("Should do nothing if node is already being in pilot mode") {

	}

	SUBCASE("Should change the pilot mode from one node to another") {

	}
}

TEST_CASE("[TestNode3DEditorCameraManager] Stop piloting") {

	SUBCASE("Should stop moving the node with the camera") {

	}

	SUBCASE("Should stop moving the node with the camera") {

	}

	SUBCASE("Should not leave pilot move if a previously piloted node is deleted") {

	}

	SUBCASE("Should emit signal 'camera_mode_changed'") {

	}

	SUBCASE("Should do nothing if not in pilot mode") {

	}
}

TEST_CASE("[TestNode3DEditorCameraManager] Allow pilot previewing camera") {

	SUBCASE("Turn on while previewing a camera") {

	}

	SUBCASE("Turn off while previewing a camera") {

	}

	SUBCASE("Turn on without previewing a camera") {

	}

	SUBCASE("Turn off while piloting another node") {

	}
}

TEST_CASE("[TestNode3DEditorCameraManager] Preview camera") {

	SUBCASE("Should set the camera as viewport's camera") {

	}

	SUBCASE("Should stop pilot mode") {

	}

	SUBCASE("Should pilot camera if set_allow_pilot_previewing_camera was called before") {

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

TEST_CASE("[TestNode3DEditorCameraManager] Stop previewing camera") {

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

TEST_CASE("[TestNode3DEditorCameraManager] Set cinematic preview mode") {

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

TEST_CASE("[TestNode3DEditorCameraManager] Toggle orthogonal / perspective") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Set FOV scale") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Navigation move") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Navigation freelook move") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Navigation look") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Navigation pan") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Navigation zoom") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Navigation orbit") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Orbit view") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Change view") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Center to origin") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Focus selection") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Update") {
}

TEST_CASE("[TestNode3DEditorCameraManager] Update camera") {
}

} // namespace TestNode3DEditorCameraManager

#endif // TEST_NODE_3D_EDITOR_CAMERA_MANAGER_H
