/**************************************************************************/
/*  xr_debugger.h                                                         */
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

#pragma once

#ifdef DEBUG_ENABLED

#include "core/templates/local_vector.h"
#include "scene/3d/xr/xr_nodes.h"

class SceneDebugger;
class XRDebuggerRuntimeEditor;

class XRDebugger : public Object {
	GDSOFTCLASS(XRDebugger, Object);

	friend class SceneDebugger;

	inline static XRDebugger *singleton = nullptr;

	ObjectID runtime_editor_id;

	XRDebuggerRuntimeEditor *get_runtime_editor();

public:
	static XRDebugger *get_singleton() { return singleton; }

	void initialize();
	void deinitialize();

	XRDebugger();
	~XRDebugger();
};

class XRDebuggerRuntimeEditor : public XROrigin3D {
	GDCLASS(XRDebuggerRuntimeEditor, XROrigin3D);

	bool enabled = false;

	ObjectID original_xr_origin_id;
	ObjectID original_xr_camera_id;

	struct ControllerInfo {
		StringName tracker;
		ObjectID controller_id;
	};
	LocalVector<ControllerInfo> original_xr_controllers;

	XRCamera3D *xr_camera = nullptr;
	XRController3D *xr_controller_left = nullptr;
	XRController3D *xr_controller_right = nullptr;

	bool select_pressed = false;
	bool grab_pressed = false;
	ObjectID selected_node;
	Transform3D selected_node_orig_gt;
	Transform3D grab_orig_t;

	void _on_controller_button_pressed(const String &p_name, XRController3D *p_controller);
	void _on_controller_button_released(const String &p_name, XRController3D *p_controller);
	bool _is_descendent(Node *p_node);
	void _select_with_ray();

	void set_enabled(bool p_enable);

	Node3D *_find_nearest_scene_node(Node3D *p_node);

protected:
	void _notification(int p_what);

public:
	XRDebuggerRuntimeEditor();
	~XRDebuggerRuntimeEditor();
};

#endif
