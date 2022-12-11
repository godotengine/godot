/*************************************************************************/
/*  gltf_camera.cpp                                                      */
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

#include "gltf_camera.h"

#include "scene/3d/camera.h"

void GLTFCamera::_bind_methods() {
	ClassDB::bind_method(D_METHOD("to_node"), &GLTFCamera::to_node);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFCamera::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_perspective"), &GLTFCamera::get_perspective);
	ClassDB::bind_method(D_METHOD("set_perspective", "perspective"), &GLTFCamera::set_perspective);
	ClassDB::bind_method(D_METHOD("get_fov_size"), &GLTFCamera::get_fov_size);
	ClassDB::bind_method(D_METHOD("set_fov_size", "fov_size"), &GLTFCamera::set_fov_size);
	ClassDB::bind_method(D_METHOD("get_size_mag"), &GLTFCamera::get_size_mag);
	ClassDB::bind_method(D_METHOD("set_size_mag", "size_mag"), &GLTFCamera::set_size_mag);
	ClassDB::bind_method(D_METHOD("get_zfar"), &GLTFCamera::get_zfar);
	ClassDB::bind_method(D_METHOD("set_zfar", "zfar"), &GLTFCamera::set_zfar);
	ClassDB::bind_method(D_METHOD("get_znear"), &GLTFCamera::get_znear);
	ClassDB::bind_method(D_METHOD("set_znear", "znear"), &GLTFCamera::set_znear);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "perspective"), "set_perspective", "get_perspective"); // bool
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "fov_size"), "set_fov_size", "get_fov_size"); // float
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "size_mag"), "set_size_mag", "get_size_mag"); // float
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "zfar"), "set_zfar", "get_zfar"); // float
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "znear"), "set_znear", "get_znear"); // float
}

Ref<GLTFCamera> GLTFCamera::from_node(const Camera *p_camera) {
	Ref<GLTFCamera> c;
	c.instance();
	ERR_FAIL_COND_V_MSG(!p_camera, c, "Tried to create a GLTFCamera from a Camera node, but the given node was null.");
	c->set_perspective(p_camera->get_projection() == Camera::Projection::PROJECTION_PERSPECTIVE);
	// GLTF spec (yfov) is in radians, Godot's camera (fov) is in degrees.
	c->set_fov_size(Math::deg2rad(p_camera->get_fov()));
	// GLTF spec (xmag and ymag) is a radius in meters, Godot's camera (size) is a diameter in meters.
	c->set_size_mag(p_camera->get_size() * 0.5f);
	c->set_zfar(p_camera->get_zfar());
	c->set_znear(p_camera->get_znear());
	return c;
}

Camera *GLTFCamera::to_node() const {
	Camera *camera = memnew(Camera);
	camera->set_projection(perspective ? Camera::PROJECTION_PERSPECTIVE : Camera::PROJECTION_ORTHOGONAL);
	// GLTF spec (yfov) is in radians, Godot's camera (fov) is in degrees.
	camera->set_fov(Math::rad2deg(fov));
	// GLTF spec (xmag and ymag) is a radius in meters, Godot's camera (size) is a diameter in meters.
	camera->set_size(size_mag * 2.0f);
	camera->set_znear(znear);
	camera->set_zfar(zfar);
	return camera;
}

Ref<GLTFCamera> GLTFCamera::from_dictionary(const Dictionary p_dictionary) {
	ERR_FAIL_COND_V_MSG(!p_dictionary.has("type"), Ref<GLTFCamera>(), "Failed to parse GLTF camera, missing required field 'type'.");
	Ref<GLTFCamera> camera;
	camera.instance();
	const String &type = p_dictionary["type"];
	if (type == "perspective") {
		camera->set_perspective(true);
		if (p_dictionary.has("perspective")) {
			const Dictionary &persp = p_dictionary["perspective"];
			camera->set_fov_size(persp["yfov"]);
			if (persp.has("zfar")) {
				camera->set_zfar(persp["zfar"]);
			}
			camera->set_znear(persp["znear"]);
		}
	} else if (type == "orthographic") {
		camera->set_perspective(false);
		if (p_dictionary.has("orthographic")) {
			const Dictionary &ortho = p_dictionary["orthographic"];
			camera->set_size_mag(ortho["ymag"]);
			camera->set_zfar(ortho["zfar"]);
			camera->set_znear(ortho["znear"]);
		}
	} else {
		ERR_PRINT("Error parsing GLTF camera: Camera type '" + type + "' is unknown, should be perspective or orthographic.");
	}
	return camera;
}

Dictionary GLTFCamera::to_dictionary() const {
	Dictionary d;
	if (perspective) {
		Dictionary persp;
		persp["yfov"] = fov;
		persp["zfar"] = zfar;
		persp["znear"] = znear;
		d["perspective"] = persp;
		d["type"] = "perspective";
	} else {
		Dictionary ortho;
		ortho["ymag"] = size_mag;
		ortho["xmag"] = size_mag;
		ortho["zfar"] = zfar;
		ortho["znear"] = znear;
		d["orthographic"] = ortho;
		d["type"] = "orthographic";
	}
	return d;
}
