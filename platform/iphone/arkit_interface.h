/*************************************************************************/
/*  arkit_interface.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifdef ARKIT_ENABLED

#ifndef ARKIT_INTERFACE_H
#define ARKIT_INTERFACE_H

#include "servers/arvr/arvr_interface.h"
#include "servers/arvr/arvr_positional_tracker.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>

	ARKit interface between iPhone and Godot
*/

// forward declaration for some needed objects
class ARKitShader;

class ARKitInterface : public ARVRInterface {
	GDCLASS(ARKitInterface, ARVRInterface);

private:
	bool initialized;
	bool session_start;
	bool plane_detection_is_enabled;
	bool light_estimation_is_enabled;
	real_t ambient_intensity;
	real_t ambient_color_temperature;

	Transform transform;
	CameraMatrix projection;
	float eye_height, z_near, z_far;

	int image_width;
	int image_height;

	struct anchor_map {
		ARVRPositionalTracker *tracker;
		unsigned char uuid[16];
	};

	///@TODO should use memory vector object from Godot?
	unsigned int num_anchors;
	unsigned int max_anchors;
	anchor_map *anchors;
	ARVRPositionalTracker *get_anchor_for_uuid(const unsigned char *p_uuid);
	void remove_anchor(const unsigned int p_idx);
	void remove_all_anchors();

protected:
	static void _bind_methods();

public:
	void start_session();
	void stop_session();

	bool get_anchor_detection_is_enabled() const;
	void set_anchor_detection_is_enabled(bool p_enable);

	bool get_light_estimation_is_enabled() const;
	void set_light_estimation_is_enabled(bool p_enable);

	real_t get_ambient_intensity() const;
	real_t get_ambient_color_temperature() const;

	/* while Godot has its own raycast logic this takes ARKits camera into account and hits on any ARAnchor */
	Array raycast(Vector2 p_screen_coord);

	virtual StringName get_name() const;
	virtual int get_capabilities() const;

	virtual bool is_initialized();
	virtual bool initialize();
	virtual void uninitialize();

	virtual Size2 get_render_targetsize();
	virtual bool is_stereo();
	virtual Transform get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform);
	virtual CameraMatrix get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far);
	virtual void commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect);

	virtual void process();

	ARKitInterface();
	~ARKitInterface();
};

#endif /* !ARKIT_INTERFACE_H */

#endif /* ARKIT_ENABLED */
