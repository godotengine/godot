/*************************************************************************/
/*  arkit_interface.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ARKIT_INTERFACE_H
#define ARKIT_INTERFACE_H

#include "servers/camera/camera_feed.h"
#include "servers/xr/xr_interface.h"
#include "servers/xr/xr_positional_tracker.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>

	ARKit interface between iPhone and Godot
*/

// forward declaration for some needed objects
class ARKitShader;

#ifdef __OBJC__

typedef ARAnchor GodotARAnchor;

#else

typedef void GodotARAnchor;
#endif

class ARKitInterface : public XRInterface {
	GDCLASS(ARKitInterface, XRInterface);

private:
	bool initialized;
	bool session_was_started;
	bool plane_detection_is_enabled;
	bool light_estimation_is_enabled;
	real_t ambient_intensity;
	real_t ambient_color_temperature;

	Transform transform;
	CameraMatrix projection;
	float eye_height, z_near, z_far;

	Ref<CameraFeed> feed;
	size_t image_width[2];
	size_t image_height[2];
	Vector<uint8_t> img_data[2];

	struct anchor_map {
		XRPositionalTracker *tracker;
		unsigned char uuid[16];
	};

	///@TODO should use memory map object from Godot?
	unsigned int num_anchors;
	unsigned int max_anchors;
	anchor_map *anchors;
	XRPositionalTracker *get_anchor_for_uuid(const unsigned char *p_uuid);
	void remove_anchor_for_uuid(const unsigned char *p_uuid);
	void remove_all_anchors();

protected:
	static void _bind_methods();

public:
	void start_session();
	void stop_session();

	bool get_anchor_detection_is_enabled() const override;
	void set_anchor_detection_is_enabled(bool p_enable) override;
	virtual int get_camera_feed_id() override;

	bool get_light_estimation_is_enabled() const;
	void set_light_estimation_is_enabled(bool p_enable);

	real_t get_ambient_intensity() const;
	real_t get_ambient_color_temperature() const;

	/* while Godot has its own raycast logic this takes ARKits camera into account and hits on any ARAnchor */
	Array raycast(Vector2 p_screen_coord);

	virtual void notification(int p_what) override;

	virtual StringName get_name() const override;
	virtual int get_capabilities() const override;

	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	virtual Size2 get_render_targetsize() override;
	virtual bool is_stereo() override;
	virtual Transform get_transform_for_eye(XRInterface::Eyes p_eye, const Transform &p_cam_transform) override;
	virtual CameraMatrix get_projection_for_eye(XRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) override;
	virtual void commit_for_eye(XRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) override;

	virtual void process() override;

	// called by delegate (void * because C++ and Obj-C don't always mix, should really change all platform/iphone/*.cpp files to .mm)
	void _add_or_update_anchor(GodotARAnchor *p_anchor);
	void _remove_anchor(GodotARAnchor *p_anchor);

	ARKitInterface();
	~ARKitInterface();
};

#endif /* !ARKIT_INTERFACE_H */
