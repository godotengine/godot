/*************************************************************************/
/*  arvr_interface_gdnative.h                                            */
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

#ifndef ARVR_INTERFACE_GDNATIVE_H
#define ARVR_INTERFACE_GDNATIVE_H

#include "modules/gdnative/gdnative.h"
#include "servers/arvr/arvr_interface.h"

/**
	@authors Hinsbart & Karroffel & Mux213

	This subclass of our AR/VR interface forms a bridge to GDNative.
*/

class ARVRInterfaceGDNative : public ARVRInterface {
	GDCLASS(ARVRInterfaceGDNative, ARVRInterface);

	void cleanup();

protected:
	const godot_arvr_interface_gdnative *interface;
	void *data;

	static void _bind_methods();

public:
	/** general interface information **/
	ARVRInterfaceGDNative();
	~ARVRInterfaceGDNative();

	void set_interface(const godot_arvr_interface_gdnative *p_interface);

	virtual StringName get_name() const;
	virtual int get_capabilities() const;

	virtual bool is_initialized() const;
	virtual bool initialize();
	virtual void uninitialize();

	/** specific to AR **/
	virtual bool get_anchor_detection_is_enabled() const;
	virtual void set_anchor_detection_is_enabled(bool p_enable);
	virtual int get_camera_feed_id();

	/** rendering and internal **/
	virtual Size2 get_render_targetsize();
	virtual bool is_stereo();
	virtual Transform get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform);

	// and a CameraMatrix version to ARVRServer
	virtual CameraMatrix get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far);

	virtual unsigned int get_external_texture_for_eye(ARVRInterface::Eyes p_eye);
	virtual unsigned int get_external_depth_for_eye(ARVRInterface::Eyes p_eye);
	virtual void commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect);

	virtual void process();
	virtual void notification(int p_what);
};

#endif // ARVR_INTERFACE_GDNATIVE_H
