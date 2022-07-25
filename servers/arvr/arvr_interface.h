/*************************************************************************/
/*  arvr_interface.h                                                     */
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

#ifndef ARVR_INTERFACE_H
#define ARVR_INTERFACE_H

#include "core/math/camera_matrix.h"
#include "core/os/thread_safe.h"
#include "scene/main/viewport.h"
#include "servers/arvr_server.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>

	The ARVR interface is a template class on top of which we build interface to different AR, VR and tracking SDKs.
	The idea is that we subclass this class, implement the logic, and then instantiate a singleton of each interface
	when Godot starts. These instances do not initialize themselves but register themselves with the AR/VR server.

	If the user wants to enable AR/VR the choose the interface they want to use and initialize it.

	Note that we may make this into a fully instantiable class for GDNative support.
*/

class ARVRInterface : public Reference {
	GDCLASS(ARVRInterface, Reference);

public:
	enum Capabilities { /* purely meta data, provides some info about what this interface supports */
		ARVR_NONE = 0, /* no capabilities */
		ARVR_MONO = 1, /* can be used with mono output */
		ARVR_STEREO = 2, /* can be used with stereo output */
		ARVR_AR = 4, /* offers a camera feed for AR */
		ARVR_EXTERNAL = 8 /* renders to external device */
	};

	enum Eyes {
		EYE_MONO, /* my son says we should call this EYE_CYCLOPS */
		EYE_LEFT,
		EYE_RIGHT
	};

	enum Tracking_status { /* tracking status currently based on AR but we can start doing more with this for VR as well */
		ARVR_NORMAL_TRACKING,
		ARVR_EXCESSIVE_MOTION,
		ARVR_INSUFFICIENT_FEATURES,
		ARVR_UNKNOWN_TRACKING,
		ARVR_NOT_TRACKING
	};

protected:
	_THREAD_SAFE_CLASS_

	Tracking_status tracking_state;
	static void _bind_methods();

public:
	/** general interface information **/
	virtual StringName get_name() const;
	virtual int get_capabilities() const = 0;

	bool is_primary();
	void set_is_primary(bool p_is_primary);

	virtual bool is_initialized() const = 0; /* returns true if we've initialized this interface */
	void set_is_initialized(bool p_initialized); /* helper function, will call initialize or uninitialize */
	virtual bool initialize() = 0; /* initialize this interface, if this has an HMD it becomes the primary interface */
	virtual void uninitialize() = 0; /* deinitialize this interface */

	Tracking_status get_tracking_status() const; /* get the status of our current tracking */

	/** specific to VR **/
	// nothing yet

	/** specific to AR **/
	virtual bool get_anchor_detection_is_enabled() const;
	virtual void set_anchor_detection_is_enabled(bool p_enable);
	virtual int get_camera_feed_id();

	/** rendering and internal **/

	virtual Size2 get_render_targetsize() = 0; /* returns the recommended render target size per eye for this device */
	virtual bool is_stereo() = 0; /* returns true if this interface requires stereo rendering (for VR HMDs) or mono rendering (for mobile AR) */
	virtual Transform get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform) = 0; /* get each eyes camera transform, also implement EYE_MONO */
	virtual CameraMatrix get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) = 0; /* get each eyes projection matrix */
	virtual unsigned int get_external_texture_for_eye(ARVRInterface::Eyes p_eye); /* if applicable return external texture to render to */
	virtual unsigned int get_external_depth_for_eye(ARVRInterface::Eyes p_eye); /* if applicable return external depth texture to render to */
	virtual void commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) = 0; /* output the left or right eye */

	virtual void process() = 0;
	virtual void notification(int p_what) = 0;

	ARVRInterface();
	~ARVRInterface();
};

VARIANT_ENUM_CAST(ARVRInterface::Capabilities);
VARIANT_ENUM_CAST(ARVRInterface::Eyes);
VARIANT_ENUM_CAST(ARVRInterface::Tracking_status);

#endif // ARVR_INTERFACE_H
