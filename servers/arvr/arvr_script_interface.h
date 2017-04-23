#ifndef SCRIPT_INTERFACE_H
#define SCRIPT_INTERFACE_H

#include "arvr_interface.h"

/**
	@authors Hinsbart & Karroffel

	This subclass of our AR/VR interface forms a bridge to GDNative.
*/

class ARVRScriptInterface : public ARVRInterface {
	GDCLASS(ARVRScriptInterface, ARVRInterface);

protected:
	static void _bind_methods();

public:
	ARVRScriptInterface();
	~ARVRScriptInterface();

	virtual StringName get_name() const;

	virtual bool is_installed();
	virtual bool hmd_is_present();
	virtual bool supports_hmd();

	virtual bool is_initialized();
	virtual bool initialize();
	virtual void uninitialize();

	virtual Size2 get_recommended_render_targetsize();
	virtual bool is_stereo();
	virtual Transform get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform);

	// we expose a PoolVector<float> version of this function to GDNative
	PoolVector<float> _get_projection_for_eye(Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far);

	// and a CameraMatrix version to ARVRServer
	virtual CameraMatrix get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far);

	virtual void commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect);

	virtual void process();
};

#endif // SCRIPT_INTERFACE_H
