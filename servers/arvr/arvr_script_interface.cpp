#include "arvr_script_interface.h"

ARVRScriptInterface::ARVRScriptInterface() {
	// testing
	printf("Construct script interface");
}

ARVRScriptInterface::~ARVRScriptInterface() {
	if (is_initialized()) {
		uninitialize();
	};

	// testing
	printf("Destruct script interface");
}

StringName ARVRScriptInterface::get_name() const {
	if (get_script_instance() && get_script_instance()->has_method("get_name")) {
		return get_script_instance()->call("get_name");
	} else {
		// just return something for now
		return "ARVR Script interface";
	}
}

int ARVRScriptInterface::get_capabilities() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("get_capabilities")), ARVRInterface::ARVR_NONE);
	return get_script_instance()->call("get_capabilities");
};

ARVRInterface::Tracking_status ARVRScriptInterface::get_tracking_status() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("get_tracking_status")), ARVRInterface::ARVR_NOT_TRACKING);
	int status = get_script_instance()->call("get_tracking_status");
	return (ARVRInterface::Tracking_status)status;
}

bool ARVRScriptInterface::get_anchor_detection_is_enabled() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("get_anchor_detection_is_enabled")), false);
	return get_script_instance()->call("get_anchor_detection_is_enabled");
};

void ARVRScriptInterface::set_anchor_detection_is_enabled(bool p_enable) {
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("set_anchor_detection_is_enabled")));
	get_script_instance()->call("set_anchor_detection_is_enabled");
};

bool ARVRScriptInterface::is_stereo() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("is_stereo")), false);
	return get_script_instance()->call("is_stereo");
}

bool ARVRScriptInterface::is_initialized() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("is_initialized")), false);
	return get_script_instance()->call("is_initialized");
}

bool ARVRScriptInterface::initialize() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("initialize")), false);
	return get_script_instance()->call("initialize");
}

void ARVRScriptInterface::uninitialize() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	if (arvr_server != NULL) {
		// Whatever happens, make sure this is no longer our primary interface
		arvr_server->clear_primary_interface_if(this);
	}

	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("uninitialize")));
	get_script_instance()->call("uninitialize");
}

Size2 ARVRScriptInterface::get_recommended_render_targetsize() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("get_recommended_render_targetsize")), Size2());
	return get_script_instance()->call("get_recommended_render_targetsize");
}

Transform ARVRScriptInterface::get_transform_for_eye(Eyes p_eye, const Transform &p_cam_transform) {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("get_transform_for_eye")), Transform());
	return get_script_instance()->call("get_transform_for_eye", p_eye, p_cam_transform);
}

// Suggestion from Reduz, as we can't return a CameraMatrix, return a PoolVector with our 16 floats
PoolVector<float> ARVRScriptInterface::_get_projection_for_eye(Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_projection_for_eye")), PoolVector<float>());
	return get_script_instance()->call("_get_projection_for_eye", p_eye, p_aspect, p_z_near, p_z_far);
}

CameraMatrix ARVRScriptInterface::get_projection_for_eye(Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	CameraMatrix cm;
	int i = 0;
	int j = 0;

	PoolVector<float> cm_as_floats = _get_projection_for_eye(p_eye, p_aspect, p_z_near, p_z_far);

	for (int k = 0; k < cm_as_floats.size() && i < 4; k++) {
		cm.matrix[i][j] = cm_as_floats[k];
		j++;
		if (j == 4) {
			j = 0;
			i++;
		};
	};

	return cm;
}

void ARVRScriptInterface::commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("commit_for_eye")));
	get_script_instance()->call("commit_for_eye");
}

void ARVRScriptInterface::process() {
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("process")));
	get_script_instance()->call("process");
}

void ARVRScriptInterface::_bind_methods() {
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::INT, "get_capabilities"));

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "is_initialized"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "initialize"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("uninitialize"));

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::INT, "get_tracking_status"));

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "get_anchor_detection_is_enabled"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("set_anchor_detection_is_enabled", PropertyInfo(Variant::BOOL, "enabled")));

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "is_stereo"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::VECTOR2, "get_recommended_render_targetsize"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::TRANSFORM, "get_transform_for_eye", PropertyInfo(Variant::INT, "eye"), PropertyInfo(Variant::TRANSFORM, "cam_transform")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("_get_projection_for_eye"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("commit_for_eye", PropertyInfo(Variant::INT, "eye"), PropertyInfo(Variant::_RID, "render_target")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("process"));
}
