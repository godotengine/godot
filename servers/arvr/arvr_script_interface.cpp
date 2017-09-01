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
	if (get_script_instance() && get_script_instance()->has_method("_get_name")) {
		return get_script_instance()->call("_get_name");
	} else {
		// just return something for now
		return "ARVR Script interface";
	}
}

bool ARVRScriptInterface::is_installed() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_is_installed")), false);
	return get_script_instance()->call("_is_installed");
}

bool ARVRScriptInterface::hmd_is_present() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_hmd_is_present")), false);
	return get_script_instance()->call("_hmd_is_present");
}

bool ARVRScriptInterface::supports_hmd() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_supports_hmd")), false);
	return get_script_instance()->call("_supports_hmd");
}

bool ARVRScriptInterface::is_stereo() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_is_stereo")), false);
	return get_script_instance()->call("_is_stereo");
}

bool ARVRScriptInterface::is_initialized() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_is_initialized")), false);
	return get_script_instance()->call("_is_initialized");
}

bool ARVRScriptInterface::initialize() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_initialize")), false);
	return get_script_instance()->call("_initialize");
}

void ARVRScriptInterface::uninitialize() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	if (arvr_server != NULL) {
		// Whatever happens, make sure this is no longer our primary interface
		arvr_server->clear_primary_interface_if(this);
	}

	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("_uninitialize")));
	get_script_instance()->call("_uninitialize");
}

Size2 ARVRScriptInterface::get_recommended_render_targetsize() {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_recommended_render_targetsize")), Size2());
	return get_script_instance()->call("_get_recommended_render_targetsize");
}

Transform ARVRScriptInterface::get_transform_for_eye(Eyes p_eye, const Transform &p_cam_transform) {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_transform_for_eye")), Transform());
	return get_script_instance()->call("_get_transform_for_eye", p_eye, p_cam_transform);
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
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("_commit_for_eye")));
	get_script_instance()->call("_commit_for_eye");
}

void ARVRScriptInterface::process() {
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("_process")));
	get_script_instance()->call("_process");
}

void ARVRScriptInterface::_bind_methods() {
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_is_installed"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_hmd_is_present"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_supports_hmd"));

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_is_initialized"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_initialize"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("_uninitialize"));

	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::BOOL, "_is_stereo"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::VECTOR2, "_get_recommended_render_targetsize"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo(Variant::TRANSFORM, "_get_transform_for_eye", PropertyInfo(Variant::INT, "eye"), PropertyInfo(Variant::TRANSFORM, "cam_transform")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("_get_projection_for_eye"));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("_commit_for_eye", PropertyInfo(Variant::INT, "eye"), PropertyInfo(Variant::_RID, "render_target")));
	ClassDB::add_virtual_method(get_class_static(), MethodInfo("_process"));
}
