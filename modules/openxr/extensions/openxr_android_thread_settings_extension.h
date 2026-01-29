/**************************************************************************/
/*  openxr_android_thread_settings_extension.h                            */
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

#include "../openxr_interface.h"
#include "core/templates/hash_map.h"

#include "openxr_extension_wrapper.h"

#ifdef XR_USE_PLATFORM_ANDROID
#include "../util.h"
#include <jni.h>
#include <openxr/openxr_platform.h>
#endif

class OpenXRAndroidThreadSettingsExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRAndroidThreadSettingsExtension, OpenXRExtensionWrapper);

public:
	static OpenXRAndroidThreadSettingsExtension *get_singleton();

	OpenXRAndroidThreadSettingsExtension();
	virtual ~OpenXRAndroidThreadSettingsExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;
	virtual void on_instance_created(XrInstance p_instance) override;
	virtual void on_session_created(XrSession p_session) override;

	enum ThreadType {
		THREAD_TYPE_APPLICATION_MAIN,
		THREAD_TYPE_APPLICATION_WORKER,
		THREAD_TYPE_RENDERER_MAIN,
		THREAD_TYPE_RENDERER_WORKER,
	};
	bool set_application_thread_type(ThreadType p_thread_type, uint32_t p_thread_id = 0);

protected:
	static void _bind_methods();

private:
	static OpenXRAndroidThreadSettingsExtension *singleton;

	bool _initialize_openxr_android_thread_settings_extension();
	void _set_render_thread_type();

	bool available = false;

#ifdef XR_USE_PLATFORM_ANDROID
	EXT_PROTO_XRRESULT_FUNC3(xrSetAndroidApplicationThreadKHR, (XrSession), session, (XrAndroidThreadTypeKHR), threadType, (uint32_t), threadId);
#endif
};

VARIANT_ENUM_CAST(OpenXRAndroidThreadSettingsExtension::ThreadType)
