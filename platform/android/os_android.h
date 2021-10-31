/*************************************************************************/
/*  os_android.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef OS_ANDROID_H
#define OS_ANDROID_H

#include "audio_driver_opensl.h"
#include "core/os/main_loop.h"
#include "drivers/unix/os_unix.h"
#include "servers/audio_server.h"

class GodotJavaWrapper;
class GodotIOJavaWrapper;

struct ANativeWindow;

class OS_Android : public OS_Unix {
private:
	Size2i display_size;

	bool use_apk_expansion;

#if defined(GLES3_ENABLED)
	const char *gl_extensions;
#endif

#if defined(VULKAN_ENABLED)
	ANativeWindow *native_window;
#endif

	mutable String data_dir_cache;
	mutable String cache_dir_cache;

	AudioDriverOpenSL audio_driver_android;

	MainLoop *main_loop;

	GodotJavaWrapper *godot_java;
	GodotIOJavaWrapper *godot_io_java;

public:
	virtual void initialize_core() override;
	virtual void initialize() override;

	virtual void initialize_joypads() override;

	virtual void set_main_loop(MainLoop *p_main_loop) override;
	virtual void delete_main_loop() override;

	virtual void finalize() override;

	typedef int64_t ProcessID;

	static OS_Android *get_singleton();
	GodotJavaWrapper *get_godot_java();
	GodotIOJavaWrapper *get_godot_io_java();

	virtual bool request_permission(const String &p_name) override;
	virtual bool request_permissions() override;
	virtual Vector<String> get_granted_permissions() const override;

	virtual void alert(const String &p_alert, const String &p_title) override;

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) override;

	virtual String get_name() const override;
	virtual MainLoop *get_main_loop() const override;

	void main_loop_begin();
	bool main_loop_iterate();
	void main_loop_end();
	void main_loop_focusout();
	void main_loop_focusin();

	void set_display_size(const Size2i &p_size);
	Size2i get_display_size() const;

	void set_opengl_extensions(const char *p_gl_extensions);

	void set_native_window(ANativeWindow *p_native_window);
	ANativeWindow *get_native_window() const;

	virtual Error shell_open(String p_uri) override;
	virtual String get_user_data_dir() const override;
	virtual String get_data_path() const override;
	virtual String get_cache_path() const override;
	virtual String get_resource_dir() const override;
	virtual String get_locale() const override;
	virtual String get_model_name() const override;

	virtual String get_unique_id() const override;

	virtual String get_system_dir(SystemDir p_dir, bool p_shared_storage = true) const override;

	void vibrate_handheld(int p_duration_ms) override;

	virtual bool _check_internal_feature_support(const String &p_feature) override;
	OS_Android(GodotJavaWrapper *p_godot_java, GodotIOJavaWrapper *p_godot_io_java, bool p_use_apk_expansion);
	~OS_Android();
};

#endif
