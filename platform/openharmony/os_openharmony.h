/**************************************************************************/
/*  os_openharmony.h                                                      */
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

#include "audio_driver_openharmony.h"

#include "core/os/main_loop.h"
#include "drivers/unix/os_unix.h"
#include "drivers/vulkan/godot_vulkan.h"
#include "servers/audio_server.h"

struct OH_Drawing_FontDescriptor;

class OS_OpenHarmony : public OS_Unix {
	Size2i display_size;
	OHNativeWindow *native_window = nullptr;
	MainLoop *main_loop = nullptr;
	AudioDriverOpenHarmony audio_driver;
	int32_t window_id = -1;
	bool is_focused = false;
	HashSet<String> allowed_permissions;

	struct FontInfo {
		String font_name;
		HashSet<String> lang;
		HashSet<String> script;
		int weight = 400;
		int stretch = 100;
		bool italic = false;
		int priority = 0;
		String path;
		OH_Drawing_FontDescriptor *descriptor;
	};
	mutable bool font_config_loaded = false;
	mutable HashMap<String, String> font_aliases;
	mutable List<FontInfo> fonts;
	mutable HashSet<String> font_names;

	void _load_system_font_config() const;

public:
	static const char *EXEC_PATH;
	static const char *BUNDLE_RESOURCE_DIR;
	static const char *USER_DATA_DIR;

	static OS_OpenHarmony *get_singleton();

	OS_OpenHarmony();

	void set_native_window(OHNativeWindow *p_native_window);
	OHNativeWindow *get_native_window() const;

	void set_window_id(int32_t p_window_id);
	int32_t get_window_id() const;

	void set_display_size(const Size2i &p_size);
	Size2i get_display_size() const;

	void set_allowed_permissions(const char *p_allowed_permissions);

	virtual bool request_permission(const String &p_name) override;
	virtual bool request_permissions() override;

	virtual void initialize() override;
	virtual void initialize_joypads() override;
	virtual void set_main_loop(MainLoop *p_main_loop) override;
	virtual MainLoop *get_main_loop() const override;
	virtual void delete_main_loop() override;
	virtual void finalize() override;
	virtual bool _check_internal_feature_support(const String &p_feature) override;

	virtual String get_user_data_dir(const String &p_user_dir) const override;
	virtual String get_bundle_resource_dir() const override;
	virtual String get_executable_path() const override;

	virtual Vector<String> get_system_fonts() const override;
	virtual String get_system_font_path(const String &p_font_name, int p_weight = 400, int p_stretch = 100, bool p_italic = false) const override;
	virtual Vector<String> get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale = String(), const String &p_script = String(), int p_weight = 400, int p_stretch = 100, bool p_italic = false) const override;

	virtual String get_system_ca_certificates() override;

	void main_loop_begin();
	bool main_loop_iterate();
	void main_loop_end();

	void on_focus_out();
	void on_focus_in();
	void on_enter_background();
	void on_exit_background();
};

class Logger_OpenHarmony : public Logger {
public:
	virtual void logv(const char *p_format, va_list p_list, bool p_err) override;
};
