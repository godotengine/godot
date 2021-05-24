/*************************************************************************/
/*  os_appletv.h                                                         */
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

#ifdef TVOS_ENABLED

#ifndef OS_APPLETV_H
#define OS_APPLETV_H

#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "drivers/unix/os_unix.h"
#include "joypad_appletv.h"
#include "servers/audio_server.h"
#include "servers/rendering/renderer_compositor.h"
#include "tvos.h"

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "vulkan_context_appletv.h"
#endif

extern void godot_tvos_plugins_initialize();
extern void godot_tvos_plugins_deinitialize();

class OSAppleTV : public OS_Unix {
private:
	static HashMap<String, void *> dynamic_symbol_lookup_table;
	friend void register_dynamic_symbol(char *name, void *address);

	AudioDriverCoreAudio audio_driver;

	tvOS *tvos;

	JoypadAppleTV *joypad_appletv;

	MainLoop *main_loop;

	virtual void initialize_core() override;
	virtual void initialize() override;

	virtual void initialize_joypads() override {
	}

	virtual void set_main_loop(MainLoop *p_main_loop) override;
	virtual MainLoop *get_main_loop() const override;

	virtual void delete_main_loop() override;

	virtual void finalize() override;

	String user_data_dir;

	bool is_focused = false;
	bool overrides_menu_button = true;

	void deinitialize_modules();

public:
	static OSAppleTV *get_singleton();

	OSAppleTV(String p_data_dir);
	~OSAppleTV();

	void initialize_modules();

	bool iterate();

	void start();

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) override;
	virtual Error close_dynamic_library(void *p_library_handle) override;
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false) override;

	virtual void alert(const String &p_alert,
			const String &p_title = "ALERT!") override;

	virtual String get_name() const override;
	virtual String get_model_name() const override;

	virtual Error shell_open(String p_uri) override;

	void set_user_data_dir(String p_dir);
	virtual String get_user_data_dir() const override;

	virtual String get_locale() const override;

	virtual String get_unique_id() const override;

	virtual bool _check_internal_feature_support(const String &p_feature) override;

	int joy_id_for_name(const String &p_name);

	void on_focus_out();
	void on_focus_in();

	bool get_overrides_menu_button() const;
	void set_overrides_menu_button(bool p_flag);
};

#endif // OS_IPHONE_H

#endif // IPHONE_ENABLED
