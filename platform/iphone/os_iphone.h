/*************************************************************************/
/*  os_iphone.h                                                          */
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

#ifdef IPHONE_ENABLED

#ifndef OS_IPHONE_H
#define OS_IPHONE_H

#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "drivers/unix/os_unix.h"
#include "ios.h"
#include "joypad_iphone.h"
#include "servers/audio_server.h"
#include "servers/rendering/renderer_compositor.h"

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/iphone/vulkan_context_iphone.h"
#endif

extern void godot_ios_plugins_initialize();
extern void godot_ios_plugins_deinitialize();

class OSIPhone : public OS_Unix {
private:
	static HashMap<String, void *> dynamic_symbol_lookup_table;
	friend void register_dynamic_symbol(char *name, void *address);

	AudioDriverCoreAudio audio_driver;

	iOS *ios;

	JoypadIPhone *joypad_iphone;

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

	void deinitialize_modules();

public:
	static OSIPhone *get_singleton();

	OSIPhone(String p_data_dir);
	~OSIPhone();

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

	virtual void vibrate_handheld(int p_duration_ms = 500) override;

	virtual bool _check_internal_feature_support(const String &p_feature) override;

	void on_focus_out();
	void on_focus_in();
};

#endif // OS_IPHONE_H

#endif // IPHONE_ENABLED
