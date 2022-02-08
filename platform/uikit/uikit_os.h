/*************************************************************************/
/*  uikit_os.h                                                           */
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

#ifndef OS_UIKIT_H
#define OS_UIKIT_H

#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "drivers/unix/os_unix.h"
#include "servers/audio_server.h"
#include "servers/rendering/renderer_compositor.h"
#include "uikit_joypad.h"

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/uikit/uikit_vulkan_context.h"
#endif

class OS_UIKit : public OS_Unix {
private:
	AudioDriverCoreAudio audio_driver;

	UIKitJoypad *uikit_joypad;

	MainLoop *main_loop;

	String user_data_dir;
	String cache_dir;

protected:
	virtual void initialize_core() override;
	virtual void initialize() override;

	virtual void initialize_joypads() override {
	}

	virtual void set_main_loop(MainLoop *p_main_loop) override;
	virtual MainLoop *get_main_loop() const override;

	virtual void delete_main_loop() override;

	virtual void finalize() override;

public:
	static OS_UIKit *get_singleton();

	OS_UIKit(String p_data_dir, String p_cache_dir);
	~OS_UIKit();

	virtual bool iterate();

	virtual void start();

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) override;
	virtual Error close_dynamic_library(void *p_library_handle) override;

	virtual String get_name() const override;

	virtual Error shell_open(String p_uri) override;

	void set_user_data_dir(String p_dir);
	virtual String get_user_data_dir() const override;

	virtual String get_cache_path() const override;

	virtual String get_locale() const override;

	virtual String get_unique_id() const override;

	int joy_id_for_name(const String &p_name);
};

#endif // OS_UIKIT_H
