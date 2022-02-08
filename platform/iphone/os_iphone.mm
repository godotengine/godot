/*************************************************************************/
/*  os_iphone.mm                                                         */
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

#include "os_iphone.h"

#import "app_delegate.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/file_access_pack.h"
#include "display_server_iphone.h"
#include "drivers/unix/syslog_logger.h"
#import "godot_view.h"
#import "godot_view_controller.h"
#include "main/main.h"

#import <AudioToolbox/AudioServices.h>
#import <UIKit/UIKit.h>
#import <dlfcn.h>
#include <sys/sysctl.h>

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#import <QuartzCore/CAMetalLayer.h>
#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif
#endif

// Initialization order between compilation units is not guaranteed,
// so we use this as a hack to ensure certain code is called before
// everything else, but after all units are initialized.
typedef void (*init_callback)();
static init_callback *ios_init_callbacks = nullptr;
static int ios_init_callbacks_count = 0;
static int ios_init_callbacks_capacity = 0;
HashMap<String, void *> OSIPhone::dynamic_symbol_lookup_table;

void add_ios_init_callback(init_callback cb) {
	if (ios_init_callbacks_count == ios_init_callbacks_capacity) {
		void *new_ptr = realloc(ios_init_callbacks, sizeof(cb) * 32);
		if (new_ptr) {
			ios_init_callbacks = (init_callback *)(new_ptr);
			ios_init_callbacks_capacity += 32;
		}
	}
	if (ios_init_callbacks_capacity > ios_init_callbacks_count) {
		ios_init_callbacks[ios_init_callbacks_count] = cb;
		++ios_init_callbacks_count;
	}
}

void register_dynamic_symbol(char *name, void *address) {
	OSIPhone::dynamic_symbol_lookup_table[String(name)] = address;
}

OSIPhone *OSIPhone::get_singleton() {
	return (OSIPhone *)OS::get_singleton();
}

OSIPhone::OSIPhone(String p_data_dir, String p_cache_dir) :
		OS_UIKit(p_data_dir, p_cache_dir) {
	for (int i = 0; i < ios_init_callbacks_count; ++i) {
		ios_init_callbacks[i]();
	}
	free(ios_init_callbacks);
	ios_init_callbacks = nullptr;
	ios_init_callbacks_count = 0;
	ios_init_callbacks_capacity = 0;

	DisplayServerIPhone::register_iphone_driver();
}

OSIPhone::~OSIPhone() {}

void OSIPhone::alert(const String &p_alert, const String &p_title) {
	const CharString utf8_alert = p_alert.utf8();
	const CharString utf8_title = p_title.utf8();
	iOS::alert(utf8_alert.get_data(), utf8_title.get_data());
}

void OSIPhone::initialize() {
	OS_UIKit::initialize();
}

void OSIPhone::initialize_modules() {
	ios = memnew(iOS);
	Engine::get_singleton()->add_singleton(Engine::Singleton("iOS", ios));
}

void OSIPhone::finalize() {
	if (ios) {
		memdelete(ios);
	}

	OS_UIKit::finalize();
}

// MARK: Dynamic Libraries

Error OSIPhone::get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional) {
	if (p_library_handle == RTLD_SELF) {
		void **ptr = OSIPhone::dynamic_symbol_lookup_table.getptr(p_name);
		if (ptr) {
			p_symbol_handle = *ptr;
			return OK;
		}
	}
	return OS_Unix::get_dynamic_library_symbol_handle(p_library_handle, p_name, p_symbol_handle, p_optional);
}

String OSIPhone::get_name() const {
	return "iOS";
}

String OSIPhone::get_model_name() const {
	String model = ios->get_model();
	if (model != "") {
		return model;
	}

	return OS_Unix::get_model_name();
}

String OSIPhone::get_processor_name() const {
	char buffer[256];
	size_t buffer_len = 256;
	if (sysctlbyname("machdep.cpu.brand_string", &buffer, &buffer_len, NULL, 0) == 0) {
		return String::utf8(buffer, buffer_len);
	}
	ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
}

void OSIPhone::vibrate_handheld(int p_duration_ms) {
	// iOS does not support duration for vibration
	AudioServicesPlaySystemSound(kSystemSoundID_Vibrate);
}

bool OSIPhone::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "mobile";
}

void OSIPhone::on_focus_out() {
	if (is_focused) {
		is_focused = false;

		if (DisplayServerIPhone::get_singleton()) {
			DisplayServerIPhone::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
		}

		[AppDelegate.viewController.godotView stopRendering];

		audio_driver.stop();
	}
}

void OSIPhone::on_focus_in() {
	if (!is_focused) {
		is_focused = true;

		if (DisplayServerIPhone::get_singleton()) {
			DisplayServerIPhone::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
		}

		[AppDelegate.viewController.godotView startRendering];

		audio_driver.start();
	}
}
