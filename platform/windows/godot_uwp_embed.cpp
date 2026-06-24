/**************************************************************************/
/*  godot_uwp_embed.cpp                                                   */
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
/* included in all copies or substantial portions of the Software.       */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifdef GODOT_UWP_EMBED_ENABLED

#include "godot_uwp_embed.h"

#include "display_server_embedded_win.h"
#include "os_windows.h"
#include "uwp_host.h"

#include "core/config/engine.h"
#include "core/io/logger.h"
#include "core/os/main_loop.h"
#include "main/main.h"

#if defined(D3D12_ENABLED)
#include "drivers/d3d12/rendering_context_driver_d3d12.h"
#endif

#include <unknwn.h>

#include <stdio.h>

// -----------------------------------------------------------------------
// Shared host-set state
// -----------------------------------------------------------------------

namespace GodotUwpEmbedState {
IUnknown *swap_chain_panel_native = nullptr;
Size2i initial_size;
Vector2 initial_scale = Vector2(1, 1);
} //namespace GodotUwpEmbedState

static GodotUwpLogCallback host_log_callback = nullptr;

static OS_Windows *embed_os = nullptr;
static bool embed_started = false;
static UWPHost *uwp_host_singleton = nullptr;

// Message-bus state shared with winui3_host.cpp. Engine-thread only.
namespace GodotUwpEmbedBus {
GodotUwpHostMsgCallback host_msg_callback = nullptr;
String pending_call_return;
bool pending_call_return_set = false;
} //namespace GodotUwpEmbedBus

// Forwards every Godot print/warning/error line to the host callback.
class HostCallbackLogger : public Logger {
public:
	virtual void logv(const char *p_format, va_list p_list, bool p_err) override {
		GodotUwpLogCallback cb = host_log_callback;
		if (cb == nullptr) {
			return;
		}
		char buf[4096];
		int len = vsnprintf(buf, sizeof(buf) - 1, p_format, p_list);
		if (len <= 0) {
			return;
		}
		buf[sizeof(buf) - 1] = '\0';
		// Strip a single trailing newline; the host appends its own.
		if (len < (int)sizeof(buf) && len > 0 && buf[len - 1] == '\n') {
			buf[len - 1] = '\0';
		}
		cb(buf, p_err ? 2 : 0);
	}
};

// -----------------------------------------------------------------------
// Configuration (pre-setup)
// -----------------------------------------------------------------------

GODOT_UWP_API void godot_uwp_set_log_callback(GodotUwpLogCallback p_callback) {
	host_log_callback = p_callback;
}

GODOT_UWP_API void godot_uwp_set_swap_chain_panel(void *p_panel_native_iunknown) {
	IUnknown *panel = (IUnknown *)p_panel_native_iunknown;
	if (panel != nullptr) {
		panel->AddRef();
	}
	if (GodotUwpEmbedState::swap_chain_panel_native != nullptr) {
		GodotUwpEmbedState::swap_chain_panel_native->Release();
	}
	GodotUwpEmbedState::swap_chain_panel_native = panel;
}

GODOT_UWP_API void godot_uwp_set_ui_dispatcher(GodotUwpUiDispatchFunc p_dispatch) {
#if defined(D3D12_ENABLED)
	RenderingContextDriverD3D12::embed_ui_dispatch = p_dispatch;
#endif
}

// -----------------------------------------------------------------------
// Lifecycle (engine thread)
// -----------------------------------------------------------------------

GODOT_UWP_API int godot_uwp_engine_setup(int p_argc, char **p_argv) {
	if (embed_os != nullptr) {
		ERR_PRINT("godot_uwp_engine_setup called twice.");
		return 0;
	}
	if (p_argc < 1 || p_argv == nullptr) {
		return 0;
	}

	embed_os = new OS_Windows(GetModuleHandle(nullptr));
	embed_os->add_logger(memnew(HostCallbackLogger));

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
	if (err != OK) {
		if (host_log_callback) {
			host_log_callback("Main::setup failed.", 2);
		}
		delete embed_os;
		embed_os = nullptr;
		return 0;
	}
	return 1;
}

GODOT_UWP_API int godot_uwp_engine_start(void) {
	ERR_FAIL_NULL_V_MSG(embed_os, 0, "Engine not set up.");

	Error err = Main::setup2();
	if (err != OK) {
		if (host_log_callback) {
			host_log_callback("Main::setup2 failed (display/rendering init).", 2);
		}
		return 0;
	}

	// Expose the host message bus to GDScript before any script runs, so
	// autoload _ready can already see Engine.has_singleton("UWPHost").
	GDREGISTER_CLASS(UWPHost);
	uwp_host_singleton = memnew(UWPHost);
	Engine::get_singleton()->add_singleton(Engine::Singleton("UWPHost", uwp_host_singleton));

	if (Main::start() != EXIT_SUCCESS) {
		if (host_log_callback) {
			host_log_callback("Main::start failed (project load).", 2);
		}
		return 0;
	}

	MainLoop *main_loop = OS::get_singleton()->get_main_loop();
	if (main_loop != nullptr) {
		main_loop->initialize();
	}
	embed_started = true;
	return 1;
}

GODOT_UWP_API int godot_uwp_engine_iteration(void) {
	if (!embed_started) {
		return 1;
	}
	DisplayServer::get_singleton()->process_events();
	return Main::iteration() ? 1 : 0;
}

GODOT_UWP_API void godot_uwp_engine_shutdown(void) {
	if (embed_os == nullptr) {
		return;
	}
	if (embed_started) {
		MainLoop *main_loop = OS::get_singleton()->get_main_loop();
		if (main_loop != nullptr) {
			main_loop->finalize();
		}
		embed_started = false;
	}
	if (uwp_host_singleton != nullptr) {
		Engine::get_singleton()->remove_singleton("UWPHost");
		memdelete(uwp_host_singleton);
		uwp_host_singleton = nullptr;
	}
	GodotUwpEmbedBus::host_msg_callback = nullptr;
	Main::cleanup();
	delete embed_os;
	embed_os = nullptr;

	if (GodotUwpEmbedState::swap_chain_panel_native != nullptr) {
		GodotUwpEmbedState::swap_chain_panel_native->Release();
		GodotUwpEmbedState::swap_chain_panel_native = nullptr;
	}
}

// -----------------------------------------------------------------------
// Sizing / input (engine thread)
// -----------------------------------------------------------------------

GODOT_UWP_API void godot_uwp_notify_panel_resize(int p_width_px, int p_height_px) {
	DisplayServerEmbeddedWin *ds = DisplayServerEmbeddedWin::get_embedded_singleton();
	if (ds != nullptr) {
		ds->host_resize(p_width_px, p_height_px);
	} else {
		GodotUwpEmbedState::initial_size = Size2i(p_width_px, p_height_px);
	}
}

GODOT_UWP_API void godot_uwp_set_composition_scale(float p_scale_x, float p_scale_y) {
	DisplayServerEmbeddedWin *ds = DisplayServerEmbeddedWin::get_embedded_singleton();
	if (ds != nullptr) {
		ds->host_set_composition_scale(p_scale_x, p_scale_y);
	} else {
		GodotUwpEmbedState::initial_scale = Vector2(p_scale_x, p_scale_y);
	}
}

GODOT_UWP_API void godot_uwp_inject_mouse_button(int p_button, int p_pressed, float p_x, float p_y, int p_double_click) {
	DisplayServerEmbeddedWin *ds = DisplayServerEmbeddedWin::get_embedded_singleton();
	if (ds != nullptr) {
		ds->host_inject_mouse_button((MouseButton)p_button, p_pressed != 0, p_x, p_y, p_double_click != 0);
	}
}

GODOT_UWP_API void godot_uwp_inject_mouse_motion(float p_x, float p_y, float p_rel_x, float p_rel_y) {
	DisplayServerEmbeddedWin *ds = DisplayServerEmbeddedWin::get_embedded_singleton();
	if (ds != nullptr) {
		ds->host_inject_mouse_motion(p_x, p_y, p_rel_x, p_rel_y);
	}
}

GODOT_UWP_API void godot_uwp_inject_mouse_wheel(float p_x, float p_y, float p_delta_x, float p_delta_y) {
	DisplayServerEmbeddedWin *ds = DisplayServerEmbeddedWin::get_embedded_singleton();
	if (ds != nullptr) {
		ds->host_inject_mouse_wheel(p_x, p_y, p_delta_x, p_delta_y);
	}
}

GODOT_UWP_API void godot_uwp_inject_key(unsigned int p_win_vk, int p_pressed, int p_echo, unsigned int p_unicode) {
	DisplayServerEmbeddedWin *ds = DisplayServerEmbeddedWin::get_embedded_singleton();
	if (ds != nullptr) {
		ds->host_inject_key(p_win_vk, p_pressed != 0, p_echo != 0, (char32_t)p_unicode);
	}
}

// -----------------------------------------------------------------------
// Host <-> engine JSON message bus (engine thread)
// -----------------------------------------------------------------------

GODOT_UWP_API void godot_uwp_set_host_message_callback(GodotUwpHostMsgCallback p_callback) {
	GodotUwpEmbedBus::host_msg_callback = p_callback;
}

GODOT_UWP_API void godot_uwp_set_call_return(const char *p_json_utf8) {
	if (p_json_utf8 == nullptr) {
		GodotUwpEmbedBus::pending_call_return = String();
		GodotUwpEmbedBus::pending_call_return_set = false;
		return;
	}
	GodotUwpEmbedBus::pending_call_return = String::utf8(p_json_utf8);
	GodotUwpEmbedBus::pending_call_return_set = true;
}

GODOT_UWP_API int godot_uwp_call_engine(const char *p_method_utf8, const char *p_args_json_utf8, char **r_ret_json_utf8) {
	if (r_ret_json_utf8 != nullptr) {
		*r_ret_json_utf8 = nullptr;
	}
	if (!embed_started || uwp_host_singleton == nullptr || p_method_utf8 == nullptr) {
		return 0;
	}

	String method = String::utf8(p_method_utf8);
	String args_json = p_args_json_utf8 != nullptr ? String::utf8(p_args_json_utf8) : String();

	String ret_json;
	uwp_host_singleton->call_handler(method, args_json, ret_json);

	if (r_ret_json_utf8 != nullptr && !ret_json.is_empty()) {
		CharString utf8 = ret_json.utf8();
		char *buf = (char *)memalloc(utf8.size());
		memcpy(buf, utf8.get_data(), utf8.size());
		*r_ret_json_utf8 = buf;
	}
	return 1;
}

GODOT_UWP_API void godot_uwp_free_string(char *p_str) {
	if (p_str != nullptr) {
		memfree(p_str);
	}
}

#endif // GODOT_UWP_EMBED_ENABLED
