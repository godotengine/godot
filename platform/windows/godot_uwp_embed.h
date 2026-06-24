/**************************************************************************/
/*  godot_uwp_embed.h                                                     */
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

#pragma once

#ifdef GODOT_UWP_EMBED_ENABLED

// Flat C ABI for hosting Godot inside a XAML SwapChainPanel (UWP / WinUI3).
//
// Threading contract:
//   The engine is single-threaded. godot_uwp_engine_setup / start /
//   iteration / shutdown and every inject/resize call MUST all happen on the
//   same host-owned "engine thread". The only exception is
//   godot_uwp_set_ui_dispatcher's callback, which the engine invokes FROM the
//   engine thread and which must run the work synchronously ON the UI thread
//   (ISwapChainPanelNative::SetSwapChain is UI-thread affine).
//
// Call order:
//   godot_uwp_set_log_callback        (optional, first — captures setup logs)
//   godot_uwp_set_swap_chain_panel    (required before setup)
//   godot_uwp_set_ui_dispatcher       (required before start)
//   godot_uwp_notify_panel_resize     (recommended before setup: initial size)
//   godot_uwp_set_composition_scale   (recommended before setup)
//   godot_uwp_engine_setup(argc, argv)   args must include:
//        --display-driver embedded --rendering-driver d3d12 --main-pack <pck>
//   godot_uwp_engine_start
//   loop: godot_uwp_engine_iteration  (returns 1 when the engine wants to quit)
//   godot_uwp_engine_shutdown

#define GODOT_UWP_API extern "C" __declspec(dllexport)

#ifdef __cplusplus

#include "core/math/vector2.h"
#include "core/math/vector2i.h"

struct IUnknown;

// Internal state shared between the C ABI and DisplayServerEmbeddedWin.
// Host-set values are stashed here before the display server exists.
namespace GodotUwpEmbedState {
extern IUnknown *swap_chain_panel_native; // AddRef'd; consumed by the display server.
extern Size2i initial_size; // Physical pixels.
extern Vector2 initial_scale; // Composition scale (px per DIP).
} //namespace GodotUwpEmbedState

#endif // __cplusplus

typedef void(__cdecl *GodotUwpLogCallback)(const char *p_message, int p_level);
typedef void(__cdecl *GodotUwpWorkFunc)(void *p_userdata);
typedef void(__cdecl *GodotUwpUiDispatchFunc)(GodotUwpWorkFunc p_work, void *p_userdata);
// Invoked (on the engine thread) when GDScript calls
// UWPHost.send_to_host(method, args). args_json is a JSON array, "[]" if
// empty. The handler may reply synchronously by calling
// godot_uwp_set_call_return BEFORE returning.
typedef void(__cdecl *GodotUwpHostMsgCallback)(const char *p_method_utf8, const char *p_args_json_utf8);

GODOT_UWP_API void godot_uwp_set_log_callback(GodotUwpLogCallback p_callback);
GODOT_UWP_API void godot_uwp_set_swap_chain_panel(void *p_panel_native_iunknown);
GODOT_UWP_API void godot_uwp_set_ui_dispatcher(GodotUwpUiDispatchFunc p_dispatch);

GODOT_UWP_API int godot_uwp_engine_setup(int p_argc, char **p_argv);
GODOT_UWP_API int godot_uwp_engine_start(void);
GODOT_UWP_API int godot_uwp_engine_iteration(void);
GODOT_UWP_API void godot_uwp_engine_shutdown(void);

GODOT_UWP_API void godot_uwp_notify_panel_resize(int p_width_px, int p_height_px);
GODOT_UWP_API void godot_uwp_set_composition_scale(float p_scale_x, float p_scale_y);

// Input injection. Coordinates are physical pixels relative to the panel.
// p_button uses Godot MouseButton values (1=L, 2=R, 3=M, ...).
// p_win_vk is a Windows virtual-key code (mapped to Godot keys internally).
GODOT_UWP_API void godot_uwp_inject_mouse_button(int p_button, int p_pressed, float p_x, float p_y, int p_double_click);
GODOT_UWP_API void godot_uwp_inject_mouse_motion(float p_x, float p_y, float p_rel_x, float p_rel_y);
GODOT_UWP_API void godot_uwp_inject_mouse_wheel(float p_x, float p_y, float p_delta_x, float p_delta_y);
GODOT_UWP_API void godot_uwp_inject_key(unsigned int p_win_vk, int p_pressed, int p_echo, unsigned int p_unicode);

// ----------------------------------------------------------------------
// Host <-> engine JSON message bus.
// GDScript side: the "UWPHost" singleton (send_to_host / register_handler).
// ----------------------------------------------------------------------

// Installs the engine->host message handler. Install BEFORE engine start so
// messages emitted during script _ready are not dropped. Pass null to clear.
GODOT_UWP_API void godot_uwp_set_host_message_callback(GodotUwpHostMsgCallback p_callback);

// Provides the synchronous JSON reply for the message currently being
// delivered to the host callback. Only valid DURING that callback.
GODOT_UWP_API void godot_uwp_set_call_return(const char *p_json_utf8);

// Invokes a GDScript handler registered via UWPHost.register_handler.
// MUST be called on the engine thread. Returns 0 when the engine is not
// running; 1 otherwise. *r_ret_json_utf8 receives the handler's JSON return
// (or null) — free it with godot_uwp_free_string.
GODOT_UWP_API int godot_uwp_call_engine(const char *p_method_utf8, const char *p_args_json_utf8, char **r_ret_json_utf8);

// Frees a string returned by godot_uwp_call_engine.
GODOT_UWP_API void godot_uwp_free_string(char *p_str);

#endif // GODOT_UWP_EMBED_ENABLED
