/**************************************************************************/
/*  godot_webgl2.cpp                                                      */
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

#include "godot_webgl2.h"

#ifdef THREADS_ENABLED

#include "core/os/memory.h"
#include "core/templates/simple_type.h"

#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <pthread.h>

#include <cstring>
#include <type_traits>

pthread_t web_canvas_thread;

void setup_canvas_proxying(bool p_canvas_is_on_runtime) {
	// Forward calls to the thread with WebGL context.
	if (p_canvas_is_on_runtime) {
		// Canvas is on the runtime thread.
		web_canvas_thread = emscripten_main_runtime_thread_id();
	} else {
		// Offscren canvas is enabled and outside of the runtime thread.
		// We are assuming that canvas is on the same thread that created
		// the display server.
		web_canvas_thread = pthread_self();
	}
}

static _FORCE_INLINE_ bool is_canvas_thread() {
	return web_canvas_thread == pthread_self();
}

template <typename R, typename... P, typename... VarArgs>
static void proxy_canvas_sync(R (*p_func)(P...), VarArgs &&...p_args) {
	using FuncType = R (*)(P... args);

	// Declare pointers as const pointers to accept const references.
	void const *argptrs[1 + sizeof...(p_args)];

	// Store only pointers, as emscripten_proxy_sync waits for the function call to finish.
	argptrs[0] = &p_func;
	size_t index = 1;
	([&] {
		argptrs[index] = &p_args;
		index += 1;
	}(),
			...);

	emscripten_proxy_sync(
			emscripten_proxy_get_system_queue(),
			web_canvas_thread,
			[](void *p_userdata) {
				// Remove const from all stored pointers and add it when unpacking if needed.
				void **argptrs = static_cast<void **>(p_userdata);

				FuncType func = *static_cast<FuncType *>(argptrs[0]);
				size_t index = 1;
				// Dereference pointers after the lambda call, it allows references to bind directly
				// to the actual values.
				func(*[&] {
					// Cast it back to a pointer, while preserving original constness.
					using ArgMaybeConst = std::remove_reference_t<VarArgs>;
					ArgMaybeConst *arg = static_cast<ArgMaybeConst *>(argptrs[index]);
					index += 1;
					return arg;
				}()...);
			},
			argptrs);
}

template <typename R, typename... P, typename... VarArgs>
static void proxy_canvas_async(R (*p_func)(P...), VarArgs &&...p_args) {
	using FuncType = R (*)(P... args);
	constexpr size_t func_size = sizeof(FuncType);
	constexpr size_t buffer_size = func_size + (0 + ... + sizeof(GetSimpleTypeT<VarArgs>));

	// Allocate buffer for function pointer + all arguments.
	uint8_t *buffer = static_cast<uint8_t *>(memalloc(buffer_size));

	// Copy function pointer to the buffer.
	memcpy(buffer, &p_func, func_size);

	// Begin offset after stored function pointer.
	size_t offset = func_size;
	([&] {
		using ArgT = GetSimpleTypeT<VarArgs>;
		// Copy arguments to the buffer while invoking appropriate copy or move constructors if they exist.
		memnew_placement((buffer + offset), ArgT(std::forward<VarArgs>(p_args)));
		offset += sizeof(ArgT);
	}(),
			...);

	emscripten_proxy_async(
			emscripten_proxy_get_system_queue(),
			web_canvas_thread,
			[](void *p_userdata) {
				uint8_t *buffer = static_cast<uint8_t *>(p_userdata);
				FuncType func;
				// Copy function pointer back.
				memcpy(&func, buffer, func_size);

				size_t offset_call = func_size;
				// Dereference pointers after the lambda call, references will be bound to the copy inside the buffer.
				func(*[&] {
					using ArgMaybeConst = std::remove_reference_t<VarArgs>;
					using ArgT = GetSimpleTypeT<VarArgs>;
					// std::launder is needed to prevent compiler optimizations and
					// indicate that the value with type ArgT * actually exists here.
					ArgT *arg = std::launder(reinterpret_cast<ArgT *>(buffer + offset_call));
					offset_call += sizeof(ArgT);
					// Add const to the argument if it was passed with const, preserves correct semantics
					// and won't allow you to pass const argument to non-const function parameter.
					return static_cast<ArgMaybeConst *>(arg);
				}()...);

				size_t offset_dstr = func_size;
				([&] {
					using ArgT = GetSimpleTypeT<VarArgs>;
					// Call destructor on the created with memnew_placement values.
					std::launder(reinterpret_cast<ArgT *>(buffer + offset_dstr))->~ArgT();
					offset_dstr += sizeof(ArgT);
				}(),
						...);

				// Deallocate buffer.
				memfree(buffer);
			},
			buffer);
}

void godot_webgl2_glFramebufferTextureMultiviewOVRDirect(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint baseViewIndex, GLsizei numViews) {
	if (!is_canvas_thread()) {
		proxy_canvas_sync(godot_webgl2_glFramebufferTextureMultiviewOVR, target, attachment, texture, level, baseViewIndex, numViews);
		return;
	}
	godot_webgl2_glFramebufferTextureMultiviewOVR(target, attachment, texture, level, baseViewIndex, numViews);
}

void godot_webgl2_glFramebufferTextureMultisampleMultiviewOVRDirect(GLenum target, GLenum attachment, GLuint texture, GLint level, GLsizei samples, GLint baseViewIndex, GLsizei numViews) {
	if (!is_canvas_thread()) {
		proxy_canvas_sync(godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR, target, attachment, texture, level, samples, baseViewIndex, numViews);
		return;
	}
	godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR(target, attachment, texture, level, samples, baseViewIndex, numViews);
}

void godot_webgl2_glGetBufferSubDataDirect(GLenum target, GLintptr offset, GLsizeiptr size, GLvoid *data) {
	if (!is_canvas_thread()) {
		proxy_canvas_sync(godot_webgl2_wrapper_glGetBufferSubData, target, offset, size, data);
		return;
	}
	godot_webgl2_wrapper_glGetBufferSubData(target, offset, size, data);
}

#else

void setup_canvas_proxying(bool p_canvas_is_on_runtime) {}

void godot_webgl2_glFramebufferTextureMultiviewOVRDirect(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint baseViewIndex, GLsizei numViews) {
	godot_webgl2_glFramebufferTextureMultiviewOVR(target, attachment, texture, level, baseViewIndex, numViews);
}

void godot_webgl2_glFramebufferTextureMultisampleMultiviewOVRDirect(GLenum target, GLenum attachment, GLuint texture, GLint level, GLsizei samples, GLint baseViewIndex, GLsizei numViews) {
	godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR(target, attachment, texture, level, samples, baseViewIndex, numViews);
}

void godot_webgl2_glGetBufferSubDataDirect(GLenum target, GLintptr offset, GLsizeiptr size, GLvoid *data) {
	godot_webgl2_wrapper_glGetBufferSubData(target, offset, size, data);
}

#endif
