/**************************************************************************/
/*  library_godot_webgl2.js                                               */
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

const GodotWebGL2 = {
	$GodotWebGL2__deps: ['$GL', '$GodotRuntime'],
	$GodotWebGL2: {},

	// This is implemented as "glGetBufferSubData" in new emscripten versions.
	// Since we have to support older (pre 2.0.17) emscripten versions, we add this wrapper function instead.
	godot_webgl2_glGetBufferSubData__proxy: 'sync',
	godot_webgl2_glGetBufferSubData__sig: 'vippp',
	godot_webgl2_glGetBufferSubData__deps: ['$GL', 'emscripten_webgl_get_current_context'],
	godot_webgl2_glGetBufferSubData: function (target, offset, size, data) {
		const gl_context_handle = _emscripten_webgl_get_current_context();
		const gl = GL.getContext(gl_context_handle);
		if (gl) {
			gl.GLctx['getBufferSubData'](target, offset, HEAPU8, data, size);
		}
	},

	godot_webgl2_glFramebufferTextureMultiviewOVR__deps: ['emscripten_webgl_get_current_context'],
	godot_webgl2_glFramebufferTextureMultiviewOVR__proxy: 'sync',
	godot_webgl2_glFramebufferTextureMultiviewOVR__sig: 'viiiiii',
	godot_webgl2_glFramebufferTextureMultiviewOVR: function (target, attachment, texture, level, base_view_index, num_views) {
		const context = GL.currentContext;
		if (typeof context.multiviewExt === 'undefined') {
			const /** OVR_multiview2 */ ext = context.GLctx.getExtension('OVR_multiview2');
			if (!ext) {
				GodotRuntime.error('Trying to call glFramebufferTextureMultiviewOVR() without the OVR_multiview2 extension');
				return;
			}
			context.multiviewExt = ext;
		}
		const /** OVR_multiview2 */ ext = context.multiviewExt;
		ext.framebufferTextureMultiviewOVR(target, attachment, GL.textures[texture], level, base_view_index, num_views);
	},

	godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR__deps: ['emscripten_webgl_get_current_context'],
	godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR__proxy: 'sync',
	godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR__sig: 'viiiiiii',
	godot_webgl2_glFramebufferTextureMultisampleMultiviewOVR: function (target, attachment, texture, level, samples, base_view_index, num_views) {
		const context = GL.currentContext;
		if (typeof context.oculusMultiviewExt === 'undefined') {
			const /** OCULUS_multiview */ ext = context.GLctx.getExtension('OCULUS_multiview');
			if (!ext) {
				GodotRuntime.error('Trying to call glFramebufferTextureMultisampleMultiviewOVR() without the OCULUS_multiview extension');
				return;
			}
			context.oculusMultiviewExt = ext;
		}
		const /** OCULUS_multiview */ ext = context.oculusMultiviewExt;
		ext.framebufferTextureMultisampleMultiviewOVR(target, attachment, GL.textures[texture], level, samples, base_view_index, num_views);
	},
};

autoAddDeps(GodotWebGL2, '$GodotWebGL2');
mergeInto(LibraryManager.library, GodotWebGL2);
