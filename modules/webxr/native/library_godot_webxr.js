/*************************************************************************/
/*  library_godot_webxr.js                                               */
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
const GodotWebXR = {

	$GodotWebXR__deps: ['$Browser', '$GL', '$GodotRuntime'],
	$GodotWebXR: {
		gl: null,

		session: null,
		space: null,
		frame: null,
		pose: null,

		// Monkey-patch the requestAnimationFrame() used by Emscripten for the main
		// loop, so that we can swap it out for XRSession.requestAnimationFrame()
		// when an XR session is started.
		orig_requestAnimationFrame: null,
		requestAnimationFrame: (callback) => {
			if (GodotWebXR.session && GodotWebXR.space) {
				const onFrame = function (time, frame) {
					GodotWebXR.frame = frame;
					GodotWebXR.pose = frame.getViewerPose(GodotWebXR.space);
					callback(time);
					GodotWebXR.frame = null;
					GodotWebXR.pose = null;
				};
				GodotWebXR.session.requestAnimationFrame(onFrame);
			} else {
				GodotWebXR.orig_requestAnimationFrame(callback);
			}
		},
		monkeyPatchRequestAnimationFrame: (enable) => {
			if (GodotWebXR.orig_requestAnimationFrame === null) {
				GodotWebXR.orig_requestAnimationFrame = Browser.requestAnimationFrame;
			}
			Browser.requestAnimationFrame = enable
				? GodotWebXR.requestAnimationFrame : GodotWebXR.orig_requestAnimationFrame;
		},
		pauseResumeMainLoop: () => {
			// Once both GodotWebXR.session and GodotWebXR.space are set or
			// unset, our monkey-patched requestAnimationFrame() should be
			// enabled or disabled. When using the WebXR API Emulator, this
			// gets picked up automatically, however, in the Oculus Browser
			// on the Quest, we need to pause and resume the main loop.
			Browser.mainLoop.pause();
			window.setTimeout(function () {
				Browser.mainLoop.resume();
			}, 0);
		},

		// Some custom WebGL code for blitting our eye textures to the
		// framebuffer we get from WebXR.
		shaderProgram: null,
		programInfo: null,
		buffer: null,
		// Vertex shader source.
		vsSource: `
			const vec2 scale = vec2(0.5, 0.5);
			attribute vec4 aVertexPosition;

			varying highp vec2 vTextureCoord;

			void main () {
				gl_Position = aVertexPosition;
				vTextureCoord = aVertexPosition.xy * scale + scale;
			}
		`,
		// Fragment shader source.
		fsSource: `
			varying highp vec2 vTextureCoord;

			uniform sampler2D uSampler;

			void main() {
				gl_FragColor = texture2D(uSampler, vTextureCoord);
			}
		`,

		initShaderProgram: (gl, vsSource, fsSource) => {
			const vertexShader = GodotWebXR.loadShader(gl, gl.VERTEX_SHADER, vsSource);
			const fragmentShader = GodotWebXR.loadShader(gl, gl.FRAGMENT_SHADER, fsSource);

			const shaderProgram = gl.createProgram();
			gl.attachShader(shaderProgram, vertexShader);
			gl.attachShader(shaderProgram, fragmentShader);
			gl.linkProgram(shaderProgram);

			if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
				GodotRuntime.error(`Unable to initialize the shader program: ${gl.getProgramInfoLog(shaderProgram)}`);
				return null;
			}

			return shaderProgram;
		},
		loadShader: (gl, type, source) => {
			const shader = gl.createShader(type);
			gl.shaderSource(shader, source);
			gl.compileShader(shader);

			if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
				GodotRuntime.error(`An error occurred compiling the shader: ${gl.getShaderInfoLog(shader)}`);
				gl.deleteShader(shader);
				return null;
			}

			return shader;
		},
		initBuffer: (gl) => {
			const positionBuffer = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
			const positions = [
				-1.0, -1.0,
				1.0, -1.0,
				-1.0, 1.0,
				1.0, 1.0,
			];
			gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
			return positionBuffer;
		},
		blitTexture: (gl, texture) => {
			if (GodotWebXR.shaderProgram === null) {
				GodotWebXR.shaderProgram = GodotWebXR.initShaderProgram(gl, GodotWebXR.vsSource, GodotWebXR.fsSource);
				GodotWebXR.programInfo = {
					program: GodotWebXR.shaderProgram,
					attribLocations: {
						vertexPosition: gl.getAttribLocation(GodotWebXR.shaderProgram, 'aVertexPosition'),
					},
					uniformLocations: {
						uSampler: gl.getUniformLocation(GodotWebXR.shaderProgram, 'uSampler'),
					},
				};
				GodotWebXR.buffer = GodotWebXR.initBuffer(gl);
			}

			const orig_program = gl.getParameter(gl.CURRENT_PROGRAM);
			gl.useProgram(GodotWebXR.shaderProgram);

			gl.bindBuffer(gl.ARRAY_BUFFER, GodotWebXR.buffer);
			gl.vertexAttribPointer(GodotWebXR.programInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
			gl.enableVertexAttribArray(GodotWebXR.programInfo.attribLocations.vertexPosition);

			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gl.TEXTURE_2D, texture);
			gl.uniform1i(GodotWebXR.programInfo.uniformLocations.uSampler, 0);

			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

			// Restore state.
			gl.bindTexture(gl.TEXTURE_2D, null);
			gl.disableVertexAttribArray(GodotWebXR.programInfo.attribLocations.vertexPosition);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
			gl.useProgram(orig_program);
		},

		// Holds the controllers list between function calls.
		controllers: [],

		// Updates controllers array, where the left hand (or sole tracker) is
		// the first element, and the right hand is the second element, and any
		// others placed at the 3rd position and up.
		sampleControllers: () => {
			if (!GodotWebXR.session || !GodotWebXR.frame) {
				return;
			}

			let other_index = 2;
			const controllers = [];
			GodotWebXR.session.inputSources.forEach((input_source) => {
				if (input_source.targetRayMode === 'tracked-pointer') {
					if (input_source.handedness === 'right') {
						controllers[1] = input_source;
					} else if (input_source.handedness === 'left' || !controllers[0]) {
						controllers[0] = input_source;
					}
				} else {
					controllers[other_index++] = input_source;
				}
			});
			GodotWebXR.controllers = controllers;
		},

		getControllerId: (input_source) => GodotWebXR.controllers.indexOf(input_source),
	},

	godot_webxr_is_supported__proxy: 'sync',
	godot_webxr_is_supported__sig: 'i',
	godot_webxr_is_supported: function () {
		return !!navigator.xr;
	},

	godot_webxr_is_session_supported__proxy: 'sync',
	godot_webxr_is_session_supported__sig: 'vii',
	godot_webxr_is_session_supported: function (p_session_mode, p_callback) {
		const session_mode = GodotRuntime.parseString(p_session_mode);
		const cb = GodotRuntime.get_func(p_callback);
		if (navigator.xr) {
			navigator.xr.isSessionSupported(session_mode).then(function (supported) {
				const c_str = GodotRuntime.allocString(session_mode);
				cb(c_str, supported ? 1 : 0);
				GodotRuntime.free(c_str);
			});
		} else {
			const c_str = GodotRuntime.allocString(session_mode);
			cb(c_str, 0);
			GodotRuntime.free(c_str);
		}
	},

	godot_webxr_initialize__deps: ['emscripten_webgl_get_current_context'],
	godot_webxr_initialize__proxy: 'sync',
	godot_webxr_initialize__sig: 'viiiiiiiiii',
	godot_webxr_initialize: function (p_session_mode, p_required_features, p_optional_features, p_requested_reference_spaces, p_on_session_started, p_on_session_ended, p_on_session_failed, p_on_controller_changed, p_on_input_event, p_on_simple_event) {
		GodotWebXR.monkeyPatchRequestAnimationFrame(true);

		const session_mode = GodotRuntime.parseString(p_session_mode);
		const required_features = GodotRuntime.parseString(p_required_features).split(',').map((s) => s.trim()).filter((s) => s !== '');
		const optional_features = GodotRuntime.parseString(p_optional_features).split(',').map((s) => s.trim()).filter((s) => s !== '');
		const requested_reference_space_types = GodotRuntime.parseString(p_requested_reference_spaces).split(',').map((s) => s.trim());
		const onstarted = GodotRuntime.get_func(p_on_session_started);
		const onended = GodotRuntime.get_func(p_on_session_ended);
		const onfailed = GodotRuntime.get_func(p_on_session_failed);
		const oncontroller = GodotRuntime.get_func(p_on_controller_changed);
		const oninputevent = GodotRuntime.get_func(p_on_input_event);
		const onsimpleevent = GodotRuntime.get_func(p_on_simple_event);

		const session_init = {};
		if (required_features.length > 0) {
			session_init['requiredFeatures'] = required_features;
		}
		if (optional_features.length > 0) {
			session_init['optionalFeatures'] = optional_features;
		}

		navigator.xr.requestSession(session_mode, session_init).then(function (session) {
			GodotWebXR.session = session;

			session.addEventListener('end', function (evt) {
				onended();
			});

			session.addEventListener('inputsourceschange', function (evt) {
				let controller_changed = false;
				[evt.added, evt.removed].forEach((lst) => {
					lst.forEach((input_source) => {
						if (input_source.targetRayMode === 'tracked-pointer') {
							controller_changed = true;
						}
					});
				});
				if (controller_changed) {
					oncontroller();
				}
			});

			['selectstart', 'select', 'selectend', 'squeezestart', 'squeeze', 'squeezeend'].forEach((input_event) => {
				session.addEventListener(input_event, function (evt) {
					const c_str = GodotRuntime.allocString(input_event);
					oninputevent(c_str, GodotWebXR.getControllerId(evt.inputSource));
					GodotRuntime.free(c_str);
				});
			});

			session.addEventListener('visibilitychange', function (evt) {
				const c_str = GodotRuntime.allocString('visibility_state_changed');
				onsimpleevent(c_str);
				GodotRuntime.free(c_str);
			});

			const gl_context_handle = _emscripten_webgl_get_current_context(); // eslint-disable-line no-undef
			const gl = GL.getContext(gl_context_handle).GLctx;
			GodotWebXR.gl = gl;

			gl.makeXRCompatible().then(function () {
				session.updateRenderState({
					baseLayer: new XRWebGLLayer(session, gl),
				});

				function onReferenceSpaceSuccess(reference_space, reference_space_type) {
					GodotWebXR.space = reference_space;

					// Using reference_space.addEventListener() crashes when
					// using the polyfill with the WebXR Emulator extension,
					// so we set the event property instead.
					reference_space.onreset = function (evt) {
						const c_str = GodotRuntime.allocString('reference_space_reset');
						onsimpleevent(c_str);
						GodotRuntime.free(c_str);
					};

					// Now that both GodotWebXR.session and GodotWebXR.space are
					// set, we need to pause and resume the main loop for the XR
					// main loop to kick in.
					GodotWebXR.pauseResumeMainLoop();

					// Call in setTimeout() so that errors in the onstarted()
					// callback don't bubble up here and cause Godot to try the
					// next reference space.
					window.setTimeout(function () {
						const c_str = GodotRuntime.allocString(reference_space_type);
						onstarted(c_str);
						GodotRuntime.free(c_str);
					}, 0);
				}

				function requestReferenceSpace() {
					const reference_space_type = requested_reference_space_types.shift();
					session.requestReferenceSpace(reference_space_type)
						.then((refSpace) => {
							onReferenceSpaceSuccess(refSpace, reference_space_type);
						})
						.catch(() => {
							if (requested_reference_space_types.length === 0) {
								const c_str = GodotRuntime.allocString('Unable to get any of the requested reference space types');
								onfailed(c_str);
								GodotRuntime.free(c_str);
							} else {
								requestReferenceSpace();
							}
						});
				}

				requestReferenceSpace();
			}).catch(function (error) {
				const c_str = GodotRuntime.allocString(`Unable to make WebGL context compatible with WebXR: ${error}`);
				onfailed(c_str);
				GodotRuntime.free(c_str);
			});
		}).catch(function (error) {
			const c_str = GodotRuntime.allocString(`Unable to start session: ${error}`);
			onfailed(c_str);
			GodotRuntime.free(c_str);
		});
	},

	godot_webxr_uninitialize__proxy: 'sync',
	godot_webxr_uninitialize__sig: 'v',
	godot_webxr_uninitialize: function () {
		if (GodotWebXR.session) {
			GodotWebXR.session.end()
				// Prevent exception when session has already ended.
				.catch((e) => { });
		}

		GodotWebXR.session = null;
		GodotWebXR.space = null;
		GodotWebXR.frame = null;
		GodotWebXR.pose = null;

		// Disable the monkey-patched window.requestAnimationFrame() and
		// pause/restart the main loop to activate it on all platforms.
		GodotWebXR.monkeyPatchRequestAnimationFrame(false);
		GodotWebXR.pauseResumeMainLoop();
	},

	godot_webxr_get_view_count__proxy: 'sync',
	godot_webxr_get_view_count__sig: 'i',
	godot_webxr_get_view_count: function () {
		if (!GodotWebXR.session || !GodotWebXR.pose) {
			return 0;
		}
		return GodotWebXR.pose.views.length;
	},

	godot_webxr_get_render_targetsize__proxy: 'sync',
	godot_webxr_get_render_targetsize__sig: 'i',
	godot_webxr_get_render_targetsize: function () {
		if (!GodotWebXR.session || !GodotWebXR.pose) {
			return 0;
		}

		const glLayer = GodotWebXR.session.renderState.baseLayer;
		const view = GodotWebXR.pose.views[0];
		const viewport = glLayer.getViewport(view);

		const buf = GodotRuntime.malloc(2 * 4);
		GodotRuntime.setHeapValue(buf + 0, viewport.width, 'i32');
		GodotRuntime.setHeapValue(buf + 4, viewport.height, 'i32');
		return buf;
	},

	godot_webxr_get_transform_for_eye__proxy: 'sync',
	godot_webxr_get_transform_for_eye__sig: 'ii',
	godot_webxr_get_transform_for_eye: function (p_eye) {
		if (!GodotWebXR.session || !GodotWebXR.pose) {
			return 0;
		}

		const views = GodotWebXR.pose.views;
		let matrix;
		if (p_eye === 0) {
			matrix = GodotWebXR.pose.transform.matrix;
		} else {
			matrix = views[p_eye - 1].transform.matrix;
		}
		const buf = GodotRuntime.malloc(16 * 4);
		for (let i = 0; i < 16; i++) {
			GodotRuntime.setHeapValue(buf + (i * 4), matrix[i], 'float');
		}
		return buf;
	},

	godot_webxr_get_projection_for_eye__proxy: 'sync',
	godot_webxr_get_projection_for_eye__sig: 'ii',
	godot_webxr_get_projection_for_eye: function (p_eye) {
		if (!GodotWebXR.session || !GodotWebXR.pose) {
			return 0;
		}

		const view_index = (p_eye === 2 /* ARVRInterface::EYE_RIGHT */) ? 1 : 0;
		const matrix = GodotWebXR.pose.views[view_index].projectionMatrix;
		const buf = GodotRuntime.malloc(16 * 4);
		for (let i = 0; i < 16; i++) {
			GodotRuntime.setHeapValue(buf + (i * 4), matrix[i], 'float');
		}
		return buf;
	},

	godot_webxr_commit_for_eye__proxy: 'sync',
	godot_webxr_commit_for_eye__sig: 'vii',
	godot_webxr_commit_for_eye: function (p_eye, p_texture_id) {
		if (!GodotWebXR.session || !GodotWebXR.pose) {
			return;
		}

		const view_index = (p_eye === 2 /* ARVRInterface::EYE_RIGHT */) ? 1 : 0;
		const glLayer = GodotWebXR.session.renderState.baseLayer;
		const view = GodotWebXR.pose.views[view_index];
		const viewport = glLayer.getViewport(view);
		const gl = GodotWebXR.gl;

		const orig_framebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
		const orig_viewport = gl.getParameter(gl.VIEWPORT);

		// Bind to WebXR's framebuffer.
		gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
		gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);

		GodotWebXR.blitTexture(gl, GL.textures[p_texture_id]);

		// Restore state.
		gl.bindFramebuffer(gl.FRAMEBUFFER, orig_framebuffer);
		gl.viewport(orig_viewport[0], orig_viewport[1], orig_viewport[2], orig_viewport[3]);
	},

	godot_webxr_sample_controller_data__proxy: 'sync',
	godot_webxr_sample_controller_data__sig: 'v',
	godot_webxr_sample_controller_data: function () {
		GodotWebXR.sampleControllers();
	},

	godot_webxr_get_controller_count__proxy: 'sync',
	godot_webxr_get_controller_count__sig: 'i',
	godot_webxr_get_controller_count: function () {
		if (!GodotWebXR.session || !GodotWebXR.frame) {
			return 0;
		}
		return GodotWebXR.controllers.length;
	},

	godot_webxr_is_controller_connected__proxy: 'sync',
	godot_webxr_is_controller_connected__sig: 'ii',
	godot_webxr_is_controller_connected: function (p_controller) {
		if (!GodotWebXR.session || !GodotWebXR.frame) {
			return false;
		}
		return !!GodotWebXR.controllers[p_controller];
	},

	godot_webxr_get_controller_transform__proxy: 'sync',
	godot_webxr_get_controller_transform__sig: 'ii',
	godot_webxr_get_controller_transform: function (p_controller) {
		if (!GodotWebXR.session || !GodotWebXR.frame) {
			return 0;
		}

		const controller = GodotWebXR.controllers[p_controller];
		if (!controller) {
			return 0;
		}

		const frame = GodotWebXR.frame;
		const space = GodotWebXR.space;

		const pose = frame.getPose(controller.targetRaySpace, space);
		if (!pose) {
			// This can mean that the controller lost tracking.
			return 0;
		}
		const matrix = pose.transform.matrix;

		const buf = GodotRuntime.malloc(16 * 4);
		for (let i = 0; i < 16; i++) {
			GodotRuntime.setHeapValue(buf + (i * 4), matrix[i], 'float');
		}
		return buf;
	},

	godot_webxr_get_controller_buttons__proxy: 'sync',
	godot_webxr_get_controller_buttons__sig: 'ii',
	godot_webxr_get_controller_buttons: function (p_controller) {
		if (GodotWebXR.controllers.length === 0) {
			return 0;
		}

		const controller = GodotWebXR.controllers[p_controller];
		if (!controller || !controller.gamepad) {
			return 0;
		}

		const button_count = controller.gamepad.buttons.length;

		const buf = GodotRuntime.malloc((button_count + 1) * 4);
		GodotRuntime.setHeapValue(buf, button_count, 'i32');
		for (let i = 0; i < button_count; i++) {
			GodotRuntime.setHeapValue(buf + 4 + (i * 4), controller.gamepad.buttons[i].value, 'float');
		}
		return buf;
	},

	godot_webxr_get_controller_axes__proxy: 'sync',
	godot_webxr_get_controller_axes__sig: 'ii',
	godot_webxr_get_controller_axes: function (p_controller) {
		if (GodotWebXR.controllers.length === 0) {
			return 0;
		}

		const controller = GodotWebXR.controllers[p_controller];
		if (!controller || !controller.gamepad) {
			return 0;
		}

		const axes_count = controller.gamepad.axes.length;

		const buf = GodotRuntime.malloc((axes_count + 1) * 4);
		GodotRuntime.setHeapValue(buf, axes_count, 'i32');
		for (let i = 0; i < axes_count; i++) {
			let value = controller.gamepad.axes[i];
			if (i === 1 || i === 3) {
				// Invert the Y-axis on thumbsticks and trackpads, in order to
				// match OpenXR and other XR platform SDKs.
				value *= -1.0;
			}
			GodotRuntime.setHeapValue(buf + 4 + (i * 4), value, 'float');
		}
		return buf;
	},

	godot_webxr_get_visibility_state__proxy: 'sync',
	godot_webxr_get_visibility_state__sig: 'i',
	godot_webxr_get_visibility_state: function () {
		if (!GodotWebXR.session || !GodotWebXR.session.visibilityState) {
			return 0;
		}

		return GodotRuntime.allocString(GodotWebXR.session.visibilityState);
	},

	godot_webxr_get_bounds_geometry__proxy: 'sync',
	godot_webxr_get_bounds_geometry__sig: 'i',
	godot_webxr_get_bounds_geometry: function () {
		if (!GodotWebXR.space || !GodotWebXR.space.boundsGeometry) {
			return 0;
		}

		const point_count = GodotWebXR.space.boundsGeometry.length;
		if (point_count === 0) {
			return 0;
		}

		const buf = GodotRuntime.malloc(((point_count * 3) + 1) * 4);
		GodotRuntime.setHeapValue(buf, point_count, 'i32');
		for (let i = 0; i < point_count; i++) {
			const point = GodotWebXR.space.boundsGeometry[i];
			GodotRuntime.setHeapValue(buf + ((i * 3) + 1) * 4, point.x, 'float');
			GodotRuntime.setHeapValue(buf + ((i * 3) + 2) * 4, point.y, 'float');
			GodotRuntime.setHeapValue(buf + ((i * 3) + 3) * 4, point.z, 'float');
		}

		return buf;
	},

};

autoAddDeps(GodotWebXR, '$GodotWebXR');
mergeInto(LibraryManager.library, GodotWebXR);
