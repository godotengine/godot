/*************************************************************************/
/*  library_godot_display.js                                             */
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

const GodotDisplayVK = {

	$GodotDisplayVK__deps: ['$GodotRuntime', '$GodotConfig', '$GodotEventListeners'],
	$GodotDisplayVK__postset: 'GodotOS.atexit(function(resolve, reject) { GodotDisplayVK.clear(); resolve(); });',
	$GodotDisplayVK: {
		textinput: null,
		textarea: null,

		available: function () {
			return GodotConfig.virtual_keyboard && 'ontouchstart' in window;
		},

		init: function (input_cb) {
			function create(what) {
				const elem = document.createElement(what);
				elem.style.display = 'none';
				elem.style.position = 'absolute';
				elem.style.zIndex = '-1';
				elem.style.background = 'transparent';
				elem.style.padding = '0px';
				elem.style.margin = '0px';
				elem.style.overflow = 'hidden';
				elem.style.width = '0px';
				elem.style.height = '0px';
				elem.style.border = '0px';
				elem.style.outline = 'none';
				elem.readonly = true;
				elem.disabled = true;
				GodotEventListeners.add(elem, 'input', function (evt) {
					const c_str = GodotRuntime.allocString(elem.value);
					input_cb(c_str, elem.selectionEnd);
					GodotRuntime.free(c_str);
				}, false);
				GodotEventListeners.add(elem, 'blur', function (evt) {
					elem.style.display = 'none';
					elem.readonly = true;
					elem.disabled = true;
				}, false);
				GodotConfig.canvas.insertAdjacentElement('beforebegin', elem);
				return elem;
			}
			GodotDisplayVK.textinput = create('input');
			GodotDisplayVK.textarea = create('textarea');
			GodotDisplayVK.updateSize();
		},
		show: function (text, multiline, start, end) {
			if (!GodotDisplayVK.textinput || !GodotDisplayVK.textarea) {
				return;
			}
			if (GodotDisplayVK.textinput.style.display !== '' || GodotDisplayVK.textarea.style.display !== '') {
				GodotDisplayVK.hide();
			}
			GodotDisplayVK.updateSize();
			const elem = multiline ? GodotDisplayVK.textarea : GodotDisplayVK.textinput;
			elem.readonly = false;
			elem.disabled = false;
			elem.value = text;
			elem.style.display = 'block';
			elem.focus();
			elem.setSelectionRange(start, end);
		},
		hide: function () {
			if (!GodotDisplayVK.textinput || !GodotDisplayVK.textarea) {
				return;
			}
			[GodotDisplayVK.textinput, GodotDisplayVK.textarea].forEach(function (elem) {
				elem.blur();
				elem.style.display = 'none';
				elem.value = '';
			});
		},
		updateSize: function () {
			if (!GodotDisplayVK.textinput || !GodotDisplayVK.textarea) {
				return;
			}
			const rect = GodotConfig.canvas.getBoundingClientRect();
			function update(elem) {
				elem.style.left = `${rect.left}px`;
				elem.style.top = `${rect.top}px`;
				elem.style.width = `${rect.width}px`;
				elem.style.height = `${rect.height}px`;
			}
			update(GodotDisplayVK.textinput);
			update(GodotDisplayVK.textarea);
		},
		clear: function () {
			if (GodotDisplayVK.textinput) {
				GodotDisplayVK.textinput.remove();
				GodotDisplayVK.textinput = null;
			}
			if (GodotDisplayVK.textarea) {
				GodotDisplayVK.textarea.remove();
				GodotDisplayVK.textarea = null;
			}
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayVK);

/*
 * Display server cursor helper.
 * Keeps track of cursor status and custom shapes.
 */
const GodotDisplayCursor = {
	$GodotDisplayCursor__deps: ['$GodotOS', '$GodotConfig'],
	$GodotDisplayCursor__postset: 'GodotOS.atexit(function(resolve, reject) { GodotDisplayCursor.clear(); resolve(); });',
	$GodotDisplayCursor: {
		shape: 'auto',
		visible: true,
		cursors: {},
		set_style: function (style) {
			GodotConfig.canvas.style.cursor = style;
		},
		set_shape: function (shape) {
			GodotDisplayCursor.shape = shape;
			let css = shape;
			if (shape in GodotDisplayCursor.cursors) {
				const c = GodotDisplayCursor.cursors[shape];
				css = `url("${c.url}") ${c.x} ${c.y}, auto`;
			}
			if (GodotDisplayCursor.visible) {
				GodotDisplayCursor.set_style(css);
			}
		},
		clear: function () {
			GodotDisplayCursor.set_style('');
			GodotDisplayCursor.shape = 'auto';
			GodotDisplayCursor.visible = true;
			Object.keys(GodotDisplayCursor.cursors).forEach(function (key) {
				URL.revokeObjectURL(GodotDisplayCursor.cursors[key]);
				delete GodotDisplayCursor.cursors[key];
			});
		},
		lockPointer: function () {
			const canvas = GodotConfig.canvas;
			if (canvas.requestPointerLock) {
				canvas.requestPointerLock();
			}
		},
		releasePointer: function () {
			if (document.exitPointerLock) {
				document.exitPointerLock();
			}
		},
		isPointerLocked: function () {
			return document.pointerLockElement === GodotConfig.canvas;
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayCursor);

const GodotDisplayScreen = {
	$GodotDisplayScreen__deps: ['$GodotConfig', '$GodotOS', '$GL', 'emscripten_webgl_get_current_context'],
	$GodotDisplayScreen: {
		desired_size: [0, 0],
		hidpi: true,
		getPixelRatio: function () {
			return GodotDisplayScreen.hidpi ? window.devicePixelRatio || 1 : 1;
		},
		isFullscreen: function () {
			const elem = document.fullscreenElement || document.mozFullscreenElement
				|| document.webkitFullscreenElement || document.msFullscreenElement;
			if (elem) {
				return elem === GodotConfig.canvas;
			}
			// But maybe knowing the element is not supported.
			return document.fullscreen || document.mozFullScreen
				|| document.webkitIsFullscreen;
		},
		hasFullscreen: function () {
			return document.fullscreenEnabled || document.mozFullScreenEnabled
				|| document.webkitFullscreenEnabled;
		},
		requestFullscreen: function () {
			if (!GodotDisplayScreen.hasFullscreen()) {
				return 1;
			}
			const canvas = GodotConfig.canvas;
			try {
				const promise = (canvas.requestFullscreen || canvas.msRequestFullscreen
					|| canvas.mozRequestFullScreen || canvas.mozRequestFullscreen
					|| canvas.webkitRequestFullscreen
				).call(canvas);
				// Some browsers (Safari) return undefined.
				// For the standard ones, we need to catch it.
				if (promise) {
					promise.catch(function () {
						// nothing to do.
					});
				}
			} catch (e) {
				return 1;
			}
			return 0;
		},
		exitFullscreen: function () {
			if (!GodotDisplayScreen.isFullscreen()) {
				return 0;
			}
			try {
				const promise = document.exitFullscreen();
				if (promise) {
					promise.catch(function () {
						// nothing to do.
					});
				}
			} catch (e) {
				return 1;
			}
			return 0;
		},
		_updateGL: function () {
			const gl_context_handle = _emscripten_webgl_get_current_context(); // eslint-disable-line no-undef
			const gl = GL.getContext(gl_context_handle);
			if (gl) {
				GL.resizeOffscreenFramebuffer(gl);
			}
		},
		updateSize: function () {
			const isFullscreen = GodotDisplayScreen.isFullscreen();
			const wantsFullWindow = GodotConfig.canvas_resize_policy === 2;
			const noResize = GodotConfig.canvas_resize_policy === 0;
			const wwidth = GodotDisplayScreen.desired_size[0];
			const wheight = GodotDisplayScreen.desired_size[1];
			const canvas = GodotConfig.canvas;
			let width = wwidth;
			let height = wheight;
			if (noResize) {
				// Don't resize canvas, just update GL if needed.
				if (canvas.width !== width || canvas.height !== height) {
					GodotDisplayScreen.desired_size = [canvas.width, canvas.height];
					GodotDisplayScreen._updateGL();
					return 1;
				}
				return 0;
			}
			const scale = GodotDisplayScreen.getPixelRatio();
			if (isFullscreen || wantsFullWindow) {
				// We need to match screen size.
				width = window.innerWidth * scale;
				height = window.innerHeight * scale;
			}
			const csw = `${width / scale}px`;
			const csh = `${height / scale}px`;
			if (canvas.style.width !== csw || canvas.style.height !== csh || canvas.width !== width || canvas.height !== height) {
				// Size doesn't match.
				// Resize canvas, set correct CSS pixel size, update GL.
				canvas.width = width;
				canvas.height = height;
				canvas.style.width = csw;
				canvas.style.height = csh;
				GodotDisplayScreen._updateGL();
				return 1;
			}
			return 0;
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayScreen);

/**
 * Display server interface.
 *
 * Exposes all the functions needed by DisplayServer implementation.
 */
const GodotDisplay = {
	$GodotDisplay__deps: ['$GodotConfig', '$GodotRuntime', '$GodotDisplayCursor', '$GodotEventListeners', '$GodotDisplayScreen', '$GodotDisplayVK'],
	$GodotDisplay: {
		window_icon: '',
		findDPI: function () {
			function testDPI(dpi) {
				return window.matchMedia(`(max-resolution: ${dpi}dpi)`).matches;
			}
			function bisect(low, high, func) {
				const mid = parseInt(((high - low) / 2) + low, 10);
				if (high - low <= 1) {
					return func(high) ? high : low;
				}
				if (func(mid)) {
					return bisect(low, mid, func);
				}
				return bisect(mid, high, func);
			}
			try {
				const dpi = bisect(0, 800, testDPI);
				return dpi >= 96 ? dpi : 96;
			} catch (e) {
				return 96;
			}
		},
	},

	// This is implemented as "glGetBufferSubData" in new emscripten versions.
	// Since we have to support older (pre 2.0.17) emscripten versions, we add this wrapper function instead.
	godot_js_display_glGetBufferSubData__sig: 'viiii',
	godot_js_display_glGetBufferSubData__deps: ['$GL', 'emscripten_webgl_get_current_context'],
	godot_js_display_glGetBufferSubData: function (target, offset, size, data) {
		const gl_context_handle = _emscripten_webgl_get_current_context(); // eslint-disable-line no-undef
		const gl = GL.getContext(gl_context_handle);
		if (gl) {
			gl.GLctx['getBufferSubData'](target, offset, HEAPU8, data, size);
		}
	},

	godot_js_display_is_swap_ok_cancel__sig: 'i',
	godot_js_display_is_swap_ok_cancel: function () {
		const win = (['Windows', 'Win64', 'Win32', 'WinCE']);
		const plat = navigator.platform || '';
		if (win.indexOf(plat) !== -1) {
			return 1;
		}
		return 0;
	},

	godot_js_display_alert__sig: 'vi',
	godot_js_display_alert: function (p_text) {
		window.alert(GodotRuntime.parseString(p_text)); // eslint-disable-line no-alert
	},

	godot_js_display_screen_dpi_get__sig: 'i',
	godot_js_display_screen_dpi_get: function () {
		return GodotDisplay.findDPI();
	},

	godot_js_display_pixel_ratio_get__sig: 'f',
	godot_js_display_pixel_ratio_get: function () {
		return GodotDisplayScreen.getPixelRatio();
	},

	godot_js_display_fullscreen_request__sig: 'i',
	godot_js_display_fullscreen_request: function () {
		return GodotDisplayScreen.requestFullscreen();
	},

	godot_js_display_fullscreen_exit__sig: 'i',
	godot_js_display_fullscreen_exit: function () {
		return GodotDisplayScreen.exitFullscreen();
	},

	godot_js_display_desired_size_set__sig: 'vii',
	godot_js_display_desired_size_set: function (width, height) {
		GodotDisplayScreen.desired_size = [width, height];
		GodotDisplayScreen.updateSize();
	},

	godot_js_display_size_update__sig: 'i',
	godot_js_display_size_update: function () {
		const updated = GodotDisplayScreen.updateSize();
		if (updated) {
			GodotDisplayVK.updateSize();
		}
		return updated;
	},

	godot_js_display_screen_size_get__sig: 'vii',
	godot_js_display_screen_size_get: function (width, height) {
		const scale = GodotDisplayScreen.getPixelRatio();
		GodotRuntime.setHeapValue(width, window.screen.width * scale, 'i32');
		GodotRuntime.setHeapValue(height, window.screen.height * scale, 'i32');
	},

	godot_js_display_window_size_get: function (p_width, p_height) {
		GodotRuntime.setHeapValue(p_width, GodotConfig.canvas.width, 'i32');
		GodotRuntime.setHeapValue(p_height, GodotConfig.canvas.height, 'i32');
	},

	godot_js_display_has_webgl__sig: 'ii',
	godot_js_display_has_webgl: function (p_version) {
		if (p_version !== 1 && p_version !== 2) {
			return false;
		}
		try {
			return !!document.createElement('canvas').getContext(p_version === 2 ? 'webgl2' : 'webgl');
		} catch (e) { /* Not available */ }
		return false;
	},

	/*
	 * Canvas
	 */
	godot_js_display_canvas_focus__sig: 'v',
	godot_js_display_canvas_focus: function () {
		GodotConfig.canvas.focus();
	},

	godot_js_display_canvas_is_focused__sig: 'i',
	godot_js_display_canvas_is_focused: function () {
		return document.activeElement === GodotConfig.canvas;
	},

	/*
	 * Touchscreen
	 */
	godot_js_display_touchscreen_is_available__sig: 'i',
	godot_js_display_touchscreen_is_available: function () {
		return 'ontouchstart' in window;
	},

	/*
	 * Clipboard
	 */
	godot_js_display_clipboard_set__sig: 'ii',
	godot_js_display_clipboard_set: function (p_text) {
		const text = GodotRuntime.parseString(p_text);
		if (!navigator.clipboard || !navigator.clipboard.writeText) {
			return 1;
		}
		navigator.clipboard.writeText(text).catch(function (e) {
			// Setting OS clipboard is only possible from an input callback.
			GodotRuntime.error('Setting OS clipboard is only possible from an input callback for the HTML5 plafrom. Exception:', e);
		});
		return 0;
	},

	godot_js_display_clipboard_get__sig: 'ii',
	godot_js_display_clipboard_get: function (callback) {
		const func = GodotRuntime.get_func(callback);
		try {
			navigator.clipboard.readText().then(function (result) {
				const ptr = GodotRuntime.allocString(result);
				func(ptr);
				GodotRuntime.free(ptr);
			}).catch(function (e) {
				// Fail graciously.
			});
		} catch (e) {
			// Fail graciously.
		}
	},

	/*
	 * Window
	 */
	godot_js_display_window_title_set__sig: 'vi',
	godot_js_display_window_title_set: function (p_data) {
		document.title = GodotRuntime.parseString(p_data);
	},

	godot_js_display_window_icon_set__sig: 'vii',
	godot_js_display_window_icon_set: function (p_ptr, p_len) {
		let link = document.getElementById('-gd-engine-icon');
		if (link === null) {
			link = document.createElement('link');
			link.rel = 'icon';
			link.id = '-gd-engine-icon';
			document.head.appendChild(link);
		}
		const old_icon = GodotDisplay.window_icon;
		const png = new Blob([GodotRuntime.heapSlice(HEAPU8, p_ptr, p_len)], { type: 'image/png' });
		GodotDisplay.window_icon = URL.createObjectURL(png);
		link.href = GodotDisplay.window_icon;
		if (old_icon) {
			URL.revokeObjectURL(old_icon);
		}
	},

	/*
	 * Cursor
	 */
	godot_js_display_cursor_set_visible__sig: 'vi',
	godot_js_display_cursor_set_visible: function (p_visible) {
		const visible = p_visible !== 0;
		if (visible === GodotDisplayCursor.visible) {
			return;
		}
		GodotDisplayCursor.visible = visible;
		if (visible) {
			GodotDisplayCursor.set_shape(GodotDisplayCursor.shape);
		} else {
			GodotDisplayCursor.set_style('none');
		}
	},

	godot_js_display_cursor_is_hidden__sig: 'i',
	godot_js_display_cursor_is_hidden: function () {
		return !GodotDisplayCursor.visible;
	},

	godot_js_display_cursor_set_shape__sig: 'vi',
	godot_js_display_cursor_set_shape: function (p_string) {
		GodotDisplayCursor.set_shape(GodotRuntime.parseString(p_string));
	},

	godot_js_display_cursor_set_custom_shape__sig: 'viiiii',
	godot_js_display_cursor_set_custom_shape: function (p_shape, p_ptr, p_len, p_hotspot_x, p_hotspot_y) {
		const shape = GodotRuntime.parseString(p_shape);
		const old_shape = GodotDisplayCursor.cursors[shape];
		if (p_len > 0) {
			const png = new Blob([GodotRuntime.heapSlice(HEAPU8, p_ptr, p_len)], { type: 'image/png' });
			const url = URL.createObjectURL(png);
			GodotDisplayCursor.cursors[shape] = {
				url: url,
				x: p_hotspot_x,
				y: p_hotspot_y,
			};
		} else {
			delete GodotDisplayCursor.cursors[shape];
		}
		if (shape === GodotDisplayCursor.shape) {
			GodotDisplayCursor.set_shape(GodotDisplayCursor.shape);
		}
		if (old_shape) {
			URL.revokeObjectURL(old_shape.url);
		}
	},

	godot_js_display_cursor_lock_set__sig: 'vi',
	godot_js_display_cursor_lock_set: function (p_lock) {
		if (p_lock) {
			GodotDisplayCursor.lockPointer();
		} else {
			GodotDisplayCursor.releasePointer();
		}
	},

	godot_js_display_cursor_is_locked__sig: 'i',
	godot_js_display_cursor_is_locked: function () {
		return GodotDisplayCursor.isPointerLocked() ? 1 : 0;
	},

	/*
	 * Listeners
	 */
	godot_js_display_fullscreen_cb__sig: 'vi',
	godot_js_display_fullscreen_cb: function (callback) {
		const canvas = GodotConfig.canvas;
		const func = GodotRuntime.get_func(callback);
		function change_cb(evt) {
			if (evt.target === canvas) {
				func(GodotDisplayScreen.isFullscreen());
			}
		}
		GodotEventListeners.add(document, 'fullscreenchange', change_cb, false);
		GodotEventListeners.add(document, 'mozfullscreenchange', change_cb, false);
		GodotEventListeners.add(document, 'webkitfullscreenchange', change_cb, false);
	},

	godot_js_display_window_blur_cb__sig: 'vi',
	godot_js_display_window_blur_cb: function (callback) {
		const func = GodotRuntime.get_func(callback);
		GodotEventListeners.add(window, 'blur', function () {
			func();
		}, false);
	},

	godot_js_display_notification_cb__sig: 'viiiii',
	godot_js_display_notification_cb: function (callback, p_enter, p_exit, p_in, p_out) {
		const canvas = GodotConfig.canvas;
		const func = GodotRuntime.get_func(callback);
		const notif = [p_enter, p_exit, p_in, p_out];
		['mouseover', 'mouseleave', 'focus', 'blur'].forEach(function (evt_name, idx) {
			GodotEventListeners.add(canvas, evt_name, function () {
				func(notif[idx]);
			}, true);
		});
	},

	godot_js_display_setup_canvas__sig: 'viiii',
	godot_js_display_setup_canvas: function (p_width, p_height, p_fullscreen, p_hidpi) {
		const canvas = GodotConfig.canvas;
		GodotEventListeners.add(canvas, 'contextmenu', function (ev) {
			ev.preventDefault();
		}, false);
		GodotEventListeners.add(canvas, 'webglcontextlost', function (ev) {
			alert('WebGL context lost, please reload the page'); // eslint-disable-line no-alert
			ev.preventDefault();
		}, false);
		GodotDisplayScreen.hidpi = !!p_hidpi;
		switch (GodotConfig.canvas_resize_policy) {
		case 0: // None
			GodotDisplayScreen.desired_size = [canvas.width, canvas.height];
			break;
		case 1: // Project
			GodotDisplayScreen.desired_size = [p_width, p_height];
			break;
		default: // Full window
			// Ensure we display in the right place, the size will be handled by updateSize
			canvas.style.position = 'absolute';
			canvas.style.top = 0;
			canvas.style.left = 0;
			break;
		}
		GodotDisplayScreen.updateSize();
		if (p_fullscreen) {
			GodotDisplayScreen.requestFullscreen();
		}
	},

	/*
	 * Virtual Keyboard
	 */
	godot_js_display_vk_show__sig: 'viiii',
	godot_js_display_vk_show: function (p_text, p_multiline, p_start, p_end) {
		const text = GodotRuntime.parseString(p_text);
		const start = p_start > 0 ? p_start : 0;
		const end = p_end > 0 ? p_end : start;
		GodotDisplayVK.show(text, p_multiline, start, end);
	},

	godot_js_display_vk_hide__sig: 'v',
	godot_js_display_vk_hide: function () {
		GodotDisplayVK.hide();
	},

	godot_js_display_vk_available__sig: 'i',
	godot_js_display_vk_available: function () {
		return GodotDisplayVK.available();
	},

	godot_js_display_vk_cb__sig: 'vi',
	godot_js_display_vk_cb: function (p_input_cb) {
		const input_cb = GodotRuntime.get_func(p_input_cb);
		if (GodotDisplayVK.available()) {
			GodotDisplayVK.init(input_cb);
		}
	},
};

autoAddDeps(GodotDisplay, '$GodotDisplay');
mergeInto(LibraryManager.library, GodotDisplay);
