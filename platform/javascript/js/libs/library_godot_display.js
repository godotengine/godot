/*************************************************************************/
/*  library_godot_display.js                                             */
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

/*
 * Display Server listeners.
 * Keeps track of registered event listeners so it can remove them on shutdown.
 */
const GodotDisplayListeners = {
	$GodotDisplayListeners__deps: ['$GodotOS'],
	$GodotDisplayListeners__postset: 'GodotOS.atexit(function(resolve, reject) { GodotDisplayListeners.clear(); resolve(); });',
	$GodotDisplayListeners: {
		handlers: [],

		has: function (target, event, method, capture) {
			return GodotDisplayListeners.handlers.findIndex(function (e) {
				return e.target === target && e.event === event && e.method === method && e.capture === capture;
			}) !== -1;
		},

		add: function (target, event, method, capture) {
			if (GodotDisplayListeners.has(target, event, method, capture)) {
				return;
			}
			function Handler(p_target, p_event, p_method, p_capture) {
				this.target = p_target;
				this.event = p_event;
				this.method = p_method;
				this.capture = p_capture;
			}
			GodotDisplayListeners.handlers.push(new Handler(target, event, method, capture));
			target.addEventListener(event, method, capture);
		},

		clear: function () {
			GodotDisplayListeners.handlers.forEach(function (h) {
				h.target.removeEventListener(h.event, h.method, h.capture);
			});
			GodotDisplayListeners.handlers.length = 0;
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayListeners);

/*
 * Drag and drop handler.
 * This is pretty big, but basically detect dropped files on GodotConfig.canvas,
 * process them one by one (recursively for directories), and copies them to
 * the temporary FS path '/tmp/drop-[random]/' so it can be emitted as a godot
 * event (that requires a string array of paths).
 *
 * NOTE: The temporary files are removed after the callback. This means that
 * deferred callbacks won't be able to access the files.
 */
const GodotDisplayDragDrop = {
	$GodotDisplayDragDrop__deps: ['$FS', '$GodotFS'],
	$GodotDisplayDragDrop: {
		promises: [],
		pending_files: [],

		add_entry: function (entry) {
			if (entry.isDirectory) {
				GodotDisplayDragDrop.add_dir(entry);
			} else if (entry.isFile) {
				GodotDisplayDragDrop.add_file(entry);
			} else {
				GodotRuntime.error('Unrecognized entry...', entry);
			}
		},

		add_dir: function (entry) {
			GodotDisplayDragDrop.promises.push(new Promise(function (resolve, reject) {
				const reader = entry.createReader();
				reader.readEntries(function (entries) {
					for (let i = 0; i < entries.length; i++) {
						GodotDisplayDragDrop.add_entry(entries[i]);
					}
					resolve();
				});
			}));
		},

		add_file: function (entry) {
			GodotDisplayDragDrop.promises.push(new Promise(function (resolve, reject) {
				entry.file(function (file) {
					const reader = new FileReader();
					reader.onload = function () {
						const f = {
							'path': file.relativePath || file.webkitRelativePath,
							'name': file.name,
							'type': file.type,
							'size': file.size,
							'data': reader.result,
						};
						if (!f['path']) {
							f['path'] = f['name'];
						}
						GodotDisplayDragDrop.pending_files.push(f);
						resolve();
					};
					reader.onerror = function () {
						GodotRuntime.print('Error reading file');
						reject();
					};
					reader.readAsArrayBuffer(file);
				}, function (err) {
					GodotRuntime.print('Error!');
					reject();
				});
			}));
		},

		process: function (resolve, reject) {
			if (GodotDisplayDragDrop.promises.length === 0) {
				resolve();
				return;
			}
			GodotDisplayDragDrop.promises.pop().then(function () {
				setTimeout(function () {
					GodotDisplayDragDrop.process(resolve, reject);
				}, 0);
			});
		},

		_process_event: function (ev, callback) {
			ev.preventDefault();
			if (ev.dataTransfer.items) {
				// Use DataTransferItemList interface to access the file(s)
				for (let i = 0; i < ev.dataTransfer.items.length; i++) {
					const item = ev.dataTransfer.items[i];
					let entry = null;
					if ('getAsEntry' in item) {
						entry = item.getAsEntry();
					} else if ('webkitGetAsEntry' in item) {
						entry = item.webkitGetAsEntry();
					}
					if (entry) {
						GodotDisplayDragDrop.add_entry(entry);
					}
				}
			} else {
				GodotRuntime.error('File upload not supported');
			}
			new Promise(GodotDisplayDragDrop.process).then(function () {
				const DROP = `/tmp/drop-${parseInt(Math.random() * (1 << 30), 10)}/`;
				const drops = [];
				const files = [];
				FS.mkdir(DROP);
				GodotDisplayDragDrop.pending_files.forEach((elem) => {
					const path = elem['path'];
					GodotFS.copy_to_fs(DROP + path, elem['data']);
					let idx = path.indexOf('/');
					if (idx === -1) {
						// Root file
						drops.push(DROP + path);
					} else {
						// Subdir
						const sub = path.substr(0, idx);
						idx = sub.indexOf('/');
						if (idx < 0 && drops.indexOf(DROP + sub) === -1) {
							drops.push(DROP + sub);
						}
					}
					files.push(DROP + path);
				});
				GodotDisplayDragDrop.promises = [];
				GodotDisplayDragDrop.pending_files = [];
				callback(drops);
				const dirs = [DROP.substr(0, DROP.length - 1)];
				// Remove temporary files
				files.forEach(function (file) {
					FS.unlink(file);
					let dir = file.replace(DROP, '');
					let idx = dir.lastIndexOf('/');
					while (idx > 0) {
						dir = dir.substr(0, idx);
						if (dirs.indexOf(DROP + dir) === -1) {
							dirs.push(DROP + dir);
						}
						idx = dir.lastIndexOf('/');
					}
				});
				// Remove dirs.
				dirs.sort(function (a, b) {
					const al = (a.match(/\//g) || []).length;
					const bl = (b.match(/\//g) || []).length;
					if (al > bl) {
						return -1;
					} else if (al < bl) {
						return 1;
					}
					return 0;
				}).forEach(function (dir) {
					FS.rmdir(dir);
				});
			});
		},

		handler: function (callback) {
			return function (ev) {
				GodotDisplayDragDrop._process_event(ev, callback);
			};
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayDragDrop);

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
	},
};
mergeInto(LibraryManager.library, GodotDisplayCursor);

/*
 * Display Gamepad API helper.
 */
const GodotDisplayGamepads = {
	$GodotDisplayGamepads__deps: ['$GodotRuntime', '$GodotDisplayListeners'],
	$GodotDisplayGamepads: {
		samples: [],

		get_pads: function () {
			try {
				// Will throw in iframe when permission is denied.
				// Will throw/warn in the future for insecure contexts.
				// See https://github.com/w3c/gamepad/pull/120
				const pads = navigator.getGamepads();
				if (pads) {
					return pads;
				}
				return [];
			} catch (e) {
				return [];
			}
		},

		get_samples: function () {
			return GodotDisplayGamepads.samples;
		},

		get_sample: function (index) {
			const samples = GodotDisplayGamepads.samples;
			return index < samples.length ? samples[index] : null;
		},

		sample: function () {
			const pads = GodotDisplayGamepads.get_pads();
			const samples = [];
			for (let i = 0; i < pads.length; i++) {
				const pad = pads[i];
				if (!pad) {
					samples.push(null);
					continue;
				}
				const s = {
					standard: pad.mapping === 'standard',
					buttons: [],
					axes: [],
					connected: pad.connected,
				};
				for (let b = 0; b < pad.buttons.length; b++) {
					s.buttons.push(pad.buttons[b].value);
				}
				for (let a = 0; a < pad.axes.length; a++) {
					s.axes.push(pad.axes[a]);
				}
				samples.push(s);
			}
			GodotDisplayGamepads.samples = samples;
		},

		init: function (onchange) {
			GodotDisplayListeners.samples = [];
			function add(pad) {
				const guid = GodotDisplayGamepads.get_guid(pad);
				const c_id = GodotRuntime.allocString(pad.id);
				const c_guid = GodotRuntime.allocString(guid);
				onchange(pad.index, 1, c_id, c_guid);
				GodotRuntime.free(c_id);
				GodotRuntime.free(c_guid);
			}
			const pads = GodotDisplayGamepads.get_pads();
			for (let i = 0; i < pads.length; i++) {
				// Might be reserved space.
				if (pads[i]) {
					add(pads[i]);
				}
			}
			GodotDisplayListeners.add(window, 'gamepadconnected', function (evt) {
				add(evt.gamepad);
			}, false);
			GodotDisplayListeners.add(window, 'gamepaddisconnected', function (evt) {
				onchange(evt.gamepad.index, 0);
			}, false);
		},

		get_guid: function (pad) {
			if (pad.mapping) {
				return pad.mapping;
			}
			const ua = navigator.userAgent;
			let os = 'Unknown';
			if (ua.indexOf('Android') >= 0) {
				os = 'Android';
			} else if (ua.indexOf('Linux') >= 0) {
				os = 'Linux';
			} else if (ua.indexOf('iPhone') >= 0) {
				os = 'iOS';
			} else if (ua.indexOf('Macintosh') >= 0) {
				// Updated iPads will fall into this category.
				os = 'MacOSX';
			} else if (ua.indexOf('Windows') >= 0) {
				os = 'Windows';
			}

			const id = pad.id;
			// Chrom* style: NAME (Vendor: xxxx Product: xxxx)
			const exp1 = /vendor: ([0-9a-f]{4}) product: ([0-9a-f]{4})/i;
			// Firefox/Safari style (safari may remove leading zeores)
			const exp2 = /^([0-9a-f]+)-([0-9a-f]+)-/i;
			let vendor = '';
			let product = '';
			if (exp1.test(id)) {
				const match = exp1.exec(id);
				vendor = match[1].padStart(4, '0');
				product = match[2].padStart(4, '0');
			} else if (exp2.test(id)) {
				const match = exp2.exec(id);
				vendor = match[1].padStart(4, '0');
				product = match[2].padStart(4, '0');
			}
			if (!vendor || !product) {
				return `${os}Unknown`;
			}
			return os + vendor + product;
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayGamepads);

const GodotDisplayScreen = {
	$GodotDisplayScreen__deps: ['$GodotConfig', '$GodotOS', '$GL', 'emscripten_webgl_get_current_context'],
	$GodotDisplayScreen: {
		desired_size: [0, 0],
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
			const scale = window.devicePixelRatio || 1;
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
	$GodotDisplay__deps: ['$GodotConfig', '$GodotRuntime', '$GodotDisplayCursor', '$GodotDisplayListeners', '$GodotDisplayDragDrop', '$GodotDisplayGamepads', '$GodotDisplayScreen'],
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
		return window.devicePixelRatio || 1;
	},

	godot_js_display_fullscreen_request__sig: 'i',
	godot_js_display_fullscreen_request: function () {
		return GodotDisplayScreen.requestFullscreen();
	},

	godot_js_display_fullscreen_exit__sig: 'i',
	godot_js_display_fullscreen_exit: function () {
		return GodotDisplayScreen.exitFullscreen();
	},

	godot_js_display_desired_size_set__sig: 'v',
	godot_js_display_desired_size_set: function (width, height) {
		GodotDisplayScreen.desired_size = [width, height];
		GodotDisplayScreen.updateSize();
	},

	godot_js_display_size_update__sig: 'i',
	godot_js_display_size_update: function () {
		return GodotDisplayScreen.updateSize();
	},

	godot_js_display_screen_size_get__sig: 'vii',
	godot_js_display_screen_size_get: function (width, height) {
		const scale = window.devicePixelRatio || 1;
		GodotRuntime.setHeapValue(width, window.screen.width * scale, 'i32');
		GodotRuntime.setHeapValue(height, window.screen.height * scale, 'i32');
	},

	godot_js_display_window_size_get: function (p_width, p_height) {
		GodotRuntime.setHeapValue(p_width, GodotConfig.canvas.width, 'i32');
		GodotRuntime.setHeapValue(p_height, GodotConfig.canvas.height, 'i32');
	},

	godot_js_display_compute_position: function (x, y, r_x, r_y) {
		const canvas = GodotConfig.canvas;
		const rect = canvas.getBoundingClientRect();
		const rw = canvas.width / rect.width;
		const rh = canvas.height / rect.height;
		GodotRuntime.setHeapValue(r_x, (x - rect.x) * rw, 'i32');
		GodotRuntime.setHeapValue(r_y, (y - rect.y) * rh, 'i32');
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
		const png = new Blob([GodotRuntime.heapCopy(HEAPU8, p_ptr, p_len)], { type: 'image/png' });
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
			const png = new Blob([GodotRuntime.heapCopy(HEAPU8, p_ptr, p_len)], { type: 'image/png' });
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

	/*
	 * Listeners
	 */
	godot_js_display_notification_cb__sig: 'viiiii',
	godot_js_display_notification_cb: function (callback, p_enter, p_exit, p_in, p_out) {
		const canvas = GodotConfig.canvas;
		const func = GodotRuntime.get_func(callback);
		const notif = [p_enter, p_exit, p_in, p_out];
		['mouseover', 'mouseleave', 'focus', 'blur'].forEach(function (evt_name, idx) {
			GodotDisplayListeners.add(canvas, evt_name, function () {
				func.bind(null, notif[idx]);
			}, true);
		});
	},

	godot_js_display_paste_cb__sig: 'vi',
	godot_js_display_paste_cb: function (callback) {
		const func = GodotRuntime.get_func(callback);
		GodotDisplayListeners.add(window, 'paste', function (evt) {
			const text = evt.clipboardData.getData('text');
			const ptr = GodotRuntime.allocString(text);
			func(ptr);
			GodotRuntime.free(ptr);
		}, false);
	},

	godot_js_display_drop_files_cb__sig: 'vi',
	godot_js_display_drop_files_cb: function (callback) {
		const func = GodotRuntime.get_func(callback);
		const dropFiles = function (files) {
			const args = files || [];
			if (!args.length) {
				return;
			}
			const argc = args.length;
			const argv = GodotRuntime.allocStringArray(args);
			func(argv, argc);
			GodotRuntime.freeStringArray(argv, argc);
		};
		const canvas = GodotConfig.canvas;
		GodotDisplayListeners.add(canvas, 'dragover', function (ev) {
			// Prevent default behavior (which would try to open the file(s))
			ev.preventDefault();
		}, false);
		GodotDisplayListeners.add(canvas, 'drop', GodotDisplayDragDrop.handler(dropFiles));
	},

	godot_js_display_setup_canvas__sig: 'viii',
	godot_js_display_setup_canvas: function (p_width, p_height, p_fullscreen) {
		const canvas = GodotConfig.canvas;
		GodotDisplayListeners.add(canvas, 'contextmenu', function (ev) {
			ev.preventDefault();
		}, false);
		GodotDisplayListeners.add(canvas, 'webglcontextlost', function (ev) {
			alert('WebGL context lost, please reload the page'); // eslint-disable-line no-alert
			ev.preventDefault();
		}, false);
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
		if (p_fullscreen) {
			GodotDisplayScreen.requestFullscreen();
		}
	},

	/*
	 * Gamepads
	 */
	godot_js_display_gamepad_cb__sig: 'vi',
	godot_js_display_gamepad_cb: function (change_cb) {
		const onchange = GodotRuntime.get_func(change_cb);
		GodotDisplayGamepads.init(onchange);
	},

	godot_js_display_gamepad_sample_count__sig: 'i',
	godot_js_display_gamepad_sample_count: function () {
		return GodotDisplayGamepads.get_samples().length;
	},

	godot_js_display_gamepad_sample__sig: 'i',
	godot_js_display_gamepad_sample: function () {
		GodotDisplayGamepads.sample();
		return 0;
	},

	godot_js_display_gamepad_sample_get__sig: 'iiiiiii',
	godot_js_display_gamepad_sample_get: function (p_index, r_btns, r_btns_num, r_axes, r_axes_num, r_standard) {
		const sample = GodotDisplayGamepads.get_sample(p_index);
		if (!sample || !sample.connected) {
			return 1;
		}
		const btns = sample.buttons;
		const btns_len = btns.length < 16 ? btns.length : 16;
		for (let i = 0; i < btns_len; i++) {
			GodotRuntime.setHeapValue(r_btns + (i << 2), btns[i], 'float');
		}
		GodotRuntime.setHeapValue(r_btns_num, btns_len, 'i32');
		const axes = sample.axes;
		const axes_len = axes.length < 10 ? axes.length : 10;
		for (let i = 0; i < axes_len; i++) {
			GodotRuntime.setHeapValue(r_axes + (i << 2), axes[i], 'float');
		}
		GodotRuntime.setHeapValue(r_axes_num, axes_len, 'i32');
		const is_standard = sample.standard ? 1 : 0;
		GodotRuntime.setHeapValue(r_standard, is_standard, 'i32');
		return 0;
	},
};

autoAddDeps(GodotDisplay, '$GodotDisplay');
mergeInto(LibraryManager.library, GodotDisplay);
