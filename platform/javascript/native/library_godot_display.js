/*************************************************************************/
/*  library_godot_display.js                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
	$GodotDisplayListeners__postset: 'GodotOS.atexit(function(resolve, reject) { GodotDisplayListeners.clear(); resolve(); });',
	$GodotDisplayListeners: {
		handlers: [],

		has: function(target, event, method, capture) {
			return GodotDisplayListeners.handlers.findIndex(function(e) {
				return e.target === target && e.event === event && e.method === method && e.capture == capture;
			}) !== -1;
		},

		add: function(target, event, method, capture) {
			if (GodotDisplayListeners.has(target, event, method, capture)) {
				return;
			}
			function Handler(target, event, method, capture) {
				this.target = target;
				this.event = event;
				this.method = method;
				this.capture = capture;
			};
			GodotDisplayListeners.handlers.push(new Handler(target, event, method, capture));
			target.addEventListener(event, method, capture);
		},

		clear: function() {
			GodotDisplayListeners.handlers.forEach(function(h) {
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

		add_entry: function(entry) {
			if (entry.isDirectory) {
				GodotDisplayDragDrop.add_dir(entry);
			} else if (entry.isFile) {
				GodotDisplayDragDrop.add_file(entry);
			} else {
				console.error("Unrecognized entry...", entry);
			}
		},

		add_dir: function(entry) {
			GodotDisplayDragDrop.promises.push(new Promise(function(resolve, reject) {
				const reader = entry.createReader();
				reader.readEntries(function(entries) {
					for (let i = 0; i < entries.length; i++) {
						GodotDisplayDragDrop.add_entry(entries[i]);
					}
					resolve();
				});
			}));
		},

		add_file: function(entry) {
			GodotDisplayDragDrop.promises.push(new Promise(function(resolve, reject) {
				entry.file(function(file) {
					const reader = new FileReader();
					reader.onload = function() {
						const f = {
							"path": file.relativePath || file.webkitRelativePath,
							"name": file.name,
							"type": file.type,
							"size": file.size,
							"data": reader.result
						};
						if (!f['path']) {
							f['path'] = f['name'];
						}
						GodotDisplayDragDrop.pending_files.push(f);
						resolve()
					};
					reader.onerror = function() {
						console.log("Error reading file");
						reject();
					}
					reader.readAsArrayBuffer(file);
				}, function(err) {
					console.log("Error!");
					reject();
				});
			}));
		},

		process: function(resolve, reject) {
			if (GodotDisplayDragDrop.promises.length == 0) {
				resolve();
				return;
			}
			GodotDisplayDragDrop.promises.pop().then(function() {
				setTimeout(function() {
					GodotDisplayDragDrop.process(resolve, reject);
				}, 0);
			});
		},

		_process_event: function(ev, callback) {
			ev.preventDefault();
			if (ev.dataTransfer.items) {
				// Use DataTransferItemList interface to access the file(s)
				for (let i = 0; i < ev.dataTransfer.items.length; i++) {
					const item = ev.dataTransfer.items[i];
					let entry = null;
					if ("getAsEntry" in item) {
						entry = item.getAsEntry();
					} else if ("webkitGetAsEntry" in item) {
						entry = item.webkitGetAsEntry();
					}
					if (entry) {
						GodotDisplayDragDrop.add_entry(entry);
					}
				}
			} else {
				console.error("File upload not supported");
			}
			new Promise(GodotDisplayDragDrop.process).then(function() {
				const DROP = "/tmp/drop-" + parseInt(Math.random() * Math.pow(2, 31)) + "/";
				const drops = [];
				const files = [];
				FS.mkdir(DROP);
				GodotDisplayDragDrop.pending_files.forEach((elem) => {
					const path = elem['path'];
					GodotFS.copy_to_fs(DROP + path, elem['data']);
					let idx = path.indexOf("/");
					if (idx == -1) {
						// Root file
						drops.push(DROP + path);
					} else {
						// Subdir
						const sub = path.substr(0, idx);
						idx = sub.indexOf("/");
						if (idx < 0 && drops.indexOf(DROP + sub) == -1) {
							drops.push(DROP + sub);
						}
					}
					files.push(DROP + path);
				});
				GodotDisplayDragDrop.promises = [];
				GodotDisplayDragDrop.pending_files = [];
				callback(drops);
				const dirs = [DROP.substr(0, DROP.length -1)];
				// Remove temporary files
				files.forEach(function (file) {
					FS.unlink(file);
					let dir = file.replace(DROP, "");
					let idx = dir.lastIndexOf("/");
					while (idx > 0) {
						dir = dir.substr(0, idx);
						if (dirs.indexOf(DROP + dir) == -1) {
							dirs.push(DROP + dir);
						}
						idx = dir.lastIndexOf("/");
					}
				});
				// Remove dirs.
				dirs.sort(function(a, b) {
					const al = (a.match(/\//g) || []).length;
					const bl = (b.match(/\//g) || []).length;
					if (al > bl)
						return -1;
					else if (al < bl)
						return 1;
					return 0;
				}).forEach(function(dir) {
					FS.rmdir(dir);
				});
			});
		},

		handler: function(callback) {
			return function(ev) {
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
	$GodotDisplayCursor__postset: 'GodotOS.atexit(function(resolve, reject) { GodotDisplayCursor.clear(); resolve(); });',
	$GodotDisplayCursor__deps: ['$GodotConfig', '$GodotOS'],
	$GodotDisplayCursor: {
		shape: 'auto',
		visible: true,
		cursors: {},
		set_style: function(style) {
			GodotConfig.canvas.style.cursor = style;
		},
		set_shape: function(shape) {
			GodotDisplayCursor.shape = shape;
			let css = shape;
			if (shape in GodotDisplayCursor.cursors) {
				const c = GodotDisplayCursor.cursors[shape];
				css = 'url("' + c.url + '") ' + c.x + ' ' + c.y + ', auto';
			}
			if (GodotDisplayCursor.visible) {
				GodotDisplayCursor.set_style(css);
			}
		},
		clear: function() {
			GodotDisplayCursor.set_style('');
			GodotDisplayCursor.shape = 'auto';
			GodotDisplayCursor.visible = true;
			Object.keys(GodotDisplayCursor.cursors).forEach(function(key) {
				URL.revokeObjectURL(GodotDisplayCursor.cursors[key]);
				delete GodotDisplayCursor.cursors[key];
			});
		},
	},
};
mergeInto(LibraryManager.library, GodotDisplayCursor);

/**
 * Display server interface.
 *
 * Exposes all the functions needed by DisplayServer implementation.
 */
const GodotDisplay = {
	$GodotDisplay__deps: ['$GodotConfig', '$GodotOS', '$GodotDisplayCursor', '$GodotDisplayListeners', '$GodotDisplayDragDrop'],
	$GodotDisplay: {
		window_icon: '',
	},

	godot_js_display_is_swap_ok_cancel: function() {
		const win = (['Windows', 'Win64', 'Win32', 'WinCE']);
		const plat = navigator.platform || "";
		if (win.indexOf(plat) !== -1) {
			return 1;
		}
		return 0;
	},

	godot_js_display_alert: function(p_text) {
		window.alert(UTF8ToString(p_text));
	},

	godot_js_display_pixel_ratio_get: function() {
		return window.devicePixelRatio || 1;
	},

	/*
	 * Canvas
	 */
	godot_js_display_canvas_focus: function() {
		GodotConfig.canvas.focus();
	},

	godot_js_display_canvas_is_focused: function() {
		return document.activeElement == GodotConfig.canvas;
	},

	godot_js_display_canvas_bounding_rect_position_get: function(r_x, r_y) {
		const brect = GodotConfig.canvas.getBoundingClientRect();
		setValue(r_x, brect.x, 'i32');
		setValue(r_y, brect.y, 'i32');
	},

	/*
	 * Touchscreen
	 */
	godot_js_display_touchscreen_is_available: function() {
		return 'ontouchstart' in window;
	},

	/*
	 * Clipboard
	 */
	godot_js_display_clipboard_set: function(p_text) {
		const text = UTF8ToString(p_text);
		if (!navigator.clipboard || !navigator.clipboard.writeText) {
			return 1;
		}
		navigator.clipboard.writeText(text).catch(function(e) {
			// Setting OS clipboard is only possible from an input callback.
			console.error("Setting OS clipboard is only possible from an input callback for the HTML5 plafrom. Exception:", e);
		});
		return 0;
	},

	godot_js_display_clipboard_get_deps: ['$GodotOS'],
	godot_js_display_clipboard_get: function(callback) {
		const func = GodotOS.get_func(callback);
		try {
			navigator.clipboard.readText().then(function (result) {
				const ptr = allocate(intArrayFromString(result), ALLOC_NORMAL);
				func(ptr);
				_free(ptr);
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
	godot_js_display_window_request_fullscreen: function() {
		const canvas = GodotConfig.canvas;
		(canvas.requestFullscreen || canvas.msRequestFullscreen ||
			canvas.mozRequestFullScreen || canvas.mozRequestFullscreen ||
			canvas.webkitRequestFullscreen
		).call(canvas);
	},

	godot_js_display_window_title_set: function(p_data) {
		document.title = UTF8ToString(p_data);
	},

	godot_js_display_window_icon_set: function(p_ptr, p_len) {
		let link = document.getElementById('-gd-engine-icon');
		if (link === null) {
			link = document.createElement('link');
			link.rel = 'icon';
			link.id = '-gd-engine-icon';
			document.head.appendChild(link);
		}
		const old_icon = GodotDisplay.window_icon;
		const png = new Blob([GodotOS.heapCopy(HEAPU8, p_ptr, p_len)], { type: "image/png" });
		GodotDisplay.window_icon = URL.createObjectURL(png);
		link.href = GodotDisplay.window_icon;
		if (old_icon) {
			URL.revokeObjectURL(old_icon);
		}
	},

	/*
	 * Cursor
	 */
	godot_js_display_cursor_set_visible: function(p_visible) {
		const visible = p_visible != 0;
		if (visible == GodotDisplayCursor.visible) {
			return;
		}
		GodotDisplayCursor.visible = visible;
		if (visible) {
			GodotDisplayCursor.set_shape(GodotDisplayCursor.shape);
		} else {
			GodotDisplayCursor.set_style('none');
		}
	},

	godot_js_display_cursor_is_hidden: function() {
		return !GodotDisplayCursor.visible;
	},

	godot_js_display_cursor_set_shape: function(p_string) {
		GodotDisplayCursor.set_shape(UTF8ToString(p_string));
	},

	godot_js_display_cursor_set_custom_shape: function(p_shape, p_ptr, p_len, p_hotspot_x, p_hotspot_y) {
		const shape = UTF8ToString(p_shape);
		const old_shape = GodotDisplayCursor.cursors[shape];
		if (p_len > 0) {
			const png = new Blob([GodotOS.heapCopy(HEAPU8, p_ptr, p_len)], { type: 'image/png' });
			const url = URL.createObjectURL(png);
			GodotDisplayCursor.cursors[shape] = {
				url: url,
				x: p_hotspot_x,
				y: p_hotspot_y,
			};
		} else {
			delete GodotDisplayCursor.cursors[shape];
		}
		if (shape == GodotDisplayCursor.shape) {
			GodotDisplayCursor.set_shape(GodotDisplayCursor.shape);
		}
		if (old_shape) {
			URL.revokeObjectURL(old_shape.url);
		}
	},

	/*
	 * Listeners
	 */
	godot_js_display_notification_cb: function(callback, p_enter, p_exit, p_in, p_out) {
		const canvas = GodotConfig.canvas;
		const func = GodotOS.get_func(callback);
		const notif = [p_enter, p_exit, p_in, p_out];
		['mouseover', 'mouseleave', 'focus', 'blur'].forEach(function(evt_name, idx) {
			GodotDisplayListeners.add(canvas, evt_name, function() {
				func.bind(null, notif[idx]);
			}, true);
		});
	},

	godot_js_display_paste_cb: function(callback) {
		const func = GodotOS.get_func(callback);
		GodotDisplayListeners.add(window, 'paste', function(evt) {
			const text = evt.clipboardData.getData('text');
			const ptr = allocate(intArrayFromString(text), ALLOC_NORMAL);
			func(ptr);
			_free(ptr);
		}, false);
	},

	godot_js_display_drop_files_cb: function(callback) {
		const func = GodotOS.get_func(callback)
		const dropFiles = function(files) {
			const args = files || [];
			if (!args.length) {
				return;
			}
			const argc = args.length;
			const argv = GodotOS.allocStringArray(args);
			func(argv, argc);
			GodotOS.freeStringArray(argv, argc);
		};
		const canvas = GodotConfig.canvas;
		GodotDisplayListeners.add(canvas, 'dragover', function(ev) {
			// Prevent default behavior (which would try to open the file(s))
			ev.preventDefault();
		}, false);
		GodotDisplayListeners.add(canvas, 'drop', GodotDisplayDragDrop.handler(dropFiles));
	},
};

autoAddDeps(GodotDisplay, '$GodotDisplay');
mergeInto(LibraryManager.library, GodotDisplay);
