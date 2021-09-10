/*************************************************************************/
/*  library_godot_input.js                                               */
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

const GodotInput = {
	$GodotInput__deps: ['$GodotRuntime', '$GodotConfig', '$GodotDisplayListeners'],
	$GodotInput: {
		getModifiers: function (evt) {
			return (evt.shiftKey + 0) + ((evt.altKey + 0) << 1) + ((evt.ctrlKey + 0) << 2) + ((evt.metaKey + 0) << 3);
		},
		computePosition: function (evt, rect) {
			const canvas = GodotConfig.canvas;
			const rw = canvas.width / rect.width;
			const rh = canvas.height / rect.height;
			const x = (evt.clientX - rect.x) * rw;
			const y = (evt.clientY - rect.y) * rh;
			return [x, y];
		},
	},

	godot_js_display_mouse_move_cb__sig: 'vi',
	godot_js_display_mouse_move_cb: function (callback) {
		const func = GodotRuntime.get_func(callback);
		const canvas = GodotConfig.canvas;
		function move_cb(evt) {
			const rect = canvas.getBoundingClientRect();
			const pos = GodotInput.computePosition(evt, rect);
			// Scale movement
			const rw = canvas.width / rect.width;
			const rh = canvas.height / rect.height;
			const rel_pos_x = evt.movementX * rw;
			const rel_pos_y = evt.movementY * rh;
			const modifiers = GodotInput.getModifiers(evt);
			func(pos[0], pos[1], rel_pos_x, rel_pos_y, modifiers);
		}
		GodotDisplayListeners.add(window, 'mousemove', move_cb, false);
	},

	godot_js_display_mouse_wheel_cb__sig: 'vi',
	godot_js_display_mouse_wheel_cb: function (callback) {
		const func = GodotRuntime.get_func(callback);
		function wheel_cb(evt) {
			if (func(evt['deltaX'] || 0, evt['deltaY'] || 0)) {
				evt.preventDefault();
			}
		}
		GodotDisplayListeners.add(GodotConfig.canvas, 'wheel', wheel_cb, false);
	},

	godot_js_display_mouse_button_cb__sig: 'vi',
	godot_js_display_mouse_button_cb: function (callback) {
		const func = GodotRuntime.get_func(callback);
		const canvas = GodotConfig.canvas;
		function button_cb(p_pressed, evt) {
			const rect = canvas.getBoundingClientRect();
			const pos = GodotInput.computePosition(evt, rect);
			const modifiers = GodotInput.getModifiers(evt);
			if (func(p_pressed, evt.button, pos[0], pos[1], modifiers)) {
				evt.preventDefault();
			}
		}
		GodotDisplayListeners.add(canvas, 'mousedown', button_cb.bind(null, 1), false);
		GodotDisplayListeners.add(window, 'mouseup', button_cb.bind(null, 0), false);
	},

	godot_js_display_touch_cb__sig: 'viii',
	godot_js_display_touch_cb: function (callback, ids, coords) {
		const func = GodotRuntime.get_func(callback);
		const canvas = GodotConfig.canvas;
		function touch_cb(type, evt) {
			const rect = canvas.getBoundingClientRect();
			const touches = evt.changedTouches;
			for (let i = 0; i < touches.length; i++) {
				const touch = touches[i];
				const pos = GodotInput.computePosition(touch, rect);
				GodotRuntime.setHeapValue(coords + (i * 2), pos[0], 'double');
				GodotRuntime.setHeapValue(coords + (i * 2 + 8), pos[1], 'double');
				GodotRuntime.setHeapValue(ids + i, touch.identifier, 'i32');
			}
			func(type, touches.length);
			if (evt.cancelable) {
				evt.preventDefault();
			}
		}
		GodotDisplayListeners.add(canvas, 'touchstart', touch_cb.bind(null, 0), false);
		GodotDisplayListeners.add(canvas, 'touchend', touch_cb.bind(null, 1), false);
		GodotDisplayListeners.add(canvas, 'touchcancel', touch_cb.bind(null, 1), false);
		GodotDisplayListeners.add(canvas, 'touchmove', touch_cb.bind(null, 2), false);
	},

	godot_js_display_key_cb__sig: 'viii',
	godot_js_display_key_cb: function (callback, code, key) {
		const func = GodotRuntime.get_func(callback);
		function key_cb(pressed, evt) {
			const modifiers = GodotInput.getModifiers(evt);
			GodotRuntime.stringToHeap(evt.code, code, 32);
			GodotRuntime.stringToHeap(evt.key, key, 32);
			func(pressed, evt.repeat, modifiers);
			evt.preventDefault();
		}
		GodotDisplayListeners.add(GodotConfig.canvas, 'keydown', key_cb.bind(null, 1), false);
		GodotDisplayListeners.add(GodotConfig.canvas, 'keyup', key_cb.bind(null, 0), false);
	},
};

autoAddDeps(GodotInput, '$GodotInput');
mergeInto(LibraryManager.library, GodotInput);
