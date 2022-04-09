/*************************************************************************/
/*  GodotInputHandler.java                                               */
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

package org.godotengine.godot.input;

import static org.godotengine.godot.utils.GLUtils.DEBUG;

import org.godotengine.godot.GodotLib;
import org.godotengine.godot.GodotRenderView;
import org.godotengine.godot.input.InputManagerCompat.InputDeviceListener;

import android.os.Build;
import android.util.Log;
import android.util.SparseArray;
import android.util.SparseIntArray;
import android.view.InputDevice;
import android.view.InputDevice.MotionRange;
import android.view.KeyEvent;
import android.view.MotionEvent;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

/**
 * Handles input related events for the {@link GodotRenderView} view.
 */
public class GodotInputHandler implements InputDeviceListener {
	private final GodotRenderView mRenderView;
	private final InputManagerCompat mInputManager;

	private final String tag = this.getClass().getSimpleName();

	private final SparseIntArray mJoystickIds = new SparseIntArray(4);
	private final SparseArray<Joystick> mJoysticksDevices = new SparseArray<>(4);

	public GodotInputHandler(GodotRenderView godotView) {
		mRenderView = godotView;
		mInputManager = InputManagerCompat.Factory.getInputManager(mRenderView.getView().getContext());
		mInputManager.registerInputDeviceListener(this, null);
	}

	private boolean isKeyEvent_GameDevice(int source) {
		// Note that keyboards are often (SOURCE_KEYBOARD | SOURCE_DPAD)
		if (source == (InputDevice.SOURCE_KEYBOARD | InputDevice.SOURCE_DPAD))
			return false;

		return (source & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK || (source & InputDevice.SOURCE_DPAD) == InputDevice.SOURCE_DPAD || (source & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD;
	}

	public boolean onKeyUp(final int keyCode, KeyEvent event) {
		if (keyCode == KeyEvent.KEYCODE_BACK) {
			return true;
		}

		if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
			return false;
		}

		int source = event.getSource();
		if (isKeyEvent_GameDevice(source)) {
			// Check if the device exists
			final int deviceId = event.getDeviceId();
			if (mJoystickIds.indexOfKey(deviceId) >= 0) {
				final int button = getGodotButton(keyCode);
				final int godotJoyId = mJoystickIds.get(deviceId);
				GodotLib.joybutton(godotJoyId, button, false);
			}
		} else {
			final int scanCode = event.getScanCode();
			final int chr = event.getUnicodeChar(0);
			GodotLib.key(keyCode, scanCode, chr, false);
		}

		return true;
	}

	public boolean onKeyDown(final int keyCode, KeyEvent event) {
		if (keyCode == KeyEvent.KEYCODE_BACK) {
			mRenderView.onBackPressed();
			// press 'back' button should not terminate program
			//normal handle 'back' event in game logic
			return true;
		}

		if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
			return false;
		}

		int source = event.getSource();
		//Log.e(TAG, String.format("Key down! source %d, device %d, joystick %d, %d, %d", event.getDeviceId(), source, (source & InputDevice.SOURCE_JOYSTICK), (source & InputDevice.SOURCE_DPAD), (source & InputDevice.SOURCE_GAMEPAD)));

		final int deviceId = event.getDeviceId();
		// Check if source is a game device and that the device is a registered gamepad
		if (isKeyEvent_GameDevice(source)) {
			if (event.getRepeatCount() > 0) // ignore key echo
				return true;

			if (mJoystickIds.indexOfKey(deviceId) >= 0) {
				final int button = getGodotButton(keyCode);
				final int godotJoyId = mJoystickIds.get(deviceId);
				GodotLib.joybutton(godotJoyId, button, true);
			}
		} else {
			final int scanCode = event.getScanCode();
			final int chr = event.getUnicodeChar(0);
			GodotLib.key(keyCode, scanCode, chr, true);
		}

		return true;
	}

	public boolean onTouchEvent(final MotionEvent event) {
		// Mouse drag (mouse pressed and move) doesn't fire onGenericMotionEvent so this is needed
		if (event.isFromSource(InputDevice.SOURCE_MOUSE)) {
			if (event.getAction() != MotionEvent.ACTION_MOVE) {
				// we return true because every time a mouse event is fired, the event is already handled
				// in onGenericMotionEvent, so by touch event we can say that the event is also handled
				return true;
			}
			return handleMouseEvent(event);
		}

		final int evcount = event.getPointerCount();
		if (evcount == 0)
			return true;

		if (mRenderView != null) {
			final float[] arr = new float[event.getPointerCount() * 3]; // pointerId1, x1, y1, pointerId2, etc...

			for (int i = 0; i < event.getPointerCount(); i++) {
				arr[i * 3 + 0] = event.getPointerId(i);
				arr[i * 3 + 1] = event.getX(i);
				arr[i * 3 + 2] = event.getY(i);
			}
			final int action = event.getActionMasked();
			final int pointer_idx = event.getPointerId(event.getActionIndex());

			switch (action) {
				case MotionEvent.ACTION_DOWN:
				case MotionEvent.ACTION_CANCEL:
				case MotionEvent.ACTION_UP:
				case MotionEvent.ACTION_MOVE:
				case MotionEvent.ACTION_POINTER_UP:
				case MotionEvent.ACTION_POINTER_DOWN: {
					GodotLib.touch(event.getSource(), action, pointer_idx, evcount, arr);
				} break;
			}
		}
		return true;
	}

	public boolean onGenericMotionEvent(MotionEvent event) {
		if (event.isFromSource(InputDevice.SOURCE_JOYSTICK) && event.getAction() == MotionEvent.ACTION_MOVE) {
			// Check if the device exists
			final int deviceId = event.getDeviceId();
			if (mJoystickIds.indexOfKey(deviceId) >= 0) {
				final int godotJoyId = mJoystickIds.get(deviceId);
				Joystick joystick = mJoysticksDevices.get(deviceId);

				for (int i = 0; i < joystick.axes.size(); i++) {
					final int axis = joystick.axes.get(i);
					final float value = event.getAxisValue(axis);
					/**
					 * As all axes are polled for each event, only fire an axis event if the value has actually changed.
					 * Prevents flooding Godot with repeated events.
					 */
					if (joystick.axesValues.indexOfKey(axis) < 0 || (float)joystick.axesValues.get(axis) != value) {
						// save value to prevent repeats
						joystick.axesValues.put(axis, value);
						final int godotAxisIdx = i;
						GodotLib.joyaxis(godotJoyId, godotAxisIdx, value);
					}
				}

				if (joystick.hasAxisHat) {
					final int hatX = Math.round(event.getAxisValue(MotionEvent.AXIS_HAT_X));
					final int hatY = Math.round(event.getAxisValue(MotionEvent.AXIS_HAT_Y));
					if (joystick.hatX != hatX || joystick.hatY != hatY) {
						joystick.hatX = hatX;
						joystick.hatY = hatY;
						GodotLib.joyhat(godotJoyId, hatX, hatY);
					}
				}
				return true;
			}
		} else if (event.isFromSource(InputDevice.SOURCE_STYLUS)) {
			final float x = event.getX();
			final float y = event.getY();
			final int type = event.getAction();
			GodotLib.hover(type, x, y);
			return true;

		} else if (event.isFromSource(InputDevice.SOURCE_MOUSE) || event.isFromSource(InputDevice.SOURCE_MOUSE_RELATIVE)) {
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
				return handleMouseEvent(event);
			}
		}

		return false;
	}

	public void initInputDevices() {
		/* initially add input devices*/
		int[] deviceIds = mInputManager.getInputDeviceIds();
		for (int deviceId : deviceIds) {
			InputDevice device = mInputManager.getInputDevice(deviceId);
			if (DEBUG) {
				Log.v("GodotInputHandler", String.format("init() deviceId:%d, Name:%s\n", deviceId, device.getName()));
			}
			onInputDeviceAdded(deviceId);
		}
	}

	private int assignJoystickIdNumber(int deviceId) {
		int godotJoyId = 0;
		while (mJoystickIds.indexOfValue(godotJoyId) >= 0) {
			godotJoyId++;
		}
		mJoystickIds.put(deviceId, godotJoyId);
		return godotJoyId;
	}

	@Override
	public void onInputDeviceAdded(int deviceId) {
		// Check if the device has not been already added

		if (mJoystickIds.indexOfKey(deviceId) >= 0) {
			return;
		}

		InputDevice device = mInputManager.getInputDevice(deviceId);
		//device can be null if deviceId is not found
		if (device == null) {
			return;
		}

		int sources = device.getSources();

		// Device may not be a joystick or gamepad
		if ((sources & InputDevice.SOURCE_GAMEPAD) != InputDevice.SOURCE_GAMEPAD &&
				(sources & InputDevice.SOURCE_JOYSTICK) != InputDevice.SOURCE_JOYSTICK) {
			return;
		}

		// Assign first available number. Re-use numbers where possible.
		final int id = assignJoystickIdNumber(deviceId);

		final Joystick joystick = new Joystick();
		joystick.device_id = deviceId;
		joystick.name = device.getName();

		//Helps with creating new joypad mappings.
		Log.i(tag, "=== New Input Device: " + joystick.name);

		Set<Integer> already = new HashSet<>();
		for (InputDevice.MotionRange range : device.getMotionRanges()) {
			boolean isJoystick = range.isFromSource(InputDevice.SOURCE_JOYSTICK);
			boolean isGamepad = range.isFromSource(InputDevice.SOURCE_GAMEPAD);
			//Log.i(tag, "axis: "+range.getAxis()+ ", isJoystick: "+isJoystick+", isGamepad: "+isGamepad);
			if (!isJoystick && !isGamepad) {
				continue;
			}
			final int axis = range.getAxis();
			if (axis == MotionEvent.AXIS_HAT_X || axis == MotionEvent.AXIS_HAT_Y) {
				joystick.hasAxisHat = true;
			} else {
				if (!already.contains(axis)) {
					already.add(axis);
					joystick.axes.add(axis);
				} else {
					Log.w(tag, " - DUPLICATE AXIS VALUE IN LIST: " + axis);
				}
			}
		}
		Collections.sort(joystick.axes);
		for (int idx = 0; idx < joystick.axes.size(); idx++) {
			//Helps with creating new joypad mappings.
			Log.i(tag, " - Mapping Android axis " + joystick.axes.get(idx) + " to Godot axis " + idx);
		}
		mJoysticksDevices.put(deviceId, joystick);

		GodotLib.joyconnectionchanged(id, true, joystick.name);
	}

	@Override
	public void onInputDeviceRemoved(int deviceId) {
		// Check if the device has not been already removed
		if (mJoystickIds.indexOfKey(deviceId) < 0) {
			return;
		}
		final int godotJoyId = mJoystickIds.get(deviceId);
		mJoystickIds.delete(deviceId);
		mJoysticksDevices.delete(deviceId);
		GodotLib.joyconnectionchanged(godotJoyId, false, "");
	}

	@Override
	public void onInputDeviceChanged(int deviceId) {
		onInputDeviceRemoved(deviceId);
		onInputDeviceAdded(deviceId);
	}

	private static class RangeComparator implements Comparator<MotionRange> {
		@Override
		public int compare(MotionRange arg0, MotionRange arg1) {
			return arg0.getAxis() - arg1.getAxis();
		}
	}

	public static int getGodotButton(int keyCode) {
		int button;
		switch (keyCode) {
			case KeyEvent.KEYCODE_BUTTON_A: // Android A is SNES B
				button = 0;
				break;
			case KeyEvent.KEYCODE_BUTTON_B:
				button = 1;
				break;
			case KeyEvent.KEYCODE_BUTTON_X: // Android X is SNES Y
				button = 2;
				break;
			case KeyEvent.KEYCODE_BUTTON_Y:
				button = 3;
				break;
			case KeyEvent.KEYCODE_BUTTON_L1:
				button = 9;
				break;
			case KeyEvent.KEYCODE_BUTTON_L2:
				button = 15;
				break;
			case KeyEvent.KEYCODE_BUTTON_R1:
				button = 10;
				break;
			case KeyEvent.KEYCODE_BUTTON_R2:
				button = 16;
				break;
			case KeyEvent.KEYCODE_BUTTON_SELECT:
				button = 4;
				break;
			case KeyEvent.KEYCODE_BUTTON_START:
				button = 6;
				break;
			case KeyEvent.KEYCODE_BUTTON_THUMBL:
				button = 7;
				break;
			case KeyEvent.KEYCODE_BUTTON_THUMBR:
				button = 8;
				break;
			case KeyEvent.KEYCODE_DPAD_UP:
				button = 11;
				break;
			case KeyEvent.KEYCODE_DPAD_DOWN:
				button = 12;
				break;
			case KeyEvent.KEYCODE_DPAD_LEFT:
				button = 13;
				break;
			case KeyEvent.KEYCODE_DPAD_RIGHT:
				button = 14;
				break;
			case KeyEvent.KEYCODE_BUTTON_C:
				button = 17;
				break;
			case KeyEvent.KEYCODE_BUTTON_Z:
				button = 18;
				break;

			default:
				button = keyCode - KeyEvent.KEYCODE_BUTTON_1 + 20;
				break;
		}
		return button;
	}

	private boolean handleMouseEvent(final MotionEvent event) {
		switch (event.getActionMasked()) {
			case MotionEvent.ACTION_HOVER_ENTER:
			case MotionEvent.ACTION_HOVER_MOVE:
			case MotionEvent.ACTION_HOVER_EXIT: {
				final float x = event.getX();
				final float y = event.getY();
				final int type = event.getAction();
				GodotLib.hover(type, x, y);
				return true;
			}
			case MotionEvent.ACTION_BUTTON_PRESS:
			case MotionEvent.ACTION_BUTTON_RELEASE:
			case MotionEvent.ACTION_MOVE: {
				final float x = event.getX();
				final float y = event.getY();
				final int buttonsMask = event.getButtonState();
				final int action = event.getAction();
				GodotLib.touch(event.getSource(), action, 0, 1, new float[] { 0, x, y }, buttonsMask);
				return true;
			}
			case MotionEvent.ACTION_SCROLL: {
				final float x = event.getX();
				final float y = event.getY();
				final int buttonsMask = event.getButtonState();
				final int action = event.getAction();
				final float verticalFactor = event.getAxisValue(MotionEvent.AXIS_VSCROLL);
				final float horizontalFactor = event.getAxisValue(MotionEvent.AXIS_HSCROLL);
				GodotLib.touch(event.getSource(), action, 0, 1, new float[] { 0, x, y }, buttonsMask, verticalFactor, horizontalFactor);
			}
			case MotionEvent.ACTION_DOWN:
			case MotionEvent.ACTION_UP: {
				// we can safely ignore these cases because they are always come beside ACTION_BUTTON_PRESS and ACTION_BUTTON_RELEASE
				return true;
			}
		}
		return false;
	}
}
