/**************************************************************************/
/*  GodotInputHandler.java                                                */
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

package org.godotengine.godot.input;

import static org.godotengine.godot.utils.GLUtils.DEBUG;

import org.godotengine.godot.GodotLib;
import org.godotengine.godot.GodotRenderView;

import android.content.Context;
import android.hardware.input.InputManager;
import android.os.Build;
import android.util.Log;
import android.util.SparseArray;
import android.util.SparseIntArray;
import android.view.GestureDetector;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Handles input related events for the {@link GodotRenderView} view.
 */
public class GodotInputHandler implements InputManager.InputDeviceListener {
	private static final String TAG = GodotInputHandler.class.getSimpleName();

	private final SparseIntArray mJoystickIds = new SparseIntArray(4);
	private final SparseArray<Joystick> mJoysticksDevices = new SparseArray<>(4);

	private final GodotRenderView mRenderView;
	private final InputManager mInputManager;
	private final GestureDetector gestureDetector;
	private final ScaleGestureDetector scaleGestureDetector;
	private final GodotGestureHandler godotGestureHandler;

	/**
	 * Used to decide whether mouse capture can be enabled.
	 */
	private int lastSeenToolType = MotionEvent.TOOL_TYPE_UNKNOWN;

	public GodotInputHandler(GodotRenderView godotView) {
		final Context context = godotView.getView().getContext();
		mRenderView = godotView;
		mInputManager = (InputManager)context.getSystemService(Context.INPUT_SERVICE);
		mInputManager.registerInputDeviceListener(this, null);

		this.godotGestureHandler = new GodotGestureHandler();
		this.gestureDetector = new GestureDetector(context, godotGestureHandler);
		this.gestureDetector.setIsLongpressEnabled(false);
		this.scaleGestureDetector = new ScaleGestureDetector(context, godotGestureHandler);
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
			this.scaleGestureDetector.setStylusScaleEnabled(true);
		}
	}

	/**
	 * Enable long press events. This is false by default.
	 */
	public void enableLongPress(boolean enable) {
		this.gestureDetector.setIsLongpressEnabled(enable);
	}

	/**
	 * Enable multi-fingers pan & scale gestures. This is false by default.
	 * <p>
	 * Note: This may interfere with multi-touch handling / support.
	 */
	public void enablePanningAndScalingGestures(boolean enable) {
		this.godotGestureHandler.setPanningAndScalingEnabled(enable);
	}

	private boolean isKeyEventGameDevice(int source) {
		// Note that keyboards are often (SOURCE_KEYBOARD | SOURCE_DPAD)
		if (source == (InputDevice.SOURCE_KEYBOARD | InputDevice.SOURCE_DPAD))
			return false;

		return (source & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK || (source & InputDevice.SOURCE_DPAD) == InputDevice.SOURCE_DPAD || (source & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD;
	}

	public boolean canCapturePointer() {
		return lastSeenToolType == MotionEvent.TOOL_TYPE_MOUSE;
	}

	public void onPointerCaptureChange(boolean hasCapture) {
		godotGestureHandler.onPointerCaptureChange(hasCapture);
	}

	public boolean onKeyUp(final int keyCode, KeyEvent event) {
		if (keyCode == KeyEvent.KEYCODE_BACK) {
			return true;
		}

		if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
			return false;
		}

		int source = event.getSource();
		if (isKeyEventGameDevice(source)) {
			// Check if the device exists
			final int deviceId = event.getDeviceId();
			if (mJoystickIds.indexOfKey(deviceId) >= 0) {
				final int button = getGodotButton(keyCode);
				final int godotJoyId = mJoystickIds.get(deviceId);
				GodotLib.joybutton(godotJoyId, button, false);
			}
		} else {
			// getKeyCode(): The physical key that was pressed.
			final int physical_keycode = event.getKeyCode();
			final int unicode = event.getUnicodeChar();
			final int key_label = event.getDisplayLabel();
			GodotLib.key(physical_keycode, unicode, key_label, false, event.getRepeatCount() > 0);
		};

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

		final int deviceId = event.getDeviceId();
		// Check if source is a game device and that the device is a registered gamepad
		if (isKeyEventGameDevice(source)) {
			if (event.getRepeatCount() > 0) // ignore key echo
				return true;

			if (mJoystickIds.indexOfKey(deviceId) >= 0) {
				final int button = getGodotButton(keyCode);
				final int godotJoyId = mJoystickIds.get(deviceId);
				GodotLib.joybutton(godotJoyId, button, true);
			}
		} else {
			final int physical_keycode = event.getKeyCode();
			final int unicode = event.getUnicodeChar();
			final int key_label = event.getDisplayLabel();
			GodotLib.key(physical_keycode, unicode, key_label, true, event.getRepeatCount() > 0);
		}

		return true;
	}

	public boolean onTouchEvent(final MotionEvent event) {
		lastSeenToolType = event.getToolType(0);

		this.scaleGestureDetector.onTouchEvent(event);
		if (this.gestureDetector.onTouchEvent(event)) {
			// The gesture detector has handled the event.
			return true;
		}

		if (godotGestureHandler.onMotionEvent(event)) {
			// The gesture handler has handled the event.
			return true;
		}

		// Drag events are handled by the [GodotGestureHandler]
		if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
			return true;
		}

		if (isMouseEvent(event)) {
			return handleMouseEvent(event);
		}

		return handleTouchEvent(event);
	}

	public boolean onGenericMotionEvent(MotionEvent event) {
		lastSeenToolType = event.getToolType(0);

		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && gestureDetector.onGenericMotionEvent(event)) {
			// The gesture detector has handled the event.
			return true;
		}

		if (godotGestureHandler.onMotionEvent(event)) {
			// The gesture handler has handled the event.
			return true;
		}

		if (event.isFromSource(InputDevice.SOURCE_JOYSTICK) && event.getActionMasked() == MotionEvent.ACTION_MOVE) {
			// Check if the device exists
			final int deviceId = event.getDeviceId();
			if (mJoystickIds.indexOfKey(deviceId) >= 0) {
				final int godotJoyId = mJoystickIds.get(deviceId);
				Joystick joystick = mJoysticksDevices.get(deviceId);
				if (joystick == null) {
					return true;
				}

				for (int i = 0; i < joystick.axes.size(); i++) {
					final int axis = joystick.axes.get(i);
					final float value = event.getAxisValue(axis);
					/*
					  As all axes are polled for each event, only fire an axis event if the value has actually changed.
					  Prevents flooding Godot with repeated events.
					 */
					if (joystick.axesValues.indexOfKey(axis) < 0 || (float)joystick.axesValues.get(axis) != value) {
						// save value to prevent repeats
						joystick.axesValues.put(axis, value);
						GodotLib.joyaxis(godotJoyId, i, value);
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
		} else {
			return handleMouseEvent(event);
		}

		return false;
	}

	public void initInputDevices() {
		/* initially add input devices*/
		int[] deviceIds = mInputManager.getInputDeviceIds();
		for (int deviceId : deviceIds) {
			InputDevice device = mInputManager.getInputDevice(deviceId);
			if (DEBUG) {
				Log.v(TAG, String.format("init() deviceId:%d, Name:%s\n", deviceId, device.getName()));
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

		// Assign first available number. Reuse numbers where possible.
		final int id = assignJoystickIdNumber(deviceId);

		final Joystick joystick = new Joystick();
		joystick.device_id = deviceId;
		joystick.name = device.getName();

		//Helps with creating new joypad mappings.
		Log.i(TAG, "=== New Input Device: " + joystick.name);

		Set<Integer> already = new HashSet<>();
		for (InputDevice.MotionRange range : device.getMotionRanges()) {
			boolean isJoystick = range.isFromSource(InputDevice.SOURCE_JOYSTICK);
			boolean isGamepad = range.isFromSource(InputDevice.SOURCE_GAMEPAD);
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
					Log.w(TAG, " - DUPLICATE AXIS VALUE IN LIST: " + axis);
				}
			}
		}
		Collections.sort(joystick.axes);
		for (int idx = 0; idx < joystick.axes.size(); idx++) {
			//Helps with creating new joypad mappings.
			Log.i(TAG, " - Mapping Android axis " + joystick.axes.get(idx) + " to Godot axis " + idx);
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

	static boolean isMouseEvent(MotionEvent event) {
		return isMouseEvent(event.getSource());
	}

	private static boolean isMouseEvent(int eventSource) {
		boolean mouseSource = ((eventSource & InputDevice.SOURCE_MOUSE) == InputDevice.SOURCE_MOUSE) || ((eventSource & InputDevice.SOURCE_STYLUS) == InputDevice.SOURCE_STYLUS);
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			mouseSource = mouseSource || ((eventSource & InputDevice.SOURCE_MOUSE_RELATIVE) == InputDevice.SOURCE_MOUSE_RELATIVE);
		}
		return mouseSource;
	}

	static boolean handleMotionEvent(final MotionEvent event) {
		if (isMouseEvent(event)) {
			return handleMouseEvent(event);
		}

		return handleTouchEvent(event);
	}

	static boolean handleMotionEvent(int eventSource, int eventAction, int buttonsMask, float x, float y) {
		return handleMotionEvent(eventSource, eventAction, buttonsMask, x, y, false);
	}

	static boolean handleMotionEvent(int eventSource, int eventAction, int buttonsMask, float x, float y, boolean doubleTap) {
		return handleMotionEvent(eventSource, eventAction, buttonsMask, x, y, 0, 0, doubleTap);
	}

	static boolean handleMotionEvent(int eventSource, int eventAction, int buttonsMask, float x, float y, float deltaX, float deltaY, boolean doubleTap) {
		if (isMouseEvent(eventSource)) {
			return handleMouseEvent(eventAction, buttonsMask, x, y, deltaX, deltaY, doubleTap, false);
		}

		return handleTouchEvent(eventAction, x, y, doubleTap);
	}

	static boolean handleMouseEvent(final MotionEvent event) {
		final int eventAction = event.getActionMasked();
		final float x = event.getX();
		final float y = event.getY();
		final int buttonsMask = event.getButtonState();

		final float pressure = event.getPressure();

		// Orientation is returned as a radian value between 0 to pi clockwise or 0 to -pi counterclockwise.
		final float orientation = event.getOrientation();

		// Tilt is zero is perpendicular to the screen and pi/2 is flat on the surface.
		final float tilt = event.getAxisValue(MotionEvent.AXIS_TILT);

		float tiltMult = (float)Math.sin(tilt);

		// To be consistent with expected tilt.
		final float tiltX = (float)-Math.sin(orientation) * tiltMult;
		final float tiltY = (float)Math.cos(orientation) * tiltMult;

		final float verticalFactor = event.getAxisValue(MotionEvent.AXIS_VSCROLL);
		final float horizontalFactor = event.getAxisValue(MotionEvent.AXIS_HSCROLL);
		boolean sourceMouseRelative = false;
		if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
			sourceMouseRelative = event.isFromSource(InputDevice.SOURCE_MOUSE_RELATIVE);
		}
		return handleMouseEvent(eventAction, buttonsMask, x, y, horizontalFactor, verticalFactor, false, sourceMouseRelative, pressure, tiltX, tiltY);
	}

	static boolean handleMouseEvent(int eventAction, int buttonsMask, float x, float y) {
		return handleMouseEvent(eventAction, buttonsMask, x, y, 0, 0, false, false);
	}

	static boolean handleMouseEvent(int eventAction, int buttonsMask, float x, float y, float deltaX, float deltaY, boolean doubleClick, boolean sourceMouseRelative) {
		return handleMouseEvent(eventAction, buttonsMask, x, y, deltaX, deltaY, doubleClick, sourceMouseRelative, 1, 0, 0);
	}

	static boolean handleMouseEvent(int eventAction, int buttonsMask, float x, float y, float deltaX, float deltaY, boolean doubleClick, boolean sourceMouseRelative, float pressure, float tiltX, float tiltY) {
		// Fix the buttonsMask
		switch (eventAction) {
			case MotionEvent.ACTION_CANCEL:
			case MotionEvent.ACTION_UP:
				// Zero-up the button state
				buttonsMask = 0;
				break;
			case MotionEvent.ACTION_DOWN:
			case MotionEvent.ACTION_MOVE:
				if (buttonsMask == 0) {
					buttonsMask = MotionEvent.BUTTON_PRIMARY;
				}
				break;
		}

		// We don't handle ACTION_BUTTON_PRESS and ACTION_BUTTON_RELEASE events as they typically
		// follow ACTION_DOWN and ACTION_UP events. As such, handling them would result in duplicate
		// stream of events to the engine.
		switch (eventAction) {
			case MotionEvent.ACTION_CANCEL:
			case MotionEvent.ACTION_UP:
			case MotionEvent.ACTION_DOWN:
			case MotionEvent.ACTION_HOVER_ENTER:
			case MotionEvent.ACTION_HOVER_EXIT:
			case MotionEvent.ACTION_HOVER_MOVE:
			case MotionEvent.ACTION_MOVE:
			case MotionEvent.ACTION_SCROLL: {
				GodotLib.dispatchMouseEvent(eventAction, buttonsMask, x, y, deltaX, deltaY, doubleClick, sourceMouseRelative, pressure, tiltX, tiltY);
				return true;
			}
		}
		return false;
	}

	static boolean handleTouchEvent(final MotionEvent event) {
		final int pointerCount = event.getPointerCount();
		if (pointerCount == 0) {
			return true;
		}

		final float[] positions = new float[pointerCount * 3]; // pointerId1, x1, y1, pointerId2, etc...

		for (int i = 0; i < pointerCount; i++) {
			positions[i * 3 + 0] = event.getPointerId(i);
			positions[i * 3 + 1] = event.getX(i);
			positions[i * 3 + 2] = event.getY(i);
		}
		final int action = event.getActionMasked();
		final int actionPointerId = event.getPointerId(event.getActionIndex());

		return handleTouchEvent(action, actionPointerId, pointerCount, positions, false);
	}

	static boolean handleTouchEvent(int eventAction, float x, float y, boolean doubleTap) {
		return handleTouchEvent(eventAction, 0, 1, new float[] { 0, x, y }, doubleTap);
	}

	static boolean handleTouchEvent(int eventAction, int actionPointerId, int pointerCount, float[] positions, boolean doubleTap) {
		switch (eventAction) {
			case MotionEvent.ACTION_DOWN:
			case MotionEvent.ACTION_CANCEL:
			case MotionEvent.ACTION_UP:
			case MotionEvent.ACTION_MOVE:
			case MotionEvent.ACTION_POINTER_UP:
			case MotionEvent.ACTION_POINTER_DOWN: {
				GodotLib.dispatchTouchEvent(eventAction, actionPointerId, pointerCount, positions, doubleTap);
				return true;
			}
		}
		return false;
	}
}
