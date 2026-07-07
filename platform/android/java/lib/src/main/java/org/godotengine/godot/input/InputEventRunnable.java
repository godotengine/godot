/**************************************************************************/
/*  InputEventRunnable.java                                               */
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

import org.godotengine.godot.GodotLib;

import android.hardware.Sensor;
import android.util.Log;
import android.view.MotionEvent;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.util.Pools;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Used to dispatch input events.
 *
 * This is a specialized version of @{@link Runnable} which allows to allocate a finite pool of
 * objects for input events dispatching, thus avoid the creation (and garbage collection) of
 * spurious @{@link Runnable} objects.
 */
final class InputEventRunnable implements Runnable {
	private static final String TAG = InputEventRunnable.class.getSimpleName();

	private static final int MAX_TOUCH_POINTER_COUNT = 10; // assuming 10 fingers as max supported concurrent touch pointers

	private static final Pools.Pool<InputEventRunnable> POOL = new Pools.Pool<>() {
		private static final int MAX_POOL_SIZE = 120 * 10; // up to 120Hz input events rate for up to 5 secs (ANR limit) * 2

		private final ArrayBlockingQueue<InputEventRunnable> queue = new ArrayBlockingQueue<>(MAX_POOL_SIZE);
		private final AtomicInteger createdCount = new AtomicInteger();

		@Nullable
		@Override
		public InputEventRunnable acquire() {
			InputEventRunnable instance = queue.poll();
			if (instance == null) {
				int creationCount = createdCount.incrementAndGet();
				if (creationCount <= MAX_POOL_SIZE) {
					instance = new InputEventRunnable(creationCount - 1);
				}
			}

			return instance;
		}

		@Override
		public boolean release(@NonNull InputEventRunnable instance) {
			return queue.offer(instance);
		}
	};

	@Nullable
	static InputEventRunnable obtain() {
		InputEventRunnable runnable = POOL.acquire();
		if (runnable == null) {
			Log.w(TAG, "Input event pool is at capacity");
		}
		return runnable;
	}

	/**
	 * Used to track when this instance was created and added to the pool. Primarily used for
	 * debug purposes.
	 */
	private final int creationRank;

	private InputEventRunnable(int creationRank) {
		this.creationRank = creationRank;
	}

	/**
	 * Set of supported input events.
	 */
	private enum EventType {
		MOUSE,
		TOUCH,
		MAGNIFY,
		PAN,
		JOYSTICK_BUTTON,
		JOYSTICK_AXIS,
		JOYSTICK_HAT,
		JOYSTICK_CONNECTION_CHANGED,
		KEY,
		SENSOR
	}

	private EventType currentEventType = null;

	// common event fields
	private float eventX;
	private float eventY;
	private float eventDeltaX;
	private float eventDeltaY;
	private boolean eventPressed;

	// common touch / mouse fields
	private int eventAction;
	private boolean doubleTap;

	// Mouse event fields and setter
	private int buttonsMask;
	private boolean sourceMouseRelative;
	private float pressure;
	private float tiltX;
	private float tiltY;
	void setMouseEvent(int eventAction, int buttonsMask, float x, float y, float deltaX, float deltaY, boolean doubleClick, boolean sourceMouseRelative, float pressure, float tiltX, float tiltY) {
		this.currentEventType = EventType.MOUSE;
		this.eventAction = eventAction;
		this.buttonsMask = buttonsMask;
		this.eventX = x;
		this.eventY = y;
		this.eventDeltaX = deltaX;
		this.eventDeltaY = deltaY;
		this.doubleTap = doubleClick;
		this.sourceMouseRelative = sourceMouseRelative;
		this.pressure = pressure;
		this.tiltX = tiltX;
		this.tiltY = tiltY;
	}

	// Touch event fields and setter
	private int actionPointerId;
	private int pointerCount;
	private final float[] positions = new float[MAX_TOUCH_POINTER_COUNT * 6]; // pointerId1, x1, y1, pressure1, tiltX1, tiltY1, pointerId2, etc...
	void setTouchEvent(MotionEvent event, int eventAction, boolean doubleTap) {
		this.currentEventType = EventType.TOUCH;
		this.eventAction = eventAction;
		this.doubleTap = doubleTap;
		this.actionPointerId = event.getPointerId(event.getActionIndex());
		this.pointerCount = Math.min(event.getPointerCount(), MAX_TOUCH_POINTER_COUNT);
		for (int i = 0; i < pointerCount; i++) {
			positions[i * 6 + 0] = event.getPointerId(i);
			positions[i * 6 + 1] = event.getX(i);
			positions[i * 6 + 2] = event.getY(i);
			positions[i * 6 + 3] = event.getPressure(i);
			positions[i * 6 + 4] = GodotInputHandler.getEventTiltX(event);
			positions[i * 6 + 5] = GodotInputHandler.getEventTiltY(event);
		}
	}

	// Magnify event fields and setter
	private float magnifyFactor;
	void setMagnifyEvent(float x, float y, float factor) {
		this.currentEventType = EventType.MAGNIFY;
		this.eventX = x;
		this.eventY = y;
		this.magnifyFactor = factor;
	}

	// Pan event setter
	void setPanEvent(float x, float y, float deltaX, float deltaY) {
		this.currentEventType = EventType.PAN;
		this.eventX = x;
		this.eventY = y;
		this.eventDeltaX = deltaX;
		this.eventDeltaY = deltaY;
	}

	// common joystick field
	private int joystickDevice;

	// Joystick button event fields and setter
	private int button;
	void setJoystickButtonEvent(int device, int button, boolean pressed) {
		this.currentEventType = EventType.JOYSTICK_BUTTON;
		this.joystickDevice = device;
		this.button = button;
		this.eventPressed = pressed;
	}

	// Joystick axis event fields and setter
	private int axis;
	private float value;
	void setJoystickAxisEvent(int device, int axis, float value) {
		this.currentEventType = EventType.JOYSTICK_AXIS;
		this.joystickDevice = device;
		this.axis = axis;
		this.value = value;
	}

	// Joystick hat event fields and setter
	private int hatX;
	private int hatY;
	void setJoystickHatEvent(int device, int hatX, int hatY) {
		this.currentEventType = EventType.JOYSTICK_HAT;
		this.joystickDevice = device;
		this.hatX = hatX;
		this.hatY = hatY;
	}

	// Joystick connection changed event fields and setter
	private boolean connected;
	private String joystickName;
	void setJoystickConnectionChangedEvent(int device, boolean connected, String name) {
		this.currentEventType = EventType.JOYSTICK_CONNECTION_CHANGED;
		this.joystickDevice = device;
		this.connected = connected;
		this.joystickName = name;
	}

	// Key event fields and setter
	private int physicalKeycode;
	private int unicode;
	private int keyLabel;
	private boolean echo;
	void setKeyEvent(int physicalKeycode, int unicode, int keyLabel, boolean pressed, boolean echo) {
		this.currentEventType = EventType.KEY;
		this.physicalKeycode = physicalKeycode;
		this.unicode = unicode;
		this.keyLabel = keyLabel;
		this.eventPressed = pressed;
		this.echo = echo;
	}

	// Sensor event fields and setter
	private int sensorType;
	private float rotatedValue0;
	private float rotatedValue1;
	private float rotatedValue2;
	void setSensorEvent(int sensorType, float rotatedValue0, float rotatedValue1, float rotatedValue2) {
		this.currentEventType = EventType.SENSOR;
		this.sensorType = sensorType;
		this.rotatedValue0 = rotatedValue0;
		this.rotatedValue1 = rotatedValue1;
		this.rotatedValue2 = rotatedValue2;
	}

	@Override
	public void run() {
		try {
			if (currentEventType == null) {
				Log.w(TAG, "Invalid event type");
				return;
			}

			switch (currentEventType) {
				case MOUSE:
					GodotLib.dispatchMouseEvent(
							eventAction,
							buttonsMask,
							eventX,
							eventY,
							eventDeltaX,
							eventDeltaY,
							doubleTap,
							sourceMouseRelative,
							pressure,
							tiltX,
							tiltY);
					break;

				case TOUCH:
					GodotLib.dispatchTouchEvent(
							eventAction,
							actionPointerId,
							pointerCount,
							positions,
							doubleTap);
					break;

				case MAGNIFY:
					GodotLib.magnify(eventX, eventY, magnifyFactor);
					break;

				case PAN:
					GodotLib.pan(eventX, eventY, eventDeltaX, eventDeltaY);
					break;

				case JOYSTICK_BUTTON:
					GodotLib.joybutton(joystickDevice, button, eventPressed);
					break;

				case JOYSTICK_AXIS:
					GodotLib.joyaxis(joystickDevice, axis, value);
					break;

				case JOYSTICK_HAT:
					GodotLib.joyhat(joystickDevice, hatX, hatY);
					break;

				case JOYSTICK_CONNECTION_CHANGED:
					GodotLib.joyconnectionchanged(joystickDevice, connected, joystickName);
					break;

				case KEY:
					GodotLib.key(physicalKeycode, unicode, keyLabel, eventPressed, echo);
					break;

				case SENSOR:
					switch (sensorType) {
						case Sensor.TYPE_ACCELEROMETER:
							GodotLib.accelerometer(-rotatedValue0, -rotatedValue1, -rotatedValue2);
							break;

						case Sensor.TYPE_GRAVITY:
							GodotLib.gravity(-rotatedValue0, -rotatedValue1, -rotatedValue2);
							break;

						case Sensor.TYPE_MAGNETIC_FIELD:
							GodotLib.magnetometer(-rotatedValue0, -rotatedValue1, -rotatedValue2);
							break;

						case Sensor.TYPE_GYROSCOPE:
							GodotLib.gyroscope(rotatedValue0, rotatedValue1, rotatedValue2);
							break;
					}
					break;
			}
		} finally {
			recycle();
		}
	}

	/**
	 * Release the current instance back to the pool
	 */
	private void recycle() {
		currentEventType = null;
		POOL.release(this);
	}
}
