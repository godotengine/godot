/*************************************************************************/
/*  GodotInputHandler.java                                               */
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

package org.godotengine.godot.input;

import static org.godotengine.godot.utils.GLUtils.DEBUG;

import org.godotengine.godot.GodotLib;
import org.godotengine.godot.GodotView;
import org.godotengine.godot.input.InputManagerCompat.InputDeviceListener;

import android.os.Build;
import android.util.Log;
import android.view.InputDevice;
import android.view.InputDevice.MotionRange;
import android.view.KeyEvent;
import android.view.MotionEvent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Handles input related events for the {@link GodotView} view.
 */
public class GodotInputHandler implements InputDeviceListener {

	private final ArrayList<Joystick> joysticksDevices = new ArrayList<Joystick>();

	private final GodotView godotView;
	private final InputManagerCompat inputManager;

	public GodotInputHandler(GodotView godotView) {
		this.godotView = godotView;
		this.inputManager = InputManagerCompat.Factory.getInputManager(godotView.getContext());
		this.inputManager.registerInputDeviceListener(this, null);
	}

	private void queueEvent(Runnable task) {
		godotView.queueEvent(task);
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
		};

		int source = event.getSource();
		if (isKeyEvent_GameDevice(source)) {

			final int button = getGodotButton(keyCode);
			final int device_id = findJoystickDevice(event.getDeviceId());

			// Check if the device exists
			if (device_id > -1) {
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.joybutton(device_id, button, false);
					}
				});
			}
		} else {
			final int chr = event.getUnicodeChar(0);
			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.key(keyCode, chr, false);
				}
			});
		};

		return true;
	}

	public boolean onKeyDown(final int keyCode, KeyEvent event) {
		if (keyCode == KeyEvent.KEYCODE_BACK) {
			godotView.onBackPressed();
			// press 'back' button should not terminate program
			//normal handle 'back' event in game logic
			return true;
		}

		if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
			return false;
		};

		int source = event.getSource();
		//Log.e(TAG, String.format("Key down! source %d, device %d, joystick %d, %d, %d", event.getDeviceId(), source, (source & InputDevice.SOURCE_JOYSTICK), (source & InputDevice.SOURCE_DPAD), (source & InputDevice.SOURCE_GAMEPAD)));

		if (isKeyEvent_GameDevice(source)) {

			if (event.getRepeatCount() > 0) // ignore key echo
				return true;

			final int button = getGodotButton(keyCode);
			final int device_id = findJoystickDevice(event.getDeviceId());

			// Check if the device exists
			if (device_id > -1) {
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.joybutton(device_id, button, true);
					}
				});
			}
		} else {
			final int chr = event.getUnicodeChar(0);
			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.key(keyCode, chr, true);
				}
			});
		};

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

		if (godotView != null) {
			final float[] arr = new float[event.getPointerCount() * 3]; // pointerId1, x1, y1, pointerId2, etc...

			for (int i = 0; i < event.getPointerCount(); i++) {
				arr[i * 3 + 0] = event.getPointerId(i);
				arr[i * 3 + 1] = event.getX(i);
				arr[i * 3 + 2] = event.getY(i);
			}
			final int action = event.getActionMasked();
			final int pointer_idx = event.getPointerId(event.getActionIndex());

			godotView.queueEvent(new Runnable() {
				@Override
				public void run() {
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
			});
		}
		return true;
	}

	public boolean onGenericMotionEvent(MotionEvent event) {
		if ((event.getSource() & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK && event.getAction() == MotionEvent.ACTION_MOVE) {

			final int device_id = findJoystickDevice(event.getDeviceId());

			// Check if the device exists
			if (device_id > -1) {
				Joystick joy = joysticksDevices.get(device_id);

				for (int i = 0; i < joy.axes.size(); i++) {
					InputDevice.MotionRange range = joy.axes.get(i);
					final float value = (event.getAxisValue(range.getAxis()) - range.getMin()) / range.getRange() * 2.0f - 1.0f;
					final int idx = i;
					queueEvent(new Runnable() {
						@Override
						public void run() {
							GodotLib.joyaxis(device_id, idx, value);
						}
					});
				}

				for (int i = 0; i < joy.hats.size(); i += 2) {
					final int hatX = Math.round(event.getAxisValue(joy.hats.get(i).getAxis()));
					final int hatY = Math.round(event.getAxisValue(joy.hats.get(i + 1).getAxis()));
					queueEvent(new Runnable() {
						@Override
						public void run() {
							GodotLib.joyhat(device_id, hatX, hatY);
						}
					});
				}
				return true;
			}
		} else if ((event.getSource() & InputDevice.SOURCE_STYLUS) == InputDevice.SOURCE_STYLUS) {
			final float x = event.getX();
			final float y = event.getY();
			final int type = event.getAction();
			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.hover(type, x, y);
				}
			});
			return true;
		} else if ((event.getSource() & InputDevice.SOURCE_MOUSE) == InputDevice.SOURCE_MOUSE) {
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
				return handleMouseEvent(event);
			}
		}

		return false;
	}

	public void initInputDevices() {
		/* initially add input devices*/
		int[] deviceIds = inputManager.getInputDeviceIds();
		for (int deviceId : deviceIds) {
			InputDevice device = inputManager.getInputDevice(deviceId);
			if (DEBUG) {
				Log.v("GodotView", String.format("init() deviceId:%d, Name:%s\n", deviceId, device.getName()));
			}
			onInputDeviceAdded(deviceId);
		}
	}

	@Override
	public void onInputDeviceAdded(int deviceId) {
		int id = findJoystickDevice(deviceId);

		// Check if the device has not been already added
		if (id < 0) {
			InputDevice device = inputManager.getInputDevice(deviceId);
			//device can be null if deviceId is not found
			if (device != null) {
				int sources = device.getSources();
				if (((sources & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD) ||
						((sources & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK)) {
					id = joysticksDevices.size();

					Joystick joy = new Joystick();
					joy.device_id = deviceId;
					joy.name = device.getName();
					joy.axes = new ArrayList<InputDevice.MotionRange>();
					joy.hats = new ArrayList<InputDevice.MotionRange>();

					List<InputDevice.MotionRange> ranges = device.getMotionRanges();
					Collections.sort(ranges, new RangeComparator());

					for (InputDevice.MotionRange range : ranges) {
						if (range.getAxis() == MotionEvent.AXIS_HAT_X || range.getAxis() == MotionEvent.AXIS_HAT_Y) {
							joy.hats.add(range);
						} else {
							joy.axes.add(range);
						}
					}

					joysticksDevices.add(joy);

					final int device_id = id;
					final String name = joy.name;
					queueEvent(new Runnable() {
						@Override
						public void run() {
							GodotLib.joyconnectionchanged(device_id, true, name);
						}
					});
				}
			}
		}
	}

	@Override
	public void onInputDeviceRemoved(int deviceId) {
		final int device_id = findJoystickDevice(deviceId);

		// Check if the evice has not been already removed
		if (device_id > -1) {
			joysticksDevices.remove(device_id);

			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.joyconnectionchanged(device_id, false, "");
				}
			});
		}
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

	private int findJoystickDevice(int device_id) {
		for (int i = 0; i < joysticksDevices.size(); i++) {
			if (joysticksDevices.get(i).device_id == device_id) {
				return i;
			}
		}

		return -1;
	}

	private boolean handleMouseEvent(final MotionEvent event) {
		switch (event.getActionMasked()) {
			case MotionEvent.ACTION_HOVER_ENTER:
			case MotionEvent.ACTION_HOVER_MOVE:
			case MotionEvent.ACTION_HOVER_EXIT: {
				final float x = event.getX();
				final float y = event.getY();
				final int type = event.getAction();
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.hover(type, x, y);
					}
				});
				return true;
			}
			case MotionEvent.ACTION_BUTTON_PRESS:
			case MotionEvent.ACTION_BUTTON_RELEASE:
			case MotionEvent.ACTION_MOVE: {
				final float x = event.getX();
				final float y = event.getY();
				final int buttonsMask = event.getButtonState();
				final int action = event.getAction();
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.touch(event.getSource(), action, 0, 1, new float[] { 0, x, y }, buttonsMask);
					}
				});
				return true;
			}
			case MotionEvent.ACTION_SCROLL: {
				final float x = event.getX();
				final float y = event.getY();
				final int buttonsMask = event.getButtonState();
				final int action = event.getAction();
				final float verticalFactor = event.getAxisValue(MotionEvent.AXIS_VSCROLL);
				final float horizontalFactor = event.getAxisValue(MotionEvent.AXIS_HSCROLL);
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.touch(event.getSource(), action, 0, 1, new float[] { 0, x, y }, buttonsMask, verticalFactor, horizontalFactor);
					}
				});
			}
		}
		return false;
	}
}
