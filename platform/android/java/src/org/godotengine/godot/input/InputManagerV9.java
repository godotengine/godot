/*
 * Copyright (C) 2013 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.godotengine.godot.input;

import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.Log;
import android.util.SparseArray;
import android.view.InputDevice;
import android.view.MotionEvent;

import java.lang.ref.WeakReference;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;

public class InputManagerV9 implements InputManagerCompat {
	private static final String LOG_TAG = "InputManagerV9";
	private static final int MESSAGE_TEST_FOR_DISCONNECT = 101;
	private static final long CHECK_ELAPSED_TIME = 3000L;

	private static final int ON_DEVICE_ADDED = 0;
	private static final int ON_DEVICE_CHANGED = 1;
	private static final int ON_DEVICE_REMOVED = 2;

	private final SparseArray<long[]> mDevices;
	private final Map<InputDeviceListener, Handler> mListeners;
	private final Handler mDefaultHandler;

	private static class PollingMessageHandler extends Handler {
		private final WeakReference<InputManagerV9> mInputManager;

		PollingMessageHandler(InputManagerV9 im) {
			mInputManager = new WeakReference<InputManagerV9>(im);
		}

		@Override
		public void handleMessage(Message msg) {
			super.handleMessage(msg);
			switch (msg.what) {
				case MESSAGE_TEST_FOR_DISCONNECT:
					InputManagerV9 imv = mInputManager.get();
					if (null != imv) {
						long time = SystemClock.elapsedRealtime();
						int size = imv.mDevices.size();
						for (int i = 0; i < size; i++) {
							long[] lastContact = imv.mDevices.valueAt(i);
							if (null != lastContact) {
								if (time - lastContact[0] > CHECK_ELAPSED_TIME) {
									// check to see if the device has been
									// disconnected
									int id = imv.mDevices.keyAt(i);
									if (null == InputDevice.getDevice(id)) {
										// disconnected!
										imv.notifyListeners(ON_DEVICE_REMOVED, id);
										imv.mDevices.remove(id);
									} else {
										lastContact[0] = time;
									}
								}
							}
						}
						sendEmptyMessageDelayed(MESSAGE_TEST_FOR_DISCONNECT,
								CHECK_ELAPSED_TIME);
					}
					break;
			}
		}
	}

	public InputManagerV9() {
		mDevices = new SparseArray<long[]>();
		mListeners = new HashMap<InputDeviceListener, Handler>();
		mDefaultHandler = new PollingMessageHandler(this);
		// as a side-effect, populates our collection of watched
		// input devices
		getInputDeviceIds();
	}

	@Override
	public InputDevice getInputDevice(int id) {
		return InputDevice.getDevice(id);
	}

	@Override
	public int[] getInputDeviceIds() {
		// add any hitherto unknown devices to our
		// collection of watched input devices
		int[] activeDevices = InputDevice.getDeviceIds();
		long time = SystemClock.elapsedRealtime();
		for (int id : activeDevices) {
			long[] lastContact = mDevices.get(id);
			if (null == lastContact) {
				// we have a new device
				mDevices.put(id, new long[] { time });
			}
		}
		return activeDevices;
	}

	@Override
	public void registerInputDeviceListener(InputDeviceListener listener, Handler handler) {
		mListeners.remove(listener);
		if (handler == null) {
			handler = mDefaultHandler;
		}
		mListeners.put(listener, handler);
	}

	@Override
	public void unregisterInputDeviceListener(InputDeviceListener listener) {
		mListeners.remove(listener);
	}

	private void notifyListeners(int why, int deviceId) {
		// the state of some device has changed
		if (!mListeners.isEmpty()) {
			// yes... this will cause an object to get created... hopefully
			// it won't happen very often
			for (InputDeviceListener listener : mListeners.keySet()) {
				Handler handler = mListeners.get(listener);
				DeviceEvent odc = DeviceEvent.getDeviceEvent(why, deviceId, listener);
				handler.post(odc);
			}
		}
	}

	private static class DeviceEvent implements Runnable {
		private int mMessageType;
		private int mId;
		private InputDeviceListener mListener;
		private static Queue<DeviceEvent> sEventQueue = new ArrayDeque<DeviceEvent>();

		private DeviceEvent() {
		}

		static DeviceEvent getDeviceEvent(int messageType, int id,
				InputDeviceListener listener) {
			DeviceEvent curChanged = sEventQueue.poll();
			if (null == curChanged) {
				curChanged = new DeviceEvent();
			}
			curChanged.mMessageType = messageType;
			curChanged.mId = id;
			curChanged.mListener = listener;
			return curChanged;
		}

		@Override
		public void run() {
			switch (mMessageType) {
				case ON_DEVICE_ADDED:
					mListener.onInputDeviceAdded(mId);
					break;
				case ON_DEVICE_CHANGED:
					mListener.onInputDeviceChanged(mId);
					break;
				case ON_DEVICE_REMOVED:
					mListener.onInputDeviceRemoved(mId);
					break;
				default:
					Log.e(LOG_TAG, "Unknown Message Type");
					break;
			}
			// dump this runnable back in the queue
			sEventQueue.offer(this);
		}
	}

	@Override
	public void onGenericMotionEvent(MotionEvent event) {
		// detect new devices
		int id = event.getDeviceId();
		long[] timeArray = mDevices.get(id);
		if (null == timeArray) {
			notifyListeners(ON_DEVICE_ADDED, id);
			timeArray = new long[1];
			mDevices.put(id, timeArray);
		}
		long time = SystemClock.elapsedRealtime();
		timeArray[0] = time;
	}

	@Override
	public void onPause() {
		mDefaultHandler.removeMessages(MESSAGE_TEST_FOR_DISCONNECT);
	}

	@Override
	public void onResume() {
		mDefaultHandler.sendEmptyMessage(MESSAGE_TEST_FOR_DISCONNECT);
	}
}
