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

import android.annotation.TargetApi;
import android.content.Context;
import android.hardware.input.InputManager;
import android.os.Build;
import android.os.Handler;
import android.view.InputDevice;
import android.view.MotionEvent;

import java.util.HashMap;
import java.util.Map;

@TargetApi(Build.VERSION_CODES.JELLY_BEAN)
public class InputManagerV16 implements InputManagerCompat {

	private final InputManager mInputManager;
	private final Map<InputManagerCompat.InputDeviceListener, V16InputDeviceListener> mListeners;

	public InputManagerV16(Context context) {
		mInputManager = (InputManager)context.getSystemService(Context.INPUT_SERVICE);
		mListeners = new HashMap<InputManagerCompat.InputDeviceListener, V16InputDeviceListener>();
	}

	@Override
	public InputDevice getInputDevice(int id) {
		return mInputManager.getInputDevice(id);
	}

	@Override
	public int[] getInputDeviceIds() {
		return mInputManager.getInputDeviceIds();
	}

	static class V16InputDeviceListener implements InputManager.InputDeviceListener {
		final InputManagerCompat.InputDeviceListener mIDL;

		public V16InputDeviceListener(InputDeviceListener idl) {
			mIDL = idl;
		}

		@Override
		public void onInputDeviceAdded(int deviceId) {
			mIDL.onInputDeviceAdded(deviceId);
		}

		@Override
		public void onInputDeviceChanged(int deviceId) {
			mIDL.onInputDeviceChanged(deviceId);
		}

		@Override
		public void onInputDeviceRemoved(int deviceId) {
			mIDL.onInputDeviceRemoved(deviceId);
		}
	}

	@Override
	public void registerInputDeviceListener(InputDeviceListener listener, Handler handler) {
		V16InputDeviceListener v16Listener = new V16InputDeviceListener(listener);
		mInputManager.registerInputDeviceListener(v16Listener, handler);
		mListeners.put(listener, v16Listener);
	}

	@Override
	public void unregisterInputDeviceListener(InputDeviceListener listener) {
		V16InputDeviceListener curListener = mListeners.remove(listener);
		if (null != curListener) {
			mInputManager.unregisterInputDeviceListener(curListener);
		}
	}

	@Override
	public void onGenericMotionEvent(MotionEvent event) {
		// unused in V16
	}

	@Override
	public void onPause() {
		// unused in V16
	}

	@Override
	public void onResume() {
		// unused in V16
	}
}
