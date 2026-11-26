/**************************************************************************/
/*  SDLActivity.java                                                      */
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

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.UiModeManager;
import android.content.ActivityNotFoundException;
import android.content.ClipboardManager;
import android.content.ClipData;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.hardware.Sensor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.LocaleList;
import android.os.Message;
import android.os.ParcelFileDescriptor;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.SparseArray;
import android.view.Display;
import android.view.Gravity;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.PointerIcon;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodManager;
import android.webkit.MimeTypeMap;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Locale;

// Stub class for SDL activity
public class SDLActivity {
	// Code from SDL:
	protected static SDLGenericMotionListener_API14 mMotionListener;
	protected static HIDDeviceManager mHIDDeviceManager;

	public static SDLGenericMotionListener_API14 getMotionListener() {
		if (mMotionListener == null) {
			if (Build.VERSION.SDK_INT >= 29 /* Android 10 (Q) */) {
				mMotionListener = new SDLGenericMotionListener_API29();
			} else if (Build.VERSION.SDK_INT >= 26 /* Android 8.0 (O) */) {
				mMotionListener = new SDLGenericMotionListener_API26();
			} else if (Build.VERSION.SDK_INT >= 24 /* Android 7.0 (N) */) {
				mMotionListener = new SDLGenericMotionListener_API24();
			} else {
				mMotionListener = new SDLGenericMotionListener_API14();
			}
		}

		return mMotionListener;
	}

	public static boolean handleKeyEvent(View v, int keyCode, KeyEvent event, InputConnection ic) {
		int deviceId = event.getDeviceId();

		if (SDLControllerManager.isDeviceSDLJoystick(deviceId)) {
			// Note that we process events with specific key codes here
			if (event.getAction() == KeyEvent.ACTION_DOWN) {
				if (SDLControllerManager.onNativePadDown(deviceId, keyCode, event.getScanCode())) {
					return true;
				}
			} else if (event.getAction() == KeyEvent.ACTION_UP) {
				if (SDLControllerManager.onNativePadUp(deviceId, keyCode, event.getScanCode())) {
					return true;
				}
			}
		}

		return false;
	}

	// Code from SDL
	public static boolean isChromebook() {
		// https://stackoverflow.com/questions/39784415/how-to-detect-programmatically-if-android-app-is-running-in-chrome-book-or-in
		if (getContext() != null) {
			if (getContext().getPackageManager().hasSystemFeature("org.chromium.arc") || getContext().getPackageManager().hasSystemFeature("org.chromium.arc.device_management")) {
				return true;
			}
		}

		// Running on AVD emulator
		boolean isChromebookEmulator = (Build.MODEL != null && Build.MODEL.startsWith("sdk_gpc_"));
		return isChromebookEmulator;
	}

	public static Context getContext() {
		return SDL.getContext();
	}

	public static void onCreate(Context context) {
		// Set up JNI
		SDL.setupJNI();

		// Initialize state
		SDL.initialize();

		mHIDDeviceManager = HIDDeviceManager.acquire(context);
	}

	public static void onNativeMouse(int button, int action, float x, float y, boolean relative) {
		// no-op
	}

	public static void onNativePen(int penId, int device_type, int button, int action, float x, float y, float p) {
		// no-op
	}

	public static boolean isDeXMode() {
		return false;
	}

	static class ViewStub {
		public void requestPointerCapture() {}
		public void releasePointerCapture() {}
	}

	public static ViewStub getContentView() {
		return new ViewStub();
	}
}
