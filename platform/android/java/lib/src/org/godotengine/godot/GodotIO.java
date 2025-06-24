/**************************************************************************/
/*  GodotIO.java                                                          */
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

package org.godotengine.godot;

import org.godotengine.godot.error.Error;
import org.godotengine.godot.input.GodotEditText;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.DisplayCutout;
import android.view.Surface;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;

import androidx.core.content.FileProvider;

import java.io.File;
import java.util.List;
import java.util.Locale;

// Wrapper for native library

public class GodotIO {
	private static final String TAG = GodotIO.class.getSimpleName();

	private final Godot godot;

	private final String uniqueId;
	GodotEditText edit;

	final int SCREEN_LANDSCAPE = 0;
	final int SCREEN_PORTRAIT = 1;
	final int SCREEN_REVERSE_LANDSCAPE = 2;
	final int SCREEN_REVERSE_PORTRAIT = 3;
	final int SCREEN_SENSOR_LANDSCAPE = 4;
	final int SCREEN_SENSOR_PORTRAIT = 5;
	final int SCREEN_SENSOR = 6;

	GodotIO(Godot godot) {
		this.godot = godot;
		String androidId = Settings.Secure.getString(godot.getContext().getContentResolver(),
				Settings.Secure.ANDROID_ID);
		if (androidId == null) {
			androidId = "";
		}

		uniqueId = androidId;
	}

	private Context getContext() {
		Context context = godot.getActivity();
		if (context == null) {
			context = godot.getContext();
		}
		return context;
	}

	/////////////////////////
	// MISCELLANEOUS OS IO
	/////////////////////////

	public int openURI(String uriString) {
		try {
			Context context = getContext();

			Uri dataUri;
			String dataType = "";
			boolean grantReadUriPermission = false;

			if (uriString.startsWith("/") || uriString.startsWith("file://")) {
				String filePath = uriString;
				// File uris needs to be provided via the FileProvider
				grantReadUriPermission = true;
				if (filePath.startsWith("file://")) {
					filePath = filePath.replace("file://", "");
				}

				File targetFile = new File(filePath);
				dataUri = FileProvider.getUriForFile(context, context.getPackageName() + ".fileprovider", targetFile);
				dataType = context.getContentResolver().getType(dataUri);
			} else {
				dataUri = Uri.parse(uriString);
			}

			Intent intent = new Intent();
			intent.setAction(Intent.ACTION_VIEW).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
			if (TextUtils.isEmpty(dataType)) {
				intent.setData(dataUri);
			} else {
				intent.setDataAndType(dataUri, dataType);
			}
			if (grantReadUriPermission) {
				intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
			}

			context.startActivity(intent);
			return Error.OK.toNativeValue();
		} catch (Exception e) {
			Log.e(TAG, "Unable to open uri " + uriString, e);
			return Error.FAILED.toNativeValue();
		}
	}

	public String getCacheDir() {
		return getContext().getCacheDir().getAbsolutePath();
	}

	public String getTempDir() {
		File tempDir = new File(getCacheDir() + "/tmp");

		if (!tempDir.exists()) {
			if (!tempDir.mkdirs()) {
				Log.e(TAG, "Unable to create temp dir");
			}
		}

		return tempDir.getAbsolutePath();
	}

	public String getDataDir() {
		return getContext().getFilesDir().getAbsolutePath();
	}

	public String getLocale() {
		return Locale.getDefault().toString();
	}

	public String getModel() {
		return Build.MODEL;
	}

	public int getScreenDPI() {
		return getContext().getResources().getDisplayMetrics().densityDpi;
	}

	/**
	 * Returns bucketized density values.
	 */
	public float getScaledDensity() {
		int densityDpi = getContext().getResources().getDisplayMetrics().densityDpi;
		float selectedScaledDensity;
		if (densityDpi >= DisplayMetrics.DENSITY_XXXHIGH) {
			selectedScaledDensity = 4.0f;
		} else if (densityDpi >= DisplayMetrics.DENSITY_XXHIGH) {
			selectedScaledDensity = 3.0f;
		} else if (densityDpi >= DisplayMetrics.DENSITY_XHIGH) {
			selectedScaledDensity = 2.0f;
		} else if (densityDpi >= DisplayMetrics.DENSITY_HIGH) {
			selectedScaledDensity = 1.5f;
		} else if (densityDpi >= DisplayMetrics.DENSITY_MEDIUM) {
			selectedScaledDensity = 1.0f;
		} else {
			selectedScaledDensity = 0.75f;
		}
		return selectedScaledDensity;
	}

	public double getScreenRefreshRate(double fallback) {
		Activity activity = godot.getActivity();

		Display display = null;
		if (activity != null) {
			display = activity.getWindowManager().getDefaultDisplay();
		} else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
			display = godot.getContext().getDisplay();
		}

		if (display != null) {
			return display.getRefreshRate();
		}
		return fallback;
	}

	public int[] getDisplaySafeArea() {
		Rect rect = new Rect();
		int[] result = new int[4];

		View topView = null;
		if (godot.getActivity() != null) {
			topView = godot.getActivity().getWindow().getDecorView();
		} else if (godot.getRenderView() != null) {
			topView = godot.getRenderView().getView();
		}

		if (topView != null) {
			topView.getWindowVisibleDisplayFrame(rect);
			result[0] = rect.left;
			result[1] = rect.top;
			result[2] = rect.right;
			result[3] = rect.bottom;

			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
				WindowInsets insets = topView.getRootWindowInsets();
				DisplayCutout cutout = insets.getDisplayCutout();
				if (cutout != null) {
					int insetLeft = cutout.getSafeInsetLeft();
					int insetTop = cutout.getSafeInsetTop();
					result[0] = insetLeft;
					result[1] = insetTop;
					result[2] -= insetLeft + cutout.getSafeInsetRight();
					result[3] -= insetTop + cutout.getSafeInsetBottom();
				}
			}
		}
		return result;
	}

	public int[] getDisplayCutouts() {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
			return new int[0];
		}

		View topView = null;
		if (godot.getActivity() != null) {
			topView = godot.getActivity().getWindow().getDecorView();
		} else if (godot.getRenderView() != null) {
			topView = godot.getRenderView().getView();
		}

		if (topView == null) {
			return new int[0];
		}
		DisplayCutout cutout = topView.getRootWindowInsets().getDisplayCutout();
		if (cutout == null) {
			return new int[0];
		}
		List<Rect> rects = cutout.getBoundingRects();
		int cutouts = rects.size();
		int[] result = new int[cutouts * 4];
		int index = 0;
		for (Rect rect : rects) {
			result[index++] = rect.left;
			result[index++] = rect.top;
			result[index++] = rect.width();
			result[index++] = rect.height();
		}
		return result;
	}

	public boolean hasHardwareKeyboard() {
		if (edit != null) {
			return edit.hasHardwareKeyboard();
		} else {
			return false;
		}
	}

	public void showKeyboard(String p_existing_text, int p_type, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
		if (edit != null) {
			edit.showKeyboard(p_existing_text, GodotEditText.VirtualKeyboardType.values()[p_type], p_max_input_length, p_cursor_start, p_cursor_end);
		}
	}

	public void hideKeyboard() {
		if (edit != null)
			edit.hideKeyboard();
	}

	public void setScreenOrientation(int p_orientation) {
		final Activity activity = godot.getActivity();
		if (activity == null) {
			return;
		}

		switch (p_orientation) {
			case SCREEN_LANDSCAPE: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
			} break;
			case SCREEN_PORTRAIT: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
			} break;
			case SCREEN_REVERSE_LANDSCAPE: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE);
			} break;
			case SCREEN_REVERSE_PORTRAIT: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT);
			} break;
			case SCREEN_SENSOR_LANDSCAPE: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_USER_LANDSCAPE);
			} break;
			case SCREEN_SENSOR_PORTRAIT: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_USER_PORTRAIT);
			} break;
			case SCREEN_SENSOR: {
				activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_FULL_USER);
			} break;
		}
	}

	public int getScreenOrientation() {
		final Activity activity = godot.getActivity();
		if (activity == null) {
			return -1;
		}

		int orientation = activity.getRequestedOrientation();
		switch (orientation) {
			case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
				return SCREEN_LANDSCAPE;
			case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
				return SCREEN_PORTRAIT;
			case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
				return SCREEN_REVERSE_LANDSCAPE;
			case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
				return SCREEN_REVERSE_PORTRAIT;
			case ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE:
			case ActivityInfo.SCREEN_ORIENTATION_USER_LANDSCAPE:
				return SCREEN_SENSOR_LANDSCAPE;
			case ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT:
			case ActivityInfo.SCREEN_ORIENTATION_USER_PORTRAIT:
				return SCREEN_SENSOR_PORTRAIT;
			case ActivityInfo.SCREEN_ORIENTATION_SENSOR:
			case ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR:
			case ActivityInfo.SCREEN_ORIENTATION_FULL_USER:
				return SCREEN_SENSOR;
			case ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED:
			case ActivityInfo.SCREEN_ORIENTATION_USER:
			case ActivityInfo.SCREEN_ORIENTATION_BEHIND:
			case ActivityInfo.SCREEN_ORIENTATION_NOSENSOR:
			case ActivityInfo.SCREEN_ORIENTATION_LOCKED:
			default:
				return -1;
		}
	}

	public int getDisplayRotation() {
		Activity activity = godot.getActivity();

		Display display = null;
		if (activity != null) {
			display = activity.getWindowManager().getDefaultDisplay();
		} else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
			display = godot.getContext().getDisplay();
		}

		if (display != null) {
			int rotation = display.getRotation();
			if (rotation == Surface.ROTATION_90) {
				return 90;
			} else if (rotation == Surface.ROTATION_180) {
				return 180;
			} else if (rotation == Surface.ROTATION_270) {
				return 270;
			}
		}
		return 0;
	}

	public void setEdit(GodotEditText _edit) {
		edit = _edit;
	}

	public static final int SYSTEM_DIR_DESKTOP = 0;
	public static final int SYSTEM_DIR_DCIM = 1;
	public static final int SYSTEM_DIR_DOCUMENTS = 2;
	public static final int SYSTEM_DIR_DOWNLOADS = 3;
	public static final int SYSTEM_DIR_MOVIES = 4;
	public static final int SYSTEM_DIR_MUSIC = 5;
	public static final int SYSTEM_DIR_PICTURES = 6;
	public static final int SYSTEM_DIR_RINGTONES = 7;

	public String getSystemDir(int idx, boolean shared_storage) {
		String what;
		switch (idx) {
			case SYSTEM_DIR_DESKTOP:
			default: {
				what = null; // This leads to the app specific external root directory.
			} break;

			case SYSTEM_DIR_DCIM: {
				what = Environment.DIRECTORY_DCIM;
			} break;

			case SYSTEM_DIR_DOCUMENTS: {
				what = Environment.DIRECTORY_DOCUMENTS;
			} break;

			case SYSTEM_DIR_DOWNLOADS: {
				what = Environment.DIRECTORY_DOWNLOADS;
			} break;

			case SYSTEM_DIR_MOVIES: {
				what = Environment.DIRECTORY_MOVIES;
			} break;

			case SYSTEM_DIR_MUSIC: {
				what = Environment.DIRECTORY_MUSIC;
			} break;

			case SYSTEM_DIR_PICTURES: {
				what = Environment.DIRECTORY_PICTURES;
			} break;

			case SYSTEM_DIR_RINGTONES: {
				what = Environment.DIRECTORY_RINGTONES;
			} break;
		}

		if (shared_storage) {
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
				Log.w(TAG, "Shared storage access is limited on Android 10 and higher.");
			}
			if (TextUtils.isEmpty(what)) {
				return Environment.getExternalStorageDirectory().getAbsolutePath();
			} else {
				return Environment.getExternalStoragePublicDirectory(what).getAbsolutePath();
			}
		} else {
			return getContext().getExternalFilesDir(what).getAbsolutePath();
		}
	}

	public String getUniqueID() {
		return uniqueId;
	}
}
