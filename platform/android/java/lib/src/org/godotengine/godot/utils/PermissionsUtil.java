/**************************************************************************/
/*  PermissionsUtil.java                                                  */
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

package org.godotengine.godot.utils;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PermissionInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * This class includes utility functions for Android permissions related operations.
 * @author Cagdas Caglak <cagdascaglak@gmail.com>
 */
public final class PermissionsUtil {
	private static final String TAG = PermissionsUtil.class.getSimpleName();

	static final int REQUEST_RECORD_AUDIO_PERMISSION = 1;
	static final int REQUEST_CAMERA_PERMISSION = 2;
	static final int REQUEST_VIBRATE_PERMISSION = 3;
	public static final int REQUEST_ALL_PERMISSION_REQ_CODE = 1001;
	public static final int REQUEST_SINGLE_PERMISSION_REQ_CODE = 1002;
	public static final int REQUEST_MANAGE_EXTERNAL_STORAGE_REQ_CODE = 2002;

	private PermissionsUtil() {
	}

	/**
	 * Request a dangerous permission. name must be specified in <a href="https://github.com/aosp-mirror/platform_frameworks_base/blob/master/core/res/AndroidManifest.xml">this</a>
	 * @param permissionName the name of the requested permission.
	 * @param activity the caller activity for this method.
	 * @return true/false. "true" if permission was granted otherwise returns "false".
	 */
	public static boolean requestPermission(String permissionName, Activity activity) {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
			// Not necessary, asked on install already
			return true;
		}

		switch (permissionName) {
			case "RECORD_AUDIO":
			case Manifest.permission.RECORD_AUDIO:
				if (ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
					activity.requestPermissions(new String[] { Manifest.permission.RECORD_AUDIO }, REQUEST_RECORD_AUDIO_PERMISSION);
					return false;
				}
				return true;

			case "CAMERA":
			case Manifest.permission.CAMERA:
				if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
					activity.requestPermissions(new String[] { Manifest.permission.CAMERA }, REQUEST_CAMERA_PERMISSION);
					return false;
				}
				return true;

			case "VIBRATE":
			case Manifest.permission.VIBRATE:
				if (ContextCompat.checkSelfPermission(activity, Manifest.permission.VIBRATE) != PackageManager.PERMISSION_GRANTED) {
					activity.requestPermissions(new String[] { Manifest.permission.VIBRATE }, REQUEST_VIBRATE_PERMISSION);
					return false;
				}
				return true;

			default:
				// Check if the given permission is a dangerous permission
				try {
					PermissionInfo permissionInfo = getPermissionInfo(activity, permissionName);
					int protectionLevel = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ? permissionInfo.getProtection() : permissionInfo.protectionLevel;
					if (protectionLevel == PermissionInfo.PROTECTION_DANGEROUS && ContextCompat.checkSelfPermission(activity, permissionName) != PackageManager.PERMISSION_GRANTED) {
						activity.requestPermissions(new String[] { permissionName }, REQUEST_SINGLE_PERMISSION_REQ_CODE);
						return false;
					}
				} catch (PackageManager.NameNotFoundException e) {
					// Unknown permission - return false as it can't be granted.
					Log.w(TAG, "Unable to identify permission " + permissionName, e);
					return false;
				}
				return true;
		}
	}

	/**
	 * Request dangerous permissions which are defined in the Android manifest file from the user.
	 * @param activity the caller activity for this method.
	 * @return true/false. "true" if all permissions were granted otherwise returns "false".
	 */
	public static boolean requestManifestPermissions(Activity activity) {
		return requestManifestPermissions(activity, null);
	}

	/**
	 * Request dangerous permissions which are defined in the Android manifest file from the user.
	 * @param activity the caller activity for this method.
	 * @param excludes Set of permissions to exclude from the request
	 * @return true/false. "true" if all permissions were granted otherwise returns "false".
	 */
	public static boolean requestManifestPermissions(Activity activity, @Nullable Set<String> excludes) {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
			return true;
		}

		String[] manifestPermissions;
		try {
			manifestPermissions = getManifestPermissions(activity);
		} catch (PackageManager.NameNotFoundException e) {
			e.printStackTrace();
			return false;
		}

		if (manifestPermissions.length == 0)
			return true;

		List<String> requestedPermissions = new ArrayList<>();
		for (String manifestPermission : manifestPermissions) {
			if (excludes != null && excludes.contains(manifestPermission)) {
				continue;
			}
			try {
				if (manifestPermission.equals(Manifest.permission.MANAGE_EXTERNAL_STORAGE)) {
					if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
						try {
							Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
							intent.setData(Uri.parse(String.format("package:%s", activity.getPackageName())));
							activity.startActivityForResult(intent, REQUEST_MANAGE_EXTERNAL_STORAGE_REQ_CODE);
						} catch (Exception ignored) {
							Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
							activity.startActivityForResult(intent, REQUEST_MANAGE_EXTERNAL_STORAGE_REQ_CODE);
						}
					}
				} else {
					PermissionInfo permissionInfo = getPermissionInfo(activity, manifestPermission);
					int protectionLevel = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ? permissionInfo.getProtection() : permissionInfo.protectionLevel;
					if (protectionLevel == PermissionInfo.PROTECTION_DANGEROUS && ContextCompat.checkSelfPermission(activity, manifestPermission) != PackageManager.PERMISSION_GRANTED) {
						requestedPermissions.add(manifestPermission);
					}
				}
			} catch (PackageManager.NameNotFoundException e) {
				// Skip this permission and continue.
				Log.w(TAG, "Unable to identify permission " + manifestPermission, e);
			}
		}

		if (requestedPermissions.isEmpty()) {
			// If list is empty, all of dangerous permissions were granted.
			return true;
		}

		activity.requestPermissions(requestedPermissions.toArray(new String[0]), REQUEST_ALL_PERMISSION_REQ_CODE);
		return false;
	}

	/**
	 * With this function you can get the list of dangerous permissions that have been granted to the Android application.
	 * @param activity the caller activity for this method.
	 * @return granted permissions list
	 */
	public static String[] getGrantedPermissions(Activity activity) {
		String[] manifestPermissions;
		try {
			manifestPermissions = getManifestPermissions(activity);
		} catch (PackageManager.NameNotFoundException e) {
			e.printStackTrace();
			return new String[0];
		}
		if (manifestPermissions.length == 0)
			return manifestPermissions;

		List<String> grantedPermissions = new ArrayList<>();
		for (String manifestPermission : manifestPermissions) {
			try {
				if (manifestPermission.equals(Manifest.permission.MANAGE_EXTERNAL_STORAGE)) {
					if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && Environment.isExternalStorageManager()) {
						grantedPermissions.add(manifestPermission);
					}
				} else {
					PermissionInfo permissionInfo = getPermissionInfo(activity, manifestPermission);
					int protectionLevel = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ? permissionInfo.getProtection() : permissionInfo.protectionLevel;
					if (protectionLevel == PermissionInfo.PROTECTION_DANGEROUS && ContextCompat.checkSelfPermission(activity, manifestPermission) == PackageManager.PERMISSION_GRANTED) {
						grantedPermissions.add(manifestPermission);
					}
				}
			} catch (PackageManager.NameNotFoundException e) {
				// Skip this permission and continue.
				Log.w(TAG, "Unable to identify permission " + manifestPermission, e);
			}
		}

		return grantedPermissions.toArray(new String[0]);
	}

	/**
	 * Check if the given permission is in the AndroidManifest.xml file.
	 * @param activity the caller activity for this method.
	 * @param permission the permession to look for in the manifest file.
	 * @return "true" if the permission is in the manifest file of the activity, "false" otherwise.
	 */
	public static boolean hasManifestPermission(Activity activity, String permission) {
		try {
			for (String p : getManifestPermissions(activity)) {
				if (permission.equals(p))
					return true;
			}
		} catch (PackageManager.NameNotFoundException ignored) {
		}

		return false;
	}

	/**
	 * Returns the permissions defined in the AndroidManifest.xml file.
	 * @param activity the caller activity for this method.
	 * @return manifest permissions list
	 * @throws PackageManager.NameNotFoundException the exception is thrown when a given package, application, or component name cannot be found.
	 */
	private static String[] getManifestPermissions(Activity activity) throws PackageManager.NameNotFoundException {
		PackageManager packageManager = activity.getPackageManager();
		PackageInfo packageInfo = packageManager.getPackageInfo(activity.getPackageName(), PackageManager.GET_PERMISSIONS);
		if (packageInfo.requestedPermissions == null)
			return new String[0];
		return packageInfo.requestedPermissions;
	}

	/**
	 * Returns the information of the desired permission.
	 * @param activity the caller activity for this method.
	 * @param permission the name of the permission.
	 * @return permission info object
	 * @throws PackageManager.NameNotFoundException the exception is thrown when a given package, application, or component name cannot be found.
	 */
	private static PermissionInfo getPermissionInfo(Activity activity, String permission) throws PackageManager.NameNotFoundException {
		PackageManager packageManager = activity.getPackageManager();
		return packageManager.getPermissionInfo(permission, 0);
	}
}
