/*************************************************************************/
/*  ARCoreSetupActivity.java                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

package org.godotengine.godot.xr.arcore;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import org.godotengine.godot.Godot;
import org.godotengine.godot.R;

/**
 * Activity used to setup ARCore prior to initializing the Godot engine.
 */
public class ARCoreSetupActivity extends Activity {

	static final String PREVIOUS_ACTIVITY_START_INTENT_KEY = "previous_activity_start_intent";

	private static boolean requestARCoreInstall = true;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(new View(this));
	}

	@Override
	public void onResume() {
		super.onResume();

		// Check that ARCore is installed and up-to-date.
		try {
			switch (ArCoreApk.getInstance().requestInstall(this, requestARCoreInstall)) {
				case INSTALLED:
					break;
				case INSTALL_REQUESTED:
					requestARCoreInstall = false;
					return;
			}

			// Request CAMERA permission if needed.
			if (!ARCoreUtil.hasCameraPermission(this)) {
				ARCoreUtil.requestCameraPermission(this);
				return;
			}

			// We have everything we need. Let's restart the original activity.
			Intent previousIntent = null;
			if (getIntent() != null) {
				previousIntent = getIntent().getParcelableExtra(PREVIOUS_ACTIVITY_START_INTENT_KEY);
			}
			if (previousIntent == null) {
				previousIntent = new Intent(this, Godot.class);
			}

			startActivity(previousIntent);

		} catch (UnavailableUserDeclinedInstallationException | UnavailableDeviceNotCompatibleException e) {
			Toast.makeText(this, R.string.setup_arcore_request, Toast.LENGTH_LONG).show();
		}

		finish();
	}

	@Override
	public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
		if (ARCoreUtil.isARCoreRequestPermissionCode(requestCode)) {
			if (!ARCoreUtil.hasCameraPermission(this)) {
				Toast.makeText(this, R.string.missing_camera_permission_warning, Toast.LENGTH_LONG).show();
				if (!ARCoreUtil.shouldShowRequestPermissionRationale(this)) {
					// Permission denied with checking "Do not ask again".
					ARCoreUtil.launchPermissionSettings(this);
				}
				finish();
			}
		}
	}
}
