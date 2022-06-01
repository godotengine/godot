/*************************************************************************/
/*  GodotEditor.java                                                     */
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

package org.godotengine.editor;

import org.godotengine.godot.FullScreenGodotApp;
import org.godotengine.godot.utils.PermissionsUtil;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;

import androidx.annotation.Nullable;
import androidx.window.layout.WindowMetrics;
import androidx.window.layout.WindowMetricsCalculator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Base class for the Godot Android Editor activities.
 *
 * This provides the basic templates for the activities making up this application.
 * Each derived activity runs in its own process, which enable up to have several instances of
 * the Godot engine up and running at the same time.
 *
 * It also plays the role of the primary editor window.
 */
public class GodotEditor extends FullScreenGodotApp {
	private static final boolean WAIT_FOR_DEBUGGER = false;
	private static final String COMMAND_LINE_PARAMS = "command_line_params";

	private static final String EDITOR_ARG = "--editor";
	private static final String PROJECT_MANAGER_ARG = "--project-manager";

	private final List<String> commandLineParams = new ArrayList<>();

	@Override
	public void onCreate(Bundle savedInstanceState) {
		PermissionsUtil.requestManifestPermissions(this);

		String[] params = getIntent().getStringArrayExtra(COMMAND_LINE_PARAMS);
		updateCommandLineParams(params);

		if (BuildConfig.BUILD_TYPE.equals("debug") && WAIT_FOR_DEBUGGER) {
			Debug.waitForDebugger();
		}
		super.onCreate(savedInstanceState);
	}

	private void updateCommandLineParams(@Nullable String[] args) {
		// Update the list of command line params with the new args
		commandLineParams.clear();
		if (args != null && args.length > 0) {
			commandLineParams.addAll(Arrays.asList(args));
		}
	}

	@Override
	public List<String> getCommandLine() {
		return commandLineParams;
	}

	@Override
	public void onNewGodotInstanceRequested(String[] args) {
		// Parse the arguments to figure out which activity to start.
		Class<?> targetClass = GodotGame.class;
		// Whether we should launch the new godot instance in an adjacent window
		// https://developer.android.com/reference/android/content/Intent#FLAG_ACTIVITY_LAUNCH_ADJACENT
		boolean launchAdjacent = Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && (isInMultiWindowMode() || isLargeScreen());

		for (String arg : args) {
			if (EDITOR_ARG.equals(arg)) {
				targetClass = GodotEditor.class;
				launchAdjacent = false;
				break;
			}

			if (PROJECT_MANAGER_ARG.equals(arg)) {
				targetClass = GodotProjectManager.class;
				launchAdjacent = false;
				break;
			}
		}

		// Launch a new activity
		Intent newInstance = new Intent(this, targetClass)
									 .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
									 .putExtra(COMMAND_LINE_PARAMS, args);
		if (launchAdjacent) {
			newInstance.addFlags(Intent.FLAG_ACTIVITY_LAUNCH_ADJACENT);
		}
		startActivity(newInstance);
	}

	protected boolean isLargeScreen() {
		WindowMetrics metrics =
				WindowMetricsCalculator.getOrCreate().computeMaximumWindowMetrics(this);

		// Get the screen's density scale
		float scale = getResources().getDisplayMetrics().density;

		// Get the minimum window size
		float minSize = Math.min(metrics.getBounds().width(), metrics.getBounds().height());
		float minSizeDp = minSize / scale;
		return minSizeDp >= 840f; // Correspond to the EXPANDED window size class.
	}

	@Override
	public void setRequestedOrientation(int requestedOrientation) {
		if (!overrideOrientationRequest()) {
			super.setRequestedOrientation(requestedOrientation);
		}
	}

	/**
	 * The Godot Android Editor sets its own orientation via its AndroidManifest
	 */
	protected boolean overrideOrientationRequest() {
		return true;
	}
}
