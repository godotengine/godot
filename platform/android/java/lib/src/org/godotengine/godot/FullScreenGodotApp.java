/*************************************************************************/
/*  FullScreenGodotApp.java                                              */
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

package org.godotengine.godot;

import android.content.ComponentName;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.CallSuper;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

/**
 * Base activity for Android apps intending to use Godot as the primary and only screen.
 *
 * It's also a reference implementation for how to setup and use the {@link Godot} fragment
 * within an Android app.
 */
public abstract class FullScreenGodotApp extends FragmentActivity implements GodotHost {
	private static final String TAG = FullScreenGodotApp.class.getSimpleName();

	@Nullable
	private Godot godotFragment;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.godot_app_layout);

		Fragment currentFragment = getSupportFragmentManager().findFragmentById(R.id.godot_fragment_container);
		if (currentFragment instanceof Godot) {
			Log.v(TAG, "Reusing existing Godot fragment instance.");
			godotFragment = (Godot)currentFragment;
		} else {
			Log.v(TAG, "Creating new Godot fragment instance.");
			godotFragment = initGodotInstance();
			getSupportFragmentManager().beginTransaction().replace(R.id.godot_fragment_container, godotFragment).setPrimaryNavigationFragment(godotFragment).commitNowAllowingStateLoss();
		}
	}

	@Override
	public void onDestroy() {
		super.onDestroy();
		onGodotForceQuit(godotFragment);
	}

	@Override
	public final void onGodotForceQuit(Godot instance) {
		if (instance == godotFragment) {
			System.exit(0);
		}
	}

	@Override
	public final void onGodotRestartRequested(Godot instance) {
		if (instance == godotFragment) {
			// HACK:
			//
			// Currently it's very hard to properly deinitialize Godot on Android to restart the game
			// from scratch. Therefore, we need to kill the whole app process and relaunch it.
			//
			// Restarting only the activity, wouldn't be enough unless it did proper cleanup (including
			// releasing and reloading native libs or resetting their state somehow and clearing statics).
			//
			// Using instrumentation is a way of making the whole app process restart, because Android
			// will kill any process of the same package which was already running.
			//
			Bundle args = new Bundle();
			args.putParcelable("intent", getIntent());
			startInstrumentation(new ComponentName(this, GodotInstrumentation.class), null, args);
		}
	}

	@Override
	public void onNewIntent(Intent intent) {
		super.onNewIntent(intent);
		if (godotFragment != null) {
			godotFragment.onNewIntent(intent);
		}
	}

	@CallSuper
	@Override
	public void onActivityResult(int requestCode, int resultCode, Intent data) {
		super.onActivityResult(requestCode, resultCode, data);
		if (godotFragment != null) {
			godotFragment.onActivityResult(requestCode, resultCode, data);
		}
	}

	@CallSuper
	@Override
	public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults);
		if (godotFragment != null) {
			godotFragment.onRequestPermissionsResult(requestCode, permissions, grantResults);
		}
	}

	@Override
	public void onBackPressed() {
		if (godotFragment != null) {
			godotFragment.onBackPressed();
		} else {
			super.onBackPressed();
		}
	}

	/**
	 * Used to initialize the Godot fragment instance in {@link FullScreenGodotApp#onCreate(Bundle)}.
	 */
	@NonNull
	protected Godot initGodotInstance() {
		return new Godot();
	}

	@Nullable
	protected final Godot getGodotFragment() {
		return godotFragment;
	}
}
