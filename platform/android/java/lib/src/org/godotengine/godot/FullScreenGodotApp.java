/*************************************************************************/
/*  FullScreenGodotApp.java                                              */
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

package org.godotengine.godot;

import android.content.Intent;
import android.os.Bundle;
import android.view.KeyEvent;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.FragmentActivity;

/**
 * Base activity for Android apps intending to use Godot as the primary and only screen.
 *
 * It's also a reference implementation for how to setup and use the {@link Godot} fragment
 * within an Android app.
 */
public abstract class FullScreenGodotApp extends FragmentActivity {
	@Nullable
	private Godot godotFragment;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.godot_app_layout);
		godotFragment = initGodotInstance();
		if (godotFragment == null) {
			throw new IllegalStateException("Godot instance must be non-null.");
		}

		getSupportFragmentManager().beginTransaction().replace(R.id.godot_fragment_container, godotFragment).setPrimaryNavigationFragment(godotFragment).commitNowAllowingStateLoss();
	}

	@Override
	public void onNewIntent(Intent intent) {
		if (godotFragment != null) {
			godotFragment.onNewIntent(intent);
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

	@Override
	public boolean onKeyMultiple(final int inKeyCode, int repeatCount, KeyEvent event) {
		if (godotFragment != null && godotFragment.onKeyMultiple(inKeyCode, repeatCount, event)) {
			return true;
		}
		return super.onKeyMultiple(inKeyCode, repeatCount, event);
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
