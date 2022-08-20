/*************************************************************************/
/*  GodotVulkanRenderView.java                                           */
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

import org.godotengine.godot.input.GodotGestureHandler;
import org.godotengine.godot.input.GodotInputHandler;
import org.godotengine.godot.vulkan.VkRenderer;
import org.godotengine.godot.vulkan.VkSurfaceView;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Build;
import android.view.GestureDetector;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.PointerIcon;
import android.view.SurfaceView;

import androidx.annotation.Keep;

public class GodotVulkanRenderView extends VkSurfaceView implements GodotRenderView {
	private final Godot godot;
	private final GodotInputHandler mInputHandler;
	private final GestureDetector mGestureDetector;
	private final VkRenderer mRenderer;
	private PointerIcon pointerIcon;

	public GodotVulkanRenderView(Context context, Godot godot) {
		super(context);

		this.godot = godot;
		mInputHandler = new GodotInputHandler(this);
		mGestureDetector = new GestureDetector(context, new GodotGestureHandler(this));
		mRenderer = new VkRenderer();
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			pointerIcon = PointerIcon.getSystemIcon(getContext(), PointerIcon.TYPE_DEFAULT);
		}
		setFocusableInTouchMode(true);
		startRenderer(mRenderer);
	}

	@Override
	public SurfaceView getView() {
		return this;
	}

	@Override
	public void initInputDevices() {
		mInputHandler.initInputDevices();
	}

	@Override
	public void queueOnRenderThread(Runnable event) {
		queueOnVkThread(event);
	}

	@Override
	public void onActivityPaused() {
		onPause();
	}

	@Override
	public void onActivityResumed() {
		onResume();
	}

	@Override
	public void onBackPressed() {
		godot.onBackPressed();
	}

	@Override
	public GodotInputHandler getInputHandler() {
		return mInputHandler;
	}

	@SuppressLint("ClickableViewAccessibility")
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		super.onTouchEvent(event);
		mGestureDetector.onTouchEvent(event);
		return mInputHandler.onTouchEvent(event);
	}

	@Override
	public boolean onKeyUp(final int keyCode, KeyEvent event) {
		return mInputHandler.onKeyUp(keyCode, event);
	}

	@Override
	public boolean onKeyDown(final int keyCode, KeyEvent event) {
		return mInputHandler.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		return mInputHandler.onGenericMotionEvent(event);
	}

	@Override
	public boolean onCapturedPointerEvent(MotionEvent event) {
		return mInputHandler.onGenericMotionEvent(event);
	}

	/**
	 * called from JNI to change pointer icon
	 */
	@Keep
	public void setPointerIcon(int pointerType) {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			pointerIcon = PointerIcon.getSystemIcon(getContext(), pointerType);
		}
	}

	@Override
	public PointerIcon onResolvePointerIcon(MotionEvent me, int pointerIndex) {
		return pointerIcon;
	}

	@Override
	public void onResume() {
		super.onResume();

		queueOnVkThread(() -> {
			// Resume the renderer
			mRenderer.onVkResume();
			GodotLib.focusin();
		});
	}

	@Override
	public void onPause() {
		super.onPause();

		queueOnVkThread(() -> {
			GodotLib.focusout();
			// Pause the renderer
			mRenderer.onVkPause();
		});
	}
}
