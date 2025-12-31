/**************************************************************************/
/*  GodotVulkanRenderView.java                                            */
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

import org.godotengine.godot.input.GodotInputHandler;
import org.godotengine.godot.vulkan.VkRenderer;
import org.godotengine.godot.vulkan.VkSurfaceView;

import android.annotation.SuppressLint;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.text.TextUtils;
import android.util.SparseArray;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.PointerIcon;
import android.view.SurfaceView;

import androidx.annotation.Keep;

import java.io.InputStream;

class GodotVulkanRenderView extends VkSurfaceView implements GodotRenderView {
	private final Godot godot;
	private final GodotInputHandler mInputHandler;
	private final VkRenderer mRenderer;
	private final SparseArray<PointerIcon> customPointerIcons = new SparseArray<>();

	public GodotVulkanRenderView(Godot godot, GodotInputHandler inputHandler, boolean shouldBeTranslucent) {
		super(godot.getContext());

		this.godot = godot;
		mInputHandler = inputHandler;
		mRenderer = new VkRenderer();
		setPointerIcon(PointerIcon.getSystemIcon(getContext(), PointerIcon.TYPE_DEFAULT));
		setFocusableInTouchMode(true);
		setClickable(false);

		if (shouldBeTranslucent) {
			this.getHolder().setFormat(PixelFormat.TRANSLUCENT);
		}
	}

	@Override
	public void startRenderer() {
		startRenderer(mRenderer);
	}

	@Override
	public SurfaceView getView() {
		return this;
	}

	@Override
	public void queueOnRenderThread(Runnable event) {
		queueOnVkThread(event);
	}

	@Override
	public void onActivityPaused() {
		queueOnVkThread(() -> {
			GodotLib.focusout();
			// Pause the renderer
			mRenderer.onVkPause();
		});
	}

	@Override
	public void onActivityStopped() {
		pauseRenderThread();
	}

	@Override
	public void onActivityStarted() {
		resumeRenderThread();
	}

	@Override
	public void onActivityResumed() {
		queueOnVkThread(() -> {
			// Resume the renderer
			mRenderer.onVkResume();
			GodotLib.focusin();
		});
	}

	@Override
	public void onActivityDestroyed() {
		requestRenderThreadExitAndWait();
	}

	@Override
	public GodotInputHandler getInputHandler() {
		return mInputHandler;
	}

	@SuppressLint("ClickableViewAccessibility")
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		super.onTouchEvent(event);
		return mInputHandler.onTouchEvent(event);
	}

	@Override
	public boolean onKeyUp(final int keyCode, KeyEvent event) {
		return mInputHandler.onKeyUp(keyCode, event) || super.onKeyUp(keyCode, event);
	}

	@Override
	public boolean onKeyDown(final int keyCode, KeyEvent event) {
		return mInputHandler.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		return mInputHandler.onGenericMotionEvent(event) || super.onGenericMotionEvent(event);
	}

	@Override
	public boolean onCapturedPointerEvent(MotionEvent event) {
		return mInputHandler.onGenericMotionEvent(event);
	}

	@Override
	public boolean canCapturePointer() {
		// Pointer capture is not supported on XR devices.
		return !godot.isXrRuntime() && mInputHandler.canCapturePointer();
	}
	@Override
	public void requestPointerCapture() {
		if (canCapturePointer()) {
			super.requestPointerCapture();
			mInputHandler.onPointerCaptureChange(true);
		}
	}

	@Override
	public void releasePointerCapture() {
		super.releasePointerCapture();
		mInputHandler.onPointerCaptureChange(false);
	}

	@Override
	public void onPointerCaptureChange(boolean hasCapture) {
		super.onPointerCaptureChange(hasCapture);
		mInputHandler.onPointerCaptureChange(hasCapture);
	}

	/**
	 * Used to configure the PointerIcon for the given type.
	 *
	 * Called from JNI
	 */
	@Keep
	@Override
	public void configurePointerIcon(int pointerType, String imagePath, float hotSpotX, float hotSpotY) {
		try {
			Bitmap bitmap = null;
			if (!TextUtils.isEmpty(imagePath)) {
				if (godot.getDirectoryAccessHandler().filesystemFileExists(imagePath)) {
					// Try to load the bitmap from the file system
					bitmap = BitmapFactory.decodeFile(imagePath);
				} else if (godot.getDirectoryAccessHandler().assetsFileExists(imagePath)) {
					// Try to load the bitmap from the assets directory
					AssetManager am = getContext().getAssets();
					InputStream imageInputStream = am.open(imagePath);
					bitmap = BitmapFactory.decodeStream(imageInputStream);
				}
			}

			PointerIcon customPointerIcon = PointerIcon.create(bitmap, hotSpotX, hotSpotY);
			customPointerIcons.put(pointerType, customPointerIcon);
		} catch (Exception e) {
			// Reset the custom pointer icon
			customPointerIcons.delete(pointerType);
		}
	}

	/**
	 * called from JNI to change pointer icon
	 */
	@Keep
	@Override
	public void setPointerIcon(int pointerType) {
		PointerIcon pointerIcon = customPointerIcons.get(pointerType);
		if (pointerIcon == null) {
			pointerIcon = PointerIcon.getSystemIcon(getContext(), pointerType);
		}
		setPointerIcon(pointerIcon);
	}

	@Override
	public PointerIcon onResolvePointerIcon(MotionEvent me, int pointerIndex) {
		return getPointerIcon();
	}
}
