/**************************************************************************/
/*  GodotGLRenderView.java                                                */
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

import org.godotengine.godot.gl.GLSurfaceView;
import org.godotengine.godot.gl.GodotRenderer;
import org.godotengine.godot.input.GodotInputHandler;
import org.godotengine.godot.xr.XRMode;
import org.godotengine.godot.xr.ovr.OvrConfigChooser;
import org.godotengine.godot.xr.ovr.OvrContextFactory;
import org.godotengine.godot.xr.ovr.OvrWindowSurfaceFactory;
import org.godotengine.godot.xr.regular.RegularConfigChooser;
import org.godotengine.godot.xr.regular.RegularContextFactory;
import org.godotengine.godot.xr.regular.RegularFallbackConfigChooser;

import android.annotation.SuppressLint;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.os.Build;
import android.text.TextUtils;
import android.util.SparseArray;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.PointerIcon;
import android.view.SurfaceView;

import androidx.annotation.Keep;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;

/**
 * A simple GLSurfaceView sub-class that demonstrate how to perform
 * OpenGL ES 2.0 rendering into a GL Surface. Note the following important
 * details:
 *
 * - The class must use a custom context factory to enable 2.0 rendering.
 *   See ContextFactory class definition below.
 *
 * - The class must use a custom EGLConfigChooser to be able to select
 *   an EGLConfig that supports 3.0. This is done by providing a config
 *   specification to eglChooseConfig() that has the attribute
 *   EGL10.ELG_RENDERABLE_TYPE containing the EGL_OPENGL_ES2_BIT flag
 *   set. See ConfigChooser class definition below.
 *
 * - The class must select the surface's format, then choose an EGLConfig
 *   that matches it exactly (with regards to red/green/blue/alpha channels
 *   bit depths). Failure to do so would result in an EGL_BAD_MATCH error.
 */
class GodotGLRenderView extends GLSurfaceView implements GodotRenderView {
	private final Godot godot;
	private final GodotInputHandler inputHandler;
	private final GodotRenderer godotRenderer;
	private final SparseArray<PointerIcon> customPointerIcons = new SparseArray<>();

	public GodotGLRenderView(Godot godot, GodotInputHandler inputHandler, XRMode xrMode, boolean useDebugOpengl, boolean shouldBeTranslucent) {
		super(godot.getContext());

		this.godot = godot;
		this.inputHandler = inputHandler;
		this.godotRenderer = new GodotRenderer();
		setPointerIcon(PointerIcon.getSystemIcon(getContext(), PointerIcon.TYPE_DEFAULT));
		init(xrMode, shouldBeTranslucent, useDebugOpengl);
	}

	@Override
	public SurfaceView getView() {
		return this;
	}

	@Override
	public void queueOnRenderThread(Runnable event) {
		queueEvent(event);
	}

	@Override
	public void onActivityPaused() {
		queueEvent(() -> {
			GodotLib.focusout();
			// Pause the renderer
			godotRenderer.onActivityPaused();
		});
	}

	@Override
	public void onActivityStopped() {
		pauseGLThread();
	}

	@Override
	public void onActivityResumed() {
		queueEvent(() -> {
			// Resume the renderer
			godotRenderer.onActivityResumed();
			GodotLib.focusin();
		});
	}

	@Override
	public void onActivityStarted() {
		resumeGLThread();
	}

	@Override
	public void onActivityDestroyed() {
		requestRenderThreadExitAndWait();
	}

	@Override
	public GodotInputHandler getInputHandler() {
		return inputHandler;
	}

	@SuppressLint("ClickableViewAccessibility")
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		super.onTouchEvent(event);
		return inputHandler.onTouchEvent(event);
	}

	@Override
	public boolean onKeyUp(final int keyCode, KeyEvent event) {
		return inputHandler.onKeyUp(keyCode, event) || super.onKeyUp(keyCode, event);
	}

	@Override
	public boolean onKeyDown(final int keyCode, KeyEvent event) {
		return inputHandler.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		return inputHandler.onGenericMotionEvent(event) || super.onGenericMotionEvent(event);
	}

	@Override
	public boolean onCapturedPointerEvent(MotionEvent event) {
		return inputHandler.onGenericMotionEvent(event);
	}

	@Override
	public void onPointerCaptureChange(boolean hasCapture) {
		super.onPointerCaptureChange(hasCapture);
		inputHandler.onPointerCaptureChange(hasCapture);
	}

	@Override
	public void requestPointerCapture() {
		if (canCapturePointer()) {
			super.requestPointerCapture();
			inputHandler.onPointerCaptureChange(true);
		}
	}

	@Override
	public void releasePointerCapture() {
		super.releasePointerCapture();
		inputHandler.onPointerCaptureChange(false);
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

	private void init(XRMode xrMode, boolean translucent, boolean useDebugOpengl) {
		boolean shouldPreserveContext = !isProblematicAdrenoGpu();
		setPreserveEGLContextOnPause(shouldPreserveContext);
		setFocusableInTouchMode(true);
		switch (xrMode) {
			case OPENXR:
				// Replace the default egl config chooser.
				setEGLConfigChooser(new OvrConfigChooser());

				// Replace the default context factory.
				setEGLContextFactory(new OvrContextFactory());

				// Replace the default window surface factory.
				setEGLWindowSurfaceFactory(new OvrWindowSurfaceFactory());
				break;

			case REGULAR:
			default:
				/* By default, GLSurfaceView() creates a RGB_565 opaque surface.
				 * If we want a translucent one, we should change the surface's
				 * format here, using PixelFormat.TRANSLUCENT for GL Surfaces
				 * is interpreted as any 32-bit surface with alpha by SurfaceFlinger.
				 */
				if (translucent) {
					this.getHolder().setFormat(PixelFormat.TRANSLUCENT);
				}

				/* Setup the context factory for 2.0 rendering.
				 * See ContextFactory class definition below
				 */
				setEGLContextFactory(new RegularContextFactory(useDebugOpengl));

				/* We need to choose an EGLConfig that matches the format of
				 * our surface exactly. This is going to be done in our
				 * custom config chooser. See ConfigChooser class definition
				 * below.
				 */

				setEGLConfigChooser(
						new RegularFallbackConfigChooser(8, 8, 8, 8, 24, 0,
								new RegularConfigChooser(8, 8, 8, 8, 16, 0)));
				break;
		}
	}

	@Override
	public void startRenderer() {
		/* Set the renderer responsible for frame rendering */
		setRenderer(godotRenderer);
	}

	/**
	 * Detects problematic Adreno GPU configurations that have shader corruption issues
	 * when EGL context is preserved across surface destroy/create cycles.
	 *
	 * @return true if this is a problematic Adreno GPU configuration
	 */
	private boolean isProblematicAdrenoGpu() {
		try {
			String hardware = Build.HARDWARE.toLowerCase();
			String board = Build.BOARD.toLowerCase();
			String device = Build.DEVICE.toLowerCase();

			// Known problematic SoCs with Adreno 5XX GPUs:
			// - msm8953: Snapdragon 625 (Adreno 506)
			// - msm8937/msm8940: Snapdragon 430/435 (Adreno 505)
			// - msm8917: Snapdragon 425 (Adreno 505)
			// - msm8976: Snapdragon 652/653 (Adreno 510)
			// - msm8956: Snapdragon 650 (Adreno 510)
			// - msm8996: Snapdragon 820/821 (Adreno 530)

			boolean isProblematicSoc = hardware.contains("qcom") && (
				hardware.contains("msm8953") ||  // Snapdragon 625 (Adreno 506) - PRIMARY TARGET
				hardware.contains("msm8937") ||  // Snapdragon 430 (Adreno 505)
				hardware.contains("msm8940") ||  // Snapdragon 435 (Adreno 505)
				hardware.contains("msm8917") ||  // Snapdragon 425 (Adreno 505)
				hardware.contains("msm8976") ||  // Snapdragon 652/653 (Adreno 510)
				hardware.contains("msm8956") ||  // Snapdragon 650 (Adreno 510)
				board.contains("msm8953") ||
				board.contains("msm8937") ||
				board.contains("msm8940") ||
				board.contains("msm8917") ||
				board.contains("msm8976") ||
				board.contains("msm8956")
			);

			// Also check for Snapdragon 6XX series on Android 9 and below
			// as these often have Adreno 5XX GPUs with similar issues
			if (!isProblematicSoc && Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
				try (BufferedReader reader = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
					String line;
					while ((line = reader.readLine()) != null) {
						String lowerLine = line.toLowerCase();
						if (lowerLine.contains("hardware") && lowerLine.contains("qualcomm")) {
							if (lowerLine.contains("msm8953") || lowerLine.contains("msm8937") ||
								lowerLine.contains("msm8940") || lowerLine.contains("msm8917") ||
								lowerLine.contains("msm8976") || lowerLine.contains("msm8956")) {
								isProblematicSoc = true;
								break;
							}
						}
					}
				} catch (Exception e) {
					// Ignore errors
				}
			}

			if (isProblematicSoc) {
				return true;
			}

			return false;
		} catch (Exception e) {
			return Build.VERSION.SDK_INT <= Build.VERSION_CODES.P;
		}
	}
}
