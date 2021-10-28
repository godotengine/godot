/*************************************************************************/
/*  GodotGLRenderView.java                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
import org.godotengine.godot.utils.GLUtils;
import org.godotengine.godot.xr.XRMode;
import org.godotengine.godot.xr.ovr.OvrConfigChooser;
import org.godotengine.godot.xr.ovr.OvrContextFactory;
import org.godotengine.godot.xr.ovr.OvrWindowSurfaceFactory;
import org.godotengine.godot.xr.regular.RegularConfigChooser;
import org.godotengine.godot.xr.regular.RegularContextFactory;
import org.godotengine.godot.xr.regular.RegularFallbackConfigChooser;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.view.GestureDetector;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.PointerIcon;
import android.view.SurfaceView;

import androidx.annotation.Keep;

/**
 * A simple GLSurfaceView sub-class that demonstrate how to perform
 * OpenGL ES 2.0 rendering into a GL Surface. Note the following important
 * details:
 *
 * - The class must use a custom context factory to enable 2.0 rendering.
 *   See ContextFactory class definition below.
 *
 * - The class must use a custom EGLConfigChooser to be able to select
 *   an EGLConfig that supports 2.0. This is done by providing a config
 *   specification to eglChooseConfig() that has the attribute
 *   EGL10.ELG_RENDERABLE_TYPE containing the EGL_OPENGL_ES2_BIT flag
 *   set. See ConfigChooser class definition below.
 *
 * - The class must select the surface's format, then choose an EGLConfig
 *   that matches it exactly (with regards to red/green/blue/alpha channels
 *   bit depths). Failure to do so would result in an EGL_BAD_MATCH error.
 */
public class GodotGLRenderView extends GLSurfaceView implements GodotRenderView {
	private final Godot godot;
	private final GodotInputHandler inputHandler;
	private final GestureDetector detector;
	private final GodotRenderer godotRenderer;
	private PointerIcon pointerIcon;

	public GodotGLRenderView(Context context, Godot godot, XRMode xrMode, boolean p_use_32_bits,
			boolean p_use_debug_opengl) {
		super(context);
		GLUtils.use_32 = p_use_32_bits;
		GLUtils.use_debug_opengl = p_use_debug_opengl;

		this.godot = godot;
		this.inputHandler = new GodotInputHandler(this);
		this.detector = new GestureDetector(context, new GodotGestureHandler(this));
		this.godotRenderer = new GodotRenderer();
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			pointerIcon = PointerIcon.getSystemIcon(getContext(), PointerIcon.TYPE_DEFAULT);
		}
		init(xrMode, false, 16, 0);
	}

	@Override
	public SurfaceView getView() {
		return this;
	}

	@Override
	public void initInputDevices() {
		this.inputHandler.initInputDevices();
	}

	@Override
	public void queueOnRenderThread(Runnable event) {
		queueEvent(event);
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
		return inputHandler;
	}

	@SuppressLint("ClickableViewAccessibility")
	@Override
	public boolean onTouchEvent(MotionEvent event) {
		super.onTouchEvent(event);
		this.detector.onTouchEvent(event);
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

	private void init(XRMode xrMode, boolean translucent, int depth, int stencil) {
		setPreserveEGLContextOnPause(true);
		setFocusableInTouchMode(true);
		switch (xrMode) {
			case OVR:
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
				setEGLContextFactory(new RegularContextFactory());

				/* We need to choose an EGLConfig that matches the format of
				 * our surface exactly. This is going to be done in our
				 * custom config chooser. See ConfigChooser class definition
				 * below.
				 */

				if (GLUtils.use_32) {
					setEGLConfigChooser(translucent ?
									  new RegularFallbackConfigChooser(8, 8, 8, 8, 24, stencil,
											new RegularConfigChooser(8, 8, 8, 8, 16, stencil)) :
									  new RegularFallbackConfigChooser(8, 8, 8, 8, 24, stencil,
											new RegularConfigChooser(5, 6, 5, 0, 16, stencil)));

				} else {
					setEGLConfigChooser(translucent ?
									  new RegularConfigChooser(8, 8, 8, 8, 16, stencil) :
									  new RegularConfigChooser(5, 6, 5, 0, 16, stencil));
				}
				break;
		}

		/* Set the renderer responsible for frame rendering */
		setRenderer(godotRenderer);
	}

	@Override
	public void onResume() {
		super.onResume();

		queueEvent(() -> {
			// Resume the renderer
			godotRenderer.onActivityResumed();
			GodotLib.focusin();
		});
	}

	@Override
	public void onPause() {
		super.onPause();

		queueEvent(() -> {
			GodotLib.focusout();
			// Pause the renderer
			godotRenderer.onActivityPaused();
		});
	}
}
