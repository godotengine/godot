/**************************************************************************/
/*  Renderer.java                                                         */
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

package org.godotengine.godot.render;

import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.view.Surface;

import androidx.annotation.NonNull;

/**
 * Godot renderer interface.
 * <p>
 * The Godot renderer is responsible for driving the render thread and making calls to render a frame.
 */
public interface Renderer {
	enum RenderMode {
		/**
		 * The renderer only renders
		 * when the surface is created, or when {@link #requestRender} is called.
		 *
		 * @see #getRenderMode()
		 * @see #setRenderMode(RenderMode)
		 * @see #requestRender()
		 */
		WHEN_DIRTY,

		/**
		 * The renderer is called
		 * continuously to re-render the scene.
		 *
		 * @see #getRenderMode()
		 * @see #setRenderMode(RenderMode)
		 */
		CONTINUOUSLY
	}

	/**
	 * Get the current rendering mode.
	 *
	 * @return the current rendering mode.
	 * @see RenderMode#CONTINUOUSLY
	 * @see RenderMode#WHEN_DIRTY
	 */
	RenderMode getRenderMode();

	/**
	 * Set the rendering mode. When renderMode is
	 * RenderMode#CONTINUOUSLY, the renderer is called
	 * repeatedly to re-render the scene. When renderMode
	 * is RenderMode#WHEN_DIRTY, the renderer only rendered when the surface
	 * is created, or when {@link #requestRender} is called. Defaults to RenderMode#CONTINUOUSLY.
	 * <p>
	 * Using RenderMode#WHEN_DIRTY can improve battery life and overall system performance
	 * by allowing the GPU and CPU to idle when the view does not need to be updated.
	 *
	 * @param renderMode one of the RENDERMODE_X constants
	 * @see RenderMode#CONTINUOUSLY
	 * @see RenderMode#WHEN_DIRTY
	 */
	void setRenderMode(@NonNull RenderMode renderMode);

	/**
	 * Request that the renderer render a frame.
	 * This method is typically used when the render mode has been set to
	 * {@link RenderMode#WHEN_DIRTY}, so that frames are only rendered on demand.
	 */
	void requestRender();

	/**
	 * Called when the surface is created or recreated.
	 */
	void onRenderSurfaceCreated(Surface surface);

	/**
	 * Called when the surface changed size.
	 * <p>
	 * Called after the surface is created and whenever the surface size changes.
	 */
	void onRenderSurfaceChanged(Surface surface, int width, int height);

	// -- GODOT start --

	/**
	 * Called to draw the current frame.
	 * <p>
	 * This method is responsible for drawing the current frame.
	 * <p>
	 */
	void onRenderDrawFrame();

	/**
	 * Invoked when the render thread is in the process of starting.
	 */
	void onRenderThreadStarting();

	/**
	 * Invoked when the render thread is in the process of shutting down.
	 */
	void onRenderThreadExiting();

	/**
	 * Invoked when any of the EGL resources may have changed.
	 */
	void onRenderEglResourcesChanged(EGLDisplay display, EGLSurface surface, EGLContext context, EGLConfig config);
}
