// clang-format off

/*
 * Copyright (C) 2008 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.godotengine.godot.gl;

import android.content.Context;
import android.opengl.EGL14;
import android.opengl.EGLExt;
import android.opengl.GLDebugHelper;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.Writer;
import java.lang.ref.WeakReference;
import java.util.ArrayList;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGL11;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;
import javax.microedition.khronos.opengles.GL;
import javax.microedition.khronos.opengles.GL10;

/**
 * An implementation of SurfaceView that uses the dedicated surface for
 * displaying OpenGL rendering.
 * <p>
 * A GLSurfaceView provides the following features:
 * <p>
 * <ul>
 * <li>Manages a surface, which is a special piece of memory that can be
 * composited into the Android view system.
 * <li>Manages an EGL display, which enables OpenGL to render into a surface.
 * <li>Accepts a user-provided Renderer object that does the actual rendering.
 * <li>Renders on a dedicated thread to decouple rendering performance from the
 * UI thread.
 * <li>Supports both on-demand and continuous rendering.
 * <li>Optionally wraps, traces, and/or error-checks the renderer's OpenGL calls.
 * </ul>
 *
 * <div class="special reference">
 * <h3>Developer Guides</h3>
 * <p>For more information about how to use OpenGL, read the
 * <a href="{@docRoot}guide/topics/graphics/opengl.html">OpenGL</a> developer guide.</p>
 * </div>
 *
 * <h3>Using GLSurfaceView</h3>
 * <p>
 * Typically you use GLSurfaceView by subclassing it and overriding one or more of the
 * View system input event methods. If your application does not need to override event
 * methods then GLSurfaceView can be used as-is. For the most part
 * GLSurfaceView behavior is customized by calling "set" methods rather than by subclassing.
 * For example, unlike a regular View, drawing is delegated to a separate Renderer object which
 * is registered with the GLSurfaceView
 * using the {@link #setRenderer(Renderer)} call.
 * <p>
 * <h3>Initializing GLSurfaceView</h3>
 * All you have to do to initialize a GLSurfaceView is call {@link #setRenderer(Renderer)}.
 * However, if desired, you can modify the default behavior of GLSurfaceView by calling one or
 * more of these methods before calling setRenderer:
 * <ul>
 * <li>{@link #setDebugFlags(int)}
 * <li>{@link #setEGLConfigChooser(boolean)}
 * <li>{@link #setEGLConfigChooser(EGLConfigChooser)}
 * <li>{@link #setEGLConfigChooser(int, int, int, int, int, int)}
 * <li>{@link #setGLWrapper(GLWrapper)}
 * </ul>
 * <p>
 * <h4>Specifying the android.view.Surface</h4>
 * By default GLSurfaceView will create a PixelFormat.RGB_888 format surface. If a translucent
 * surface is required, call getHolder().setFormat(PixelFormat.TRANSLUCENT).
 * The exact format of a TRANSLUCENT surface is device dependent, but it will be
 * a 32-bit-per-pixel surface with 8 bits per component.
 * <p>
 * <h4>Choosing an EGL Configuration</h4>
 * A given Android device may support multiple EGLConfig rendering configurations.
 * The available configurations may differ in how many channels of data are present, as
 * well as how many bits are allocated to each channel. Therefore, the first thing
 * GLSurfaceView has to do when starting to render is choose what EGLConfig to use.
 * <p>
 * By default GLSurfaceView chooses a EGLConfig that has an RGB_888 pixel format,
 * with at least a 16-bit depth buffer and no stencil.
 * <p>
 * If you would prefer a different EGLConfig
 * you can override the default behavior by calling one of the
 * setEGLConfigChooser methods.
 * <p>
 * <h4>Debug Behavior</h4>
 * You can optionally modify the behavior of GLSurfaceView by calling
 * one or more of the debugging methods {@link #setDebugFlags(int)},
 * and {@link #setGLWrapper}. These methods may be called before and/or after setRenderer, but
 * typically they are called before setRenderer so that they take effect immediately.
 * <p>
 * <h4>Setting a Renderer</h4>
 * Finally, you must call {@link #setRenderer} to register a {@link Renderer}.
 * The renderer is
 * responsible for doing the actual OpenGL rendering.
 * <p>
 * <h3>Rendering Mode</h3>
 * Once the renderer is set, you can control whether the renderer draws
 * continuously or on-demand by calling
 * {@link #setRenderMode}. The default is continuous rendering.
 * <p>
 * <h3>Activity Life-cycle</h3>
 * A GLSurfaceView must be notified when to pause and resume rendering. GLSurfaceView clients
 * are required to call {@link #pauseGLThread()} when the activity stops and
 * {@link #resumeGLThread()} when the activity starts. These calls allow GLSurfaceView to
 * pause and resume the rendering thread, and also allow GLSurfaceView to release and recreate
 * the OpenGL display.
 * <p>
 * <h3>Handling events</h3>
 * <p>
 * To handle an event you will typically subclass GLSurfaceView and override the
 * appropriate method, just as you would with any other View. However, when handling
 * the event, you may need to communicate with the Renderer object
 * that's running in the rendering thread. You can do this using any
 * standard Java cross-thread communication mechanism. In addition,
 * one relatively easy way to communicate with your renderer is
 * to call
 * {@link #queueEvent(Runnable)}. For example:
 * <pre class="prettyprint">
 * class MyGLSurfaceView extends GLSurfaceView {
 *
 *     private MyRenderer mMyRenderer;
 *
 *     public void start() {
 *         mMyRenderer = ...;
 *         setRenderer(mMyRenderer);
 *     }
 *
 *     public boolean onKeyDown(int keyCode, KeyEvent event) {
 *         if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER) {
 *             queueEvent(new Runnable() {
 *                 // This method will be called on the rendering
 *                 // thread:
 *                 public void run() {
 *                     mMyRenderer.handleDpadCenter();
 *                 }});
 *             return true;
 *         }
 *         return super.onKeyDown(keyCode, event);
 *     }
 * }
 * </pre>
 *
 */
public class GLSurfaceView extends SurfaceView implements SurfaceHolder.Callback2 {
	private final static String TAG = "GLSurfaceView";
	private final static boolean LOG_ATTACH_DETACH = false;
	private final static boolean LOG_THREADS = false;
	private final static boolean LOG_PAUSE_RESUME = false;
	private final static boolean LOG_SURFACE = false;
	private final static boolean LOG_RENDERER = false;
	private final static boolean LOG_RENDERER_DRAW_FRAME = false;
	private final static boolean LOG_EGL = false;
	/**
	 * The renderer only renders
	 * when the surface is created, or when {@link #requestRender} is called.
	 *
	 * @see #getRenderMode()
	 * @see #setRenderMode(int)
	 * @see #requestRender()
	 */
	public final static int RENDERMODE_WHEN_DIRTY = 0;
	/**
	 * The renderer is called
	 * continuously to re-render the scene.
	 *
	 * @see #getRenderMode()
	 * @see #setRenderMode(int)
	 */
	public final static int RENDERMODE_CONTINUOUSLY = 1;

	/**
	 * Check glError() after every GL call and throw an exception if glError indicates
	 * that an error has occurred. This can be used to help track down which OpenGL ES call
	 * is causing an error.
	 *
	 * @see #getDebugFlags
	 * @see #setDebugFlags
	 */
	public final static int DEBUG_CHECK_GL_ERROR = 1;

	/**
	 * Log GL calls to the system log at "verbose" level with tag "GLSurfaceView".
	 *
	 * @see #getDebugFlags
	 * @see #setDebugFlags
	 */
	public final static int DEBUG_LOG_GL_CALLS = 2;

	/**
	 * Standard View constructor. In order to render something, you
	 * must call {@link #setRenderer} to register a renderer.
	 */
	public GLSurfaceView(Context context) {
		super(context);
		init();
	}

	/**
	 * Standard View constructor. In order to render something, you
	 * must call {@link #setRenderer} to register a renderer.
	 */
	public GLSurfaceView(Context context, AttributeSet attrs) {
		super(context, attrs);
		init();
	}

	@Override
	protected void finalize() throws Throwable {
		try {
			if (mGLThread != null) {
				// GLThread may still be running if this view was never
				// attached to a window.
				mGLThread.requestExitAndWait();
			}
		} finally {
			super.finalize();
		}
	}

	private void init() {
		// Install a SurfaceHolder.Callback so we get notified when the
		// underlying surface is created and destroyed
		SurfaceHolder holder = getHolder();
		holder.addCallback(this);
		// setFormat is done by SurfaceView in SDK 2.3 and newer. Uncomment
		// this statement if back-porting to 2.2 or older:
		// holder.setFormat(PixelFormat.RGB_565);
		//
		// setType is not needed for SDK 2.0 or newer. Uncomment this
		// statement if back-porting this code to older SDKs.
		// holder.setType(SurfaceHolder.SURFACE_TYPE_GPU);
	}

	/**
	 * Set the glWrapper. If the glWrapper is not null, its
	 * {@link GLWrapper#wrap(GL)} method is called
	 * whenever a surface is created. A GLWrapper can be used to wrap
	 * the GL object that's passed to the renderer. Wrapping a GL
	 * object enables examining and modifying the behavior of the
	 * GL calls made by the renderer.
	 * <p>
	 * Wrapping is typically used for debugging purposes.
	 * <p>
	 * The default value is null.
	 * @param glWrapper the new GLWrapper
	 */
	public void setGLWrapper(GLWrapper glWrapper) {
		mGLWrapper = glWrapper;
	}

	/**
	 * Set the debug flags to a new value. The value is
	 * constructed by OR-together zero or more
	 * of the DEBUG_CHECK_* constants. The debug flags take effect
	 * whenever a surface is created. The default value is zero.
	 * @param debugFlags the new debug flags
	 * @see #DEBUG_CHECK_GL_ERROR
	 * @see #DEBUG_LOG_GL_CALLS
	 */
	public void setDebugFlags(int debugFlags) {
		mDebugFlags = debugFlags;
	}

	/**
	 * Get the current value of the debug flags.
	 * @return the current value of the debug flags.
	 */
	public int getDebugFlags() {
		return mDebugFlags;
	}

	/**
	 * Control whether the EGL context is preserved when the GLSurfaceView is paused and
	 * resumed.
	 * <p>
	 * If set to true, then the EGL context may be preserved when the GLSurfaceView is paused.
	 * <p>
	 * Prior to API level 11, whether the EGL context is actually preserved or not
	 * depends upon whether the Android device can support an arbitrary number of
	 * EGL contexts or not. Devices that can only support a limited number of EGL
	 * contexts must release the EGL context in order to allow multiple applications
	 * to share the GPU.
	 * <p>
	 * If set to false, the EGL context will be released when the GLSurfaceView is paused,
	 * and recreated when the GLSurfaceView is resumed.
	 * <p>
	 *
	 * The default is false.
	 *
	 * @param preserveOnPause preserve the EGL context when paused
	 */
	public void setPreserveEGLContextOnPause(boolean preserveOnPause) {
		mPreserveEGLContextOnPause = preserveOnPause;
	}

	/**
	 * @return true if the EGL context will be preserved when paused
	 */
	public boolean getPreserveEGLContextOnPause() {
		return mPreserveEGLContextOnPause;
	}

	/**
	 * Set the renderer associated with this view. Also starts the thread that
	 * will call the renderer, which in turn causes the rendering to start.
	 * <p>This method should be called once and only once in the life-cycle of
	 * a GLSurfaceView.
	 * <p>The following GLSurfaceView methods can only be called <em>before</em>
	 * setRenderer is called:
	 * <ul>
	 * <li>{@link #setEGLConfigChooser(boolean)}
	 * <li>{@link #setEGLConfigChooser(EGLConfigChooser)}
	 * <li>{@link #setEGLConfigChooser(int, int, int, int, int, int)}
	 * </ul>
	 * <p>
	 * The following GLSurfaceView methods can only be called <em>after</em>
	 * setRenderer is called:
	 * <ul>
	 * <li>{@link #getRenderMode()}
	 * <li>{@link #pauseGLThread()}
	 * <li>{@link #resumeGLThread()}
	 * <li>{@link #queueEvent(Runnable)}
	 * <li>{@link #requestRender()}
	 * <li>{@link #setRenderMode(int)}
	 * </ul>
	 *
	 * @param renderer the renderer to use to perform OpenGL drawing.
	 */
	public void setRenderer(Renderer renderer) {
		checkRenderThreadState();
		if (mEGLConfigChooser == null) {
			mEGLConfigChooser = new SimpleEGLConfigChooser(true);
		}
		if (mEGLContextFactory == null) {
			mEGLContextFactory = new DefaultContextFactory();
		}
		if (mEGLWindowSurfaceFactory == null) {
			mEGLWindowSurfaceFactory = new DefaultWindowSurfaceFactory();
		}
		mRenderer = renderer;
		mGLThread = new GLThread(mThisWeakRef);
		mGLThread.start();
	}

	/**
	 * Install a custom EGLContextFactory.
	 * <p>If this method is
	 * called, it must be called before {@link #setRenderer(Renderer)}
	 * is called.
	 * <p>
	 * If this method is not called, then by default
	 * a context will be created with no shared context and
	 * with a null attribute list.
	 */
	public void setEGLContextFactory(EGLContextFactory factory) {
		checkRenderThreadState();
		mEGLContextFactory = factory;
	}

	/**
	 * Install a custom EGLWindowSurfaceFactory.
	 * <p>If this method is
	 * called, it must be called before {@link #setRenderer(Renderer)}
	 * is called.
	 * <p>
	 * If this method is not called, then by default
	 * a window surface will be created with a null attribute list.
	 */
	public void setEGLWindowSurfaceFactory(EGLWindowSurfaceFactory factory) {
		checkRenderThreadState();
		mEGLWindowSurfaceFactory = factory;
	}

	/**
	 * Install a custom EGLConfigChooser.
	 * <p>If this method is
	 * called, it must be called before {@link #setRenderer(Renderer)}
	 * is called.
	 * <p>
	 * If no setEGLConfigChooser method is called, then by default the
	 * view will choose an EGLConfig that is compatible with the current
	 * android.view.Surface, with a depth buffer depth of
	 * at least 16 bits.
	 * @param configChooser
	 */
	public void setEGLConfigChooser(EGLConfigChooser configChooser) {
		checkRenderThreadState();
		mEGLConfigChooser = configChooser;
	}

	/**
	 * Install a config chooser which will choose a config
	 * as close to 16-bit RGB as possible, with or without an optional depth
	 * buffer as close to 16-bits as possible.
	 * <p>If this method is
	 * called, it must be called before {@link #setRenderer(Renderer)}
	 * is called.
	 * <p>
	 * If no setEGLConfigChooser method is called, then by default the
	 * view will choose an RGB_888 surface with a depth buffer depth of
	 * at least 16 bits.
	 *
	 * @param needDepth
	 */
	public void setEGLConfigChooser(boolean needDepth) {
		setEGLConfigChooser(new SimpleEGLConfigChooser(needDepth));
	}

	/**
	 * Install a config chooser which will choose a config
	 * with at least the specified depthSize and stencilSize,
	 * and exactly the specified redSize, greenSize, blueSize and alphaSize.
	 * <p>If this method is
	 * called, it must be called before {@link #setRenderer(Renderer)}
	 * is called.
	 * <p>
	 * If no setEGLConfigChooser method is called, then by default the
	 * view will choose an RGB_888 surface with a depth buffer depth of
	 * at least 16 bits.
	 *
	 */
	public void setEGLConfigChooser(int redSize, int greenSize, int blueSize,
									int alphaSize, int depthSize, int stencilSize) {
		setEGLConfigChooser(new ComponentSizeChooser(redSize, greenSize,
				blueSize, alphaSize, depthSize, stencilSize));
	}

	/**
	 * Inform the default EGLContextFactory and default EGLConfigChooser
	 * which EGLContext client version to pick.
	 * <p>Use this method to create an OpenGL ES 2.0-compatible context.
	 * Example:
	 * <pre class="prettyprint">
	 *     public MyView(Context context) {
	 *         super(context);
	 *         setEGLContextClientVersion(2); // Pick an OpenGL ES 2.0 context.
	 *         setRenderer(new MyRenderer());
	 *     }
	 * </pre>
	 * <p>Note: Activities which require OpenGL ES 2.0 should indicate this by
	 * setting @lt;uses-feature android:glEsVersion="0x00020000" /> in the activity's
	 * AndroidManifest.xml file.
	 * <p>If this method is called, it must be called before {@link #setRenderer(Renderer)}
	 * is called.
	 * <p>This method only affects the behavior of the default EGLContexFactory and the
	 * default EGLConfigChooser. If
	 * {@link #setEGLContextFactory(EGLContextFactory)} has been called, then the supplied
	 * EGLContextFactory is responsible for creating an OpenGL ES 2.0-compatible context.
	 * If
	 * {@link #setEGLConfigChooser(EGLConfigChooser)} has been called, then the supplied
	 * EGLConfigChooser is responsible for choosing an OpenGL ES 2.0-compatible config.
	 * @param version The EGLContext client version to choose. Use 2 for OpenGL ES 2.0
	 */
	public void setEGLContextClientVersion(int version) {
		checkRenderThreadState();
		mEGLContextClientVersion = version;
	}

	/**
	 * Set the rendering mode. When renderMode is
	 * RENDERMODE_CONTINUOUSLY, the renderer is called
	 * repeatedly to re-render the scene. When renderMode
	 * is RENDERMODE_WHEN_DIRTY, the renderer only rendered when the surface
	 * is created, or when {@link #requestRender} is called. Defaults to RENDERMODE_CONTINUOUSLY.
	 * <p>
	 * Using RENDERMODE_WHEN_DIRTY can improve battery life and overall system performance
	 * by allowing the GPU and CPU to idle when the view does not need to be updated.
	 * <p>
	 * This method can only be called after {@link #setRenderer(Renderer)}
	 *
	 * @param renderMode one of the RENDERMODE_X constants
	 * @see #RENDERMODE_CONTINUOUSLY
	 * @see #RENDERMODE_WHEN_DIRTY
	 */
	public void setRenderMode(int renderMode) {
		mGLThread.setRenderMode(renderMode);
	}

	/**
	 * Get the current rendering mode. May be called
	 * from any thread. Must not be called before a renderer has been set.
	 * @return the current rendering mode.
	 * @see #RENDERMODE_CONTINUOUSLY
	 * @see #RENDERMODE_WHEN_DIRTY
	 */
	public int getRenderMode() {
		return mGLThread.getRenderMode();
	}

	/**
	 * Request that the renderer render a frame.
	 * This method is typically used when the render mode has been set to
	 * {@link #RENDERMODE_WHEN_DIRTY}, so that frames are only rendered on demand.
	 * May be called
	 * from any thread. Must not be called before a renderer has been set.
	 */
	public void requestRender() {
		mGLThread.requestRender();
	}

	/**
	 * This method is part of the SurfaceHolder.Callback interface, and is
	 * not normally called or subclassed by clients of GLSurfaceView.
	 */
	public void surfaceCreated(SurfaceHolder holder) {
		mGLThread.surfaceCreated();
	}

	/**
	 * This method is part of the SurfaceHolder.Callback interface, and is
	 * not normally called or subclassed by clients of GLSurfaceView.
	 */
	public void surfaceDestroyed(SurfaceHolder holder) {
		// Surface will be destroyed when we return
		mGLThread.surfaceDestroyed();
	}

	/**
	 * This method is part of the SurfaceHolder.Callback interface, and is
	 * not normally called or subclassed by clients of GLSurfaceView.
	 */
	public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
		mGLThread.onWindowResize(w, h);
	}

	/**
	 * This method is part of the SurfaceHolder.Callback2 interface, and is
	 * not normally called or subclassed by clients of GLSurfaceView.
	 */
	@Override
	public void surfaceRedrawNeededAsync(SurfaceHolder holder, Runnable finishDrawing) {
		if (mGLThread != null) {
			mGLThread.requestRenderAndNotify(finishDrawing);
		}
	}

	/**
	 * This method is part of the SurfaceHolder.Callback2 interface, and is
	 * not normally called or subclassed by clients of GLSurfaceView.
	 */
	@Deprecated
	@Override
	public void surfaceRedrawNeeded(SurfaceHolder holder) {
		// Since we are part of the framework we know only surfaceRedrawNeededAsync
		// will be called.
	}


	// -- GODOT start --
	/**
	 * Pause the rendering thread, optionally tearing down the EGL context
	 * depending upon the value of {@link #setPreserveEGLContextOnPause(boolean)}.
	 *
	 * This method should be called when it is no longer desirable for the
	 * GLSurfaceView to continue rendering, such as in response to
	 * {@link android.app.Activity#onStop Activity.onStop}.
	 *
	 * Must not be called before a renderer has been set.
	 */
	protected final void pauseGLThread() {
		mGLThread.onPause();
	}

	/**
	 * Resumes the rendering thread, re-creating the OpenGL context if necessary. It
	 * is the counterpart to {@link #pauseGLThread()}.
	 *
	 * This method should typically be called in
	 * {@link android.app.Activity#onStart Activity.onStart}.
	 *
	 * Must not be called before a renderer has been set.
	 */
	protected final void resumeGLThread() {
		mGLThread.onResume();
	}

	/**
	 * Requests the render thread to exit and block until it does.
	 */
	protected final void requestRenderThreadExitAndWait() {
		if (mGLThread != null) {
			mGLThread.requestExitAndWait();
		}
	}

	/**
	 * Requests the render thread to exit and block up to the given timeInMs until it's done.
	 *
	 * @return true if the thread exited, false otherwise.
	 */
	protected final boolean requestRenderThreadExitAndWait(long timeInMs) {
		if (mGLThread != null) {
			return mGLThread.requestExitAndWait(timeInMs);
		}
		return false;
	}
	// -- GODOT end --

	/**
	 * Queue a runnable to be run on the GL rendering thread. This can be used
	 * to communicate with the Renderer on the rendering thread.
	 * Must not be called before a renderer has been set.
	 * @param r the runnable to be run on the GL rendering thread.
	 */
	public void queueEvent(Runnable r) {
		mGLThread.queueEvent(r);
	}

	/**
	 * This method is used as part of the View class and is not normally
	 * called or subclassed by clients of GLSurfaceView.
	 */
	@Override
	protected void onAttachedToWindow() {
		super.onAttachedToWindow();
		if (LOG_ATTACH_DETACH) {
			Log.d(TAG, "onAttachedToWindow reattach =" + mDetached);
		}
		if (mDetached && (mRenderer != null)) {
			int renderMode = RENDERMODE_CONTINUOUSLY;
			if (mGLThread != null) {
				renderMode = mGLThread.getRenderMode();
			}
			mGLThread = new GLThread(mThisWeakRef);
			if (renderMode != RENDERMODE_CONTINUOUSLY) {
				mGLThread.setRenderMode(renderMode);
			}
			mGLThread.start();
		}
		mDetached = false;
	}

	@Override
	protected void onDetachedFromWindow() {
		if (LOG_ATTACH_DETACH) {
			Log.d(TAG, "onDetachedFromWindow");
		}
		if (mGLThread != null) {
			mGLThread.requestExitAndWait();
		}
		mDetached = true;
		super.onDetachedFromWindow();
	}

	// ----------------------------------------------------------------------

	/**
	 * An interface used to wrap a GL interface.
	 * <p>Typically
	 * used for implementing debugging and tracing on top of the default
	 * GL interface. You would typically use this by creating your own class
	 * that implemented all the GL methods by delegating to another GL instance.
	 * Then you could add your own behavior before or after calling the
	 * delegate. All the GLWrapper would do was instantiate and return the
	 * wrapper GL instance:
	 * <pre class="prettyprint">
	 * class MyGLWrapper implements GLWrapper {
	 *     GL wrap(GL gl) {
	 *         return new MyGLImplementation(gl);
	 *     }
	 *     static class MyGLImplementation implements GL,GL10,GL11,... {
	 *         ...
	 *     }
	 * }
	 * </pre>
	 * @see #setGLWrapper(GLWrapper)
	 */
	public interface GLWrapper {
		/**
		 * Wraps a gl interface in another gl interface.
		 * @param gl a GL interface that is to be wrapped.
		 * @return either the input argument or another GL object that wraps the input argument.
		 */
		GL wrap(GL gl);
	}

	/**
	 * A generic renderer interface.
	 * <p>
	 * The renderer is responsible for making OpenGL calls to render a frame.
	 * <p>
	 * GLSurfaceView clients typically create their own classes that implement
	 * this interface, and then call {@link GLSurfaceView#setRenderer} to
	 * register the renderer with the GLSurfaceView.
	 * <p>
	 *
	 * <div class="special reference">
	 * <h3>Developer Guides</h3>
	 * <p>For more information about how to use OpenGL, read the
	 * <a href="{@docRoot}guide/topics/graphics/opengl.html">OpenGL</a> developer guide.</p>
	 * </div>
	 *
	 * <h3>Threading</h3>
	 * The renderer will be called on a separate thread, so that rendering
	 * performance is decoupled from the UI thread. Clients typically need to
	 * communicate with the renderer from the UI thread, because that's where
	 * input events are received. Clients can communicate using any of the
	 * standard Java techniques for cross-thread communication, or they can
	 * use the {@link GLSurfaceView#queueEvent(Runnable)} convenience method.
	 * <p>
	 * <h3>EGL Context Lost</h3>
	 * There are situations where the EGL rendering context will be lost. This
	 * typically happens when device wakes up after going to sleep. When
	 * the EGL context is lost, all OpenGL resources (such as textures) that are
	 * associated with that context will be automatically deleted. In order to
	 * keep rendering correctly, a renderer must recreate any lost resources
	 * that it still needs. The {@link #onSurfaceCreated(GL10, EGLConfig)} method
	 * is a convenient place to do this.
	 *
	 *
	 * @see #setRenderer(Renderer)
	 */
	public interface Renderer {
		/**
		 * Called when the surface is created or recreated.
		 * <p>
		 * Called when the rendering thread
		 * starts and whenever the EGL context is lost. The EGL context will typically
		 * be lost when the Android device awakes after going to sleep.
		 * <p>
		 * Since this method is called at the beginning of rendering, as well as
		 * every time the EGL context is lost, this method is a convenient place to put
		 * code to create resources that need to be created when the rendering
		 * starts, and that need to be recreated when the EGL context is lost.
		 * Textures are an example of a resource that you might want to create
		 * here.
		 * <p>
		 * Note that when the EGL context is lost, all OpenGL resources associated
		 * with that context will be automatically deleted. You do not need to call
		 * the corresponding "glDelete" methods such as glDeleteTextures to
		 * manually delete these lost resources.
		 * <p>
		 * @param gl the GL interface. Use <code>instanceof</code> to
		 * test if the interface supports GL11 or higher interfaces.
		 * @param config the EGLConfig of the created surface. Can be used
		 * to create matching pbuffers.
		 */
		void onSurfaceCreated(GL10 gl, EGLConfig config);

		/**
		 * Called when the surface changed size.
		 * <p>
		 * Called after the surface is created and whenever
		 * the OpenGL ES surface size changes.
		 * <p>
		 * Typically you will set your viewport here. If your camera
		 * is fixed then you could also set your projection matrix here:
		 * <pre class="prettyprint">
		 * void onSurfaceChanged(GL10 gl, int width, int height) {
		 *     gl.glViewport(0, 0, width, height);
		 *     // for a fixed camera, set the projection too
		 *     float ratio = (float) width / height;
		 *     gl.glMatrixMode(GL10.GL_PROJECTION);
		 *     gl.glLoadIdentity();
		 *     gl.glFrustumf(-ratio, ratio, -1, 1, 1, 10);
		 * }
		 * </pre>
		 * @param gl the GL interface. Use <code>instanceof</code> to
		 * test if the interface supports GL11 or higher interfaces.
		 * @param width
		 * @param height
		 */
		void onSurfaceChanged(GL10 gl, int width, int height);

		// -- GODOT start --
		/**
		 * Called to draw the current frame.
		 * <p>
		 * This method is responsible for drawing the current frame.
		 * <p>
		 * The implementation of this method typically looks like this:
		 * <pre class="prettyprint">
		 * boolean onDrawFrame(GL10 gl) {
		 *     gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);
		 *     //... other gl calls to render the scene ...
		 *     return true;
		 * }
		 * </pre>
		 * @param gl the GL interface. Use <code>instanceof</code> to
		 * test if the interface supports GL11 or higher interfaces.
		 *
		 * @return true if the buffers should be swapped, false otherwise.
		 */
		boolean onDrawFrame(GL10 gl);

		/**
		 * Invoked when the render thread is in the process of shutting down.
		 */
		void onRenderThreadExiting();
		// -- GODOT end --
	}

	/**
	 * An interface for customizing the eglCreateContext and eglDestroyContext calls.
	 * <p>
	 * This interface must be implemented by clients wishing to call
	 * {@link GLSurfaceView#setEGLContextFactory(EGLContextFactory)}
	 */
	public interface EGLContextFactory {
		EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig eglConfig);
		void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context);
	}

	private class DefaultContextFactory implements EGLContextFactory {
		private int EGL_CONTEXT_CLIENT_VERSION = 0x3098;

		public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig config) {
			int[] attrib_list = {EGL_CONTEXT_CLIENT_VERSION, mEGLContextClientVersion,
					EGL10.EGL_NONE };

			return egl.eglCreateContext(display, config, EGL10.EGL_NO_CONTEXT,
					mEGLContextClientVersion != 0 ? attrib_list : null);
		}

		public void destroyContext(EGL10 egl, EGLDisplay display,
								   EGLContext context) {
			if (!egl.eglDestroyContext(display, context)) {
				Log.e("DefaultContextFactory", "display:" + display + " context: " + context);
				if (LOG_THREADS) {
					Log.i("DefaultContextFactory", "tid=" + Thread.currentThread().getId());
				}
				EglHelper.throwEglException("eglDestroyContex", egl.eglGetError());
			}
		}
	}

	/**
	 * An interface for customizing the eglCreateWindowSurface and eglDestroySurface calls.
	 * <p>
	 * This interface must be implemented by clients wishing to call
	 * {@link GLSurfaceView#setEGLWindowSurfaceFactory(EGLWindowSurfaceFactory)}
	 */
	public interface EGLWindowSurfaceFactory {
		/**
		 *  @return null if the surface cannot be constructed.
		 */
		EGLSurface createWindowSurface(EGL10 egl, EGLDisplay display, EGLConfig config,
									   Object nativeWindow);
		void destroySurface(EGL10 egl, EGLDisplay display, EGLSurface surface);
	}

	private static class DefaultWindowSurfaceFactory implements EGLWindowSurfaceFactory {

		public EGLSurface createWindowSurface(EGL10 egl, EGLDisplay display,
											  EGLConfig config, Object nativeWindow) {
			EGLSurface result = null;
			try {
				result = egl.eglCreateWindowSurface(display, config, nativeWindow, null);
			} catch (IllegalArgumentException e) {
				// This exception indicates that the surface flinger surface
				// is not valid. This can happen if the surface flinger surface has
				// been torn down, but the application has not yet been
				// notified via SurfaceHolder.Callback.surfaceDestroyed.
				// In theory the application should be notified first,
				// but in practice sometimes it is not. See b/4588890
				Log.e(TAG, "eglCreateWindowSurface", e);
			}
			return result;
		}

		public void destroySurface(EGL10 egl, EGLDisplay display,
								   EGLSurface surface) {
			egl.eglDestroySurface(display, surface);
		}
	}

	/**
	 * An interface for choosing an EGLConfig configuration from a list of
	 * potential configurations.
	 * <p>
	 * This interface must be implemented by clients wishing to call
	 * {@link GLSurfaceView#setEGLConfigChooser(EGLConfigChooser)}
	 */
	public interface EGLConfigChooser {
		/**
		 * Choose a configuration from the list. Implementers typically
		 * implement this method by calling
		 * {@link EGL10#eglChooseConfig} and iterating through the results. Please consult the
		 * EGL specification available from The Khronos Group to learn how to call eglChooseConfig.
		 * @param egl the EGL10 for the current display.
		 * @param display the current display.
		 * @return the chosen configuration.
		 */
		EGLConfig chooseConfig(EGL10 egl, EGLDisplay display);
	}

	private abstract class BaseConfigChooser
			implements EGLConfigChooser {
		public BaseConfigChooser(int[] configSpec) {
			mConfigSpec = filterConfigSpec(configSpec);
		}

		public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display) {
			int[] num_config = new int[1];
			if (!egl.eglChooseConfig(display, mConfigSpec, null, 0,
					num_config)) {
				throw new IllegalArgumentException("eglChooseConfig failed");
			}

			int numConfigs = num_config[0];

			if (numConfigs <= 0) {
				throw new IllegalArgumentException(
						"No configs match configSpec");
			}

			EGLConfig[] configs = new EGLConfig[numConfigs];
			if (!egl.eglChooseConfig(display, mConfigSpec, configs, numConfigs,
					num_config)) {
				throw new IllegalArgumentException("eglChooseConfig#2 failed");
			}
			EGLConfig config = chooseConfig(egl, display, configs);
			if (config == null) {
				throw new IllegalArgumentException("No config chosen");
			}
			return config;
		}

		abstract EGLConfig chooseConfig(EGL10 egl, EGLDisplay display,
										EGLConfig[] configs);

		protected int[] mConfigSpec;

		private int[] filterConfigSpec(int[] configSpec) {
			if (mEGLContextClientVersion != 2 && mEGLContextClientVersion != 3) {
				return configSpec;
			}
			/* We know none of the subclasses define EGL_RENDERABLE_TYPE.
			 * And we know the configSpec is well formed.
			 */
			int len = configSpec.length;
			int[] newConfigSpec = new int[len + 2];
			System.arraycopy(configSpec, 0, newConfigSpec, 0, len-1);
			newConfigSpec[len-1] = EGL10.EGL_RENDERABLE_TYPE;
			if (mEGLContextClientVersion == 2) {
				newConfigSpec[len] = EGL14.EGL_OPENGL_ES2_BIT;  /* EGL_OPENGL_ES2_BIT */
			} else {
				newConfigSpec[len] = EGLExt.EGL_OPENGL_ES3_BIT_KHR; /* EGL_OPENGL_ES3_BIT_KHR */
			}
			newConfigSpec[len+1] = EGL10.EGL_NONE;
			return newConfigSpec;
		}
	}

	/**
	 * Choose a configuration with exactly the specified r,g,b,a sizes,
	 * and at least the specified depth and stencil sizes.
	 */
	private class ComponentSizeChooser extends BaseConfigChooser {
		public ComponentSizeChooser(int redSize, int greenSize, int blueSize,
									int alphaSize, int depthSize, int stencilSize) {
			super(new int[] {
					EGL10.EGL_RED_SIZE, redSize,
					EGL10.EGL_GREEN_SIZE, greenSize,
					EGL10.EGL_BLUE_SIZE, blueSize,
					EGL10.EGL_ALPHA_SIZE, alphaSize,
					EGL10.EGL_DEPTH_SIZE, depthSize,
					EGL10.EGL_STENCIL_SIZE, stencilSize,
					EGL10.EGL_NONE});
			mValue = new int[1];
			mRedSize = redSize;
			mGreenSize = greenSize;
			mBlueSize = blueSize;
			mAlphaSize = alphaSize;
			mDepthSize = depthSize;
			mStencilSize = stencilSize;
		}

		@Override
		public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display,
									  EGLConfig[] configs) {
			for (EGLConfig config : configs) {
				int d = findConfigAttrib(egl, display, config,
						EGL10.EGL_DEPTH_SIZE, 0);
				int s = findConfigAttrib(egl, display, config,
						EGL10.EGL_STENCIL_SIZE, 0);
				if ((d >= mDepthSize) && (s >= mStencilSize)) {
					int r = findConfigAttrib(egl, display, config,
							EGL10.EGL_RED_SIZE, 0);
					int g = findConfigAttrib(egl, display, config,
							EGL10.EGL_GREEN_SIZE, 0);
					int b = findConfigAttrib(egl, display, config,
							EGL10.EGL_BLUE_SIZE, 0);
					int a = findConfigAttrib(egl, display, config,
							EGL10.EGL_ALPHA_SIZE, 0);
					if ((r == mRedSize) && (g == mGreenSize)
							&& (b == mBlueSize) && (a == mAlphaSize)) {
						return config;
					}
				}
			}
			return null;
		}

		private int findConfigAttrib(EGL10 egl, EGLDisplay display,
									 EGLConfig config, int attribute, int defaultValue) {

			if (egl.eglGetConfigAttrib(display, config, attribute, mValue)) {
				return mValue[0];
			}
			return defaultValue;
		}

		private int[] mValue;
		// Subclasses can adjust these values:
		protected int mRedSize;
		protected int mGreenSize;
		protected int mBlueSize;
		protected int mAlphaSize;
		protected int mDepthSize;
		protected int mStencilSize;
	}

	/**
	 * This class will choose a RGB_888 surface with
	 * or without a depth buffer.
	 *
	 */
	private class SimpleEGLConfigChooser extends ComponentSizeChooser {
		public SimpleEGLConfigChooser(boolean withDepthBuffer) {
			super(8, 8, 8, 0, withDepthBuffer ? 16 : 0, 0);
		}
	}

	/**
	 * An EGL helper class.
	 */

	private static class EglHelper {
		public EglHelper(WeakReference<GLSurfaceView> glSurfaceViewWeakRef) {
			mGLSurfaceViewWeakRef = glSurfaceViewWeakRef;
		}

		/**
		 * Initialize EGL for a given configuration spec.
		 */
		public void start() {
			if (LOG_EGL) {
				Log.w("EglHelper", "start() tid=" + Thread.currentThread().getId());
			}
			/*
			 * Get an EGL instance
			 */
			mEgl = (EGL10) EGLContext.getEGL();

			/*
			 * Get to the default display.
			 */
			mEglDisplay = mEgl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);

			if (mEglDisplay == EGL10.EGL_NO_DISPLAY) {
				throw new RuntimeException("eglGetDisplay failed");
			}

			/*
			 * We can now initialize EGL for that display
			 */
			int[] version = new int[2];
			if(!mEgl.eglInitialize(mEglDisplay, version)) {
				throw new RuntimeException("eglInitialize failed");
			}
			GLSurfaceView view = mGLSurfaceViewWeakRef.get();
			if (view == null) {
				mEglConfig = null;
				mEglContext = null;
			} else {
				mEglConfig = view.mEGLConfigChooser.chooseConfig(mEgl, mEglDisplay);

				/*
				 * Create an EGL context. We want to do this as rarely as we can, because an
				 * EGL context is a somewhat heavy object.
				 */
				mEglContext = view.mEGLContextFactory.createContext(mEgl, mEglDisplay, mEglConfig);
			}
			if (mEglContext == null || mEglContext == EGL10.EGL_NO_CONTEXT) {
				mEglContext = null;
				throwEglException("createContext");
			}
			if (LOG_EGL) {
				Log.w("EglHelper", "createContext " + mEglContext + " tid=" + Thread.currentThread().getId());
			}

			mEglSurface = null;
		}

		/**
		 * Create an egl surface for the current SurfaceHolder surface. If a surface
		 * already exists, destroy it before creating the new surface.
		 *
		 * @return true if the surface was created successfully.
		 */
		public boolean createSurface() {
			if (LOG_EGL) {
				Log.w("EglHelper", "createSurface()  tid=" + Thread.currentThread().getId());
			}
			/*
			 * Check preconditions.
			 */
			if (mEgl == null) {
				throw new RuntimeException("egl not initialized");
			}
			if (mEglDisplay == null) {
				throw new RuntimeException("eglDisplay not initialized");
			}
			if (mEglConfig == null) {
				throw new RuntimeException("mEglConfig not initialized");
			}

			/*
			 *  The window size has changed, so we need to create a new
			 *  surface.
			 */
			destroySurfaceImp();

			/*
			 * Create an EGL surface we can render into.
			 */
			GLSurfaceView view = mGLSurfaceViewWeakRef.get();
			if (view != null) {
				mEglSurface = view.mEGLWindowSurfaceFactory.createWindowSurface(mEgl,
						mEglDisplay, mEglConfig, view.getHolder());
			} else {
				mEglSurface = null;
			}

			if (mEglSurface == null || mEglSurface == EGL10.EGL_NO_SURFACE) {
				int error = mEgl.eglGetError();
				if (error == EGL10.EGL_BAD_NATIVE_WINDOW) {
					Log.e("EglHelper", "createWindowSurface returned EGL_BAD_NATIVE_WINDOW.");
				}
				return false;
			}

			/*
			 * Before we can issue GL commands, we need to make sure
			 * the context is current and bound to a surface.
			 */
			if (!mEgl.eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface, mEglContext)) {
				/*
				 * Could not make the context current, probably because the underlying
				 * SurfaceView surface has been destroyed.
				 */
				logEglErrorAsWarning("EGLHelper", "eglMakeCurrent", mEgl.eglGetError());
				return false;
			}

			return true;
		}

		/**
		 * Create a GL object for the current EGL context.
		 * @return
		 */
		GL createGL() {

			GL gl = mEglContext.getGL();
			GLSurfaceView view = mGLSurfaceViewWeakRef.get();
			if (view != null) {
				if (view.mGLWrapper != null) {
					gl = view.mGLWrapper.wrap(gl);
				}

				if ((view.mDebugFlags & (DEBUG_CHECK_GL_ERROR | DEBUG_LOG_GL_CALLS)) != 0) {
					int configFlags = 0;
					Writer log = null;
					if ((view.mDebugFlags & DEBUG_CHECK_GL_ERROR) != 0) {
						configFlags |= GLDebugHelper.CONFIG_CHECK_GL_ERROR;
					}
					if ((view.mDebugFlags & DEBUG_LOG_GL_CALLS) != 0) {
						log = new LogWriter();
					}
					gl = GLDebugHelper.wrap(gl, configFlags, log);
				}
			}
			return gl;
		}

		/**
		 * Display the current render surface.
		 * @return the EGL error code from eglSwapBuffers.
		 */
		public int swap() {
			if (! mEgl.eglSwapBuffers(mEglDisplay, mEglSurface)) {
				return mEgl.eglGetError();
			}
			return EGL10.EGL_SUCCESS;
		}

		public void destroySurface() {
			if (LOG_EGL) {
				Log.w("EglHelper", "destroySurface()  tid=" + Thread.currentThread().getId());
			}
			destroySurfaceImp();
		}

		private void destroySurfaceImp() {
			if (mEglSurface != null && mEglSurface != EGL10.EGL_NO_SURFACE) {
				mEgl.eglMakeCurrent(mEglDisplay, EGL10.EGL_NO_SURFACE,
						EGL10.EGL_NO_SURFACE,
						EGL10.EGL_NO_CONTEXT);
				GLSurfaceView view = mGLSurfaceViewWeakRef.get();
				if (view != null) {
					view.mEGLWindowSurfaceFactory.destroySurface(mEgl, mEglDisplay, mEglSurface);
				}
				mEglSurface = null;
			}
		}

		public void finish() {
			if (LOG_EGL) {
				Log.w("EglHelper", "finish() tid=" + Thread.currentThread().getId());
			}
			if (mEglContext != null) {
				GLSurfaceView view = mGLSurfaceViewWeakRef.get();
				if (view != null) {
					view.mEGLContextFactory.destroyContext(mEgl, mEglDisplay, mEglContext);
				}
				mEglContext = null;
			}
			if (mEglDisplay != null) {
				mEgl.eglTerminate(mEglDisplay);
				mEglDisplay = null;
			}
		}

		private void throwEglException(String function) {
			throwEglException(function, mEgl.eglGetError());
		}

		public static void throwEglException(String function, int error) {
			String message = formatEglError(function, error);
			if (LOG_THREADS) {
				Log.e("EglHelper", "throwEglException tid=" + Thread.currentThread().getId() + " "
						+ message);
			}
			throw new RuntimeException(message);
		}

		public static void logEglErrorAsWarning(String tag, String function, int error) {
			Log.w(tag, formatEglError(function, error));
		}

		public static String formatEglError(String function, int error) {
			return function + " failed: " + EGLLogWrapper.getErrorString(error);
		}

		private WeakReference<GLSurfaceView> mGLSurfaceViewWeakRef;
		EGL10 mEgl;
		EGLDisplay mEglDisplay;
		EGLSurface mEglSurface;
		EGLConfig mEglConfig;
		EGLContext mEglContext;

	}

	/**
	 * A generic GL Thread. Takes care of initializing EGL and GL. Delegates
	 * to a Renderer instance to do the actual drawing. Can be configured to
	 * render continuously or on request.
	 *
	 * All potentially blocking synchronization is done through the
	 * sGLThreadManager object. This avoids multiple-lock ordering issues.
	 *
	 */
	static class GLThread extends Thread {
		GLThread(WeakReference<GLSurfaceView> glSurfaceViewWeakRef) {
			super();
			mWidth = 0;
			mHeight = 0;
			mRequestRender = true;
			mRenderMode = RENDERMODE_CONTINUOUSLY;
			mWantRenderNotification = false;
			mGLSurfaceViewWeakRef = glSurfaceViewWeakRef;
		}

		@Override
		public void run() {
			setName("GLThread " + getId());
			if (LOG_THREADS) {
				Log.i("GLThread", "starting tid=" + getId());
			}

			try {
				guardedRun();
			} catch (InterruptedException e) {
				// fall thru and exit normally
			} finally {
				sGLThreadManager.threadExiting(this);
			}
		}

		/*
		 * This private method should only be called inside a
		 * synchronized(sGLThreadManager) block.
		 */
		private void stopEglSurfaceLocked() {
			if (mHaveEglSurface) {
				mHaveEglSurface = false;
				mEglHelper.destroySurface();
			}
		}

		/*
		 * This private method should only be called inside a
		 * synchronized(sGLThreadManager) block.
		 */
		private void stopEglContextLocked() {
			if (mHaveEglContext) {
				mEglHelper.finish();
				mHaveEglContext = false;
				sGLThreadManager.releaseEglContextLocked(this);
			}
		}
		private void guardedRun() throws InterruptedException {
			mEglHelper = new EglHelper(mGLSurfaceViewWeakRef);
			mHaveEglContext = false;
			mHaveEglSurface = false;
			mWantRenderNotification = false;

			try {
				GL10 gl = null;
				boolean createEglContext = false;
				boolean createEglSurface = false;
				boolean createGlInterface = false;
				boolean lostEglContext = false;
				boolean sizeChanged = false;
				boolean wantRenderNotification = false;
				boolean doRenderNotification = false;
				boolean askedToReleaseEglContext = false;
				int w = 0;
				int h = 0;
				Runnable event = null;
				Runnable finishDrawingRunnable = null;

				while (true) {
					synchronized (sGLThreadManager) {
						while (true) {
							if (mShouldExit) {
								return;
							}

							if (! mEventQueue.isEmpty()) {
								event = mEventQueue.remove(0);
								break;
							}

							// Update the pause state.
							boolean pausing = false;
							if (mPaused != mRequestPaused) {
								pausing = mRequestPaused;
								mPaused = mRequestPaused;
								sGLThreadManager.notifyAll();
								if (LOG_PAUSE_RESUME) {
									Log.i("GLThread", "mPaused is now " + mPaused + " tid=" + getId());
								}
							}

							// Do we need to give up the EGL context?
							if (mShouldReleaseEglContext) {
								if (LOG_SURFACE) {
									Log.i("GLThread", "releasing EGL context because asked to tid=" + getId());
								}
								stopEglSurfaceLocked();
								stopEglContextLocked();
								mShouldReleaseEglContext = false;
								askedToReleaseEglContext = true;
							}

							// Have we lost the EGL context?
							if (lostEglContext) {
								stopEglSurfaceLocked();
								stopEglContextLocked();
								lostEglContext = false;
							}

							// When pausing, release the EGL surface:
							if (pausing && mHaveEglSurface) {
								if (LOG_SURFACE) {
									Log.i("GLThread", "releasing EGL surface because paused tid=" + getId());
								}
								stopEglSurfaceLocked();
							}

							// When pausing, optionally release the EGL Context:
							if (pausing && mHaveEglContext) {
								GLSurfaceView view = mGLSurfaceViewWeakRef.get();
								boolean preserveEglContextOnPause = view == null ?
										false : view.mPreserveEGLContextOnPause;
								if (!preserveEglContextOnPause) {
									stopEglContextLocked();
									if (LOG_SURFACE) {
										Log.i("GLThread", "releasing EGL context because paused tid=" + getId());
									}
								}
							}

							// Have we lost the SurfaceView surface?
							if ((! mHasSurface) && (! mWaitingForSurface)) {
								if (LOG_SURFACE) {
									Log.i("GLThread", "noticed surfaceView surface lost tid=" + getId());
								}
								if (mHaveEglSurface) {
									stopEglSurfaceLocked();
								}
								mWaitingForSurface = true;
								mSurfaceIsBad = false;
								sGLThreadManager.notifyAll();
							}

							// Have we acquired the surface view surface?
							if (mHasSurface && mWaitingForSurface) {
								if (LOG_SURFACE) {
									Log.i("GLThread", "noticed surfaceView surface acquired tid=" + getId());
								}
								mWaitingForSurface = false;
								sGLThreadManager.notifyAll();
							}

							if (doRenderNotification) {
								if (LOG_SURFACE) {
									Log.i("GLThread", "sending render notification tid=" + getId());
								}
								mWantRenderNotification = false;
								doRenderNotification = false;
								mRenderComplete = true;
								sGLThreadManager.notifyAll();
							}

							if (mFinishDrawingRunnable != null) {
								finishDrawingRunnable = mFinishDrawingRunnable;
								mFinishDrawingRunnable = null;
							}

							// Ready to draw?
							if (readyToDraw()) {

								// If we don't have an EGL context, try to acquire one.
								if (! mHaveEglContext) {
									if (askedToReleaseEglContext) {
										askedToReleaseEglContext = false;
									} else {
										try {
											mEglHelper.start();
										} catch (RuntimeException t) {
											sGLThreadManager.releaseEglContextLocked(this);
											throw t;
										}
										mHaveEglContext = true;
										createEglContext = true;

										sGLThreadManager.notifyAll();
									}
								}

								if (mHaveEglContext && !mHaveEglSurface) {
									mHaveEglSurface = true;
									createEglSurface = true;
									createGlInterface = true;
									sizeChanged = true;
								}

								if (mHaveEglSurface) {
									if (mSizeChanged) {
										sizeChanged = true;
										w = mWidth;
										h = mHeight;
										mWantRenderNotification = true;
										if (LOG_SURFACE) {
											Log.i("GLThread",
													"noticing that we want render notification tid="
															+ getId());
										}

										// Destroy and recreate the EGL surface.
										createEglSurface = true;

										mSizeChanged = false;
									}
									mRequestRender = false;
									sGLThreadManager.notifyAll();
									if (mWantRenderNotification) {
										wantRenderNotification = true;
									}
									break;
								}
							} else {
								if (finishDrawingRunnable != null) {
									Log.w(TAG, "Warning, !readyToDraw() but waiting for " +
											"draw finished! Early reporting draw finished.");
									finishDrawingRunnable.run();
									finishDrawingRunnable = null;
								}
							}
							// By design, this is the only place in a GLThread thread where we wait().
							if (LOG_THREADS) {
								Log.i("GLThread", "waiting tid=" + getId()
										+ " mHaveEglContext: " + mHaveEglContext
										+ " mHaveEglSurface: " + mHaveEglSurface
										+ " mFinishedCreatingEglSurface: " + mFinishedCreatingEglSurface
										+ " mPaused: " + mPaused
										+ " mHasSurface: " + mHasSurface
										+ " mSurfaceIsBad: " + mSurfaceIsBad
										+ " mWaitingForSurface: " + mWaitingForSurface
										+ " mWidth: " + mWidth
										+ " mHeight: " + mHeight
										+ " mRequestRender: " + mRequestRender
										+ " mRenderMode: " + mRenderMode);
							}
							sGLThreadManager.wait();
						}
					} // end of synchronized(sGLThreadManager)

					if (event != null) {
						event.run();
						event = null;
						continue;
					}

					if (createEglSurface) {
						if (LOG_SURFACE) {
							Log.w("GLThread", "egl createSurface");
						}
						if (mEglHelper.createSurface()) {
							synchronized(sGLThreadManager) {
								mFinishedCreatingEglSurface = true;
								sGLThreadManager.notifyAll();
							}
						} else {
							synchronized(sGLThreadManager) {
								mFinishedCreatingEglSurface = true;
								mSurfaceIsBad = true;
								sGLThreadManager.notifyAll();
							}
							continue;
						}
						createEglSurface = false;
					}

					if (createGlInterface) {
						gl = (GL10) mEglHelper.createGL();

						createGlInterface = false;
					}

					// -- GODOT start --
					if (createEglContext) {
						if (LOG_RENDERER) {
							Log.w("GLThread", "onSurfaceCreated");
						}
						GLSurfaceView view = mGLSurfaceViewWeakRef.get();
						if (view != null) {
							try {
								view.mRenderer.onSurfaceCreated(gl, mEglHelper.mEglConfig);
							} finally {
							}
						}
						createEglContext = false;
					}

					if (sizeChanged) {
						if (LOG_RENDERER) {
							Log.w("GLThread", "onSurfaceChanged(" + w + ", " + h + ")");
						}
						GLSurfaceView view = mGLSurfaceViewWeakRef.get();
						if (view != null) {
							try {
								view.mRenderer.onSurfaceChanged(gl, w, h);
							} finally {
							}
						}
						sizeChanged = false;
					}

					boolean swapBuffers = false;
					if (LOG_RENDERER_DRAW_FRAME) {
						Log.w("GLThread", "onDrawFrame tid=" + getId());
					}
					{
						GLSurfaceView view = mGLSurfaceViewWeakRef.get();
						if (view != null) {
							try {
								swapBuffers = view.mRenderer.onDrawFrame(gl);
								if (finishDrawingRunnable != null) {
									finishDrawingRunnable.run();
									finishDrawingRunnable = null;
								}
							} finally {}
						}
					}
					if (swapBuffers) {
						int swapError = mEglHelper.swap();
						switch (swapError) {
							case EGL10.EGL_SUCCESS:
								break;
							case EGL11.EGL_CONTEXT_LOST:
								if (LOG_SURFACE) {
									Log.i("GLThread", "egl context lost tid=" + getId());
								}
								lostEglContext = true;
								break;
							default:
								// Other errors typically mean that the current surface is bad,
								// probably because the SurfaceView surface has been destroyed,
								// but we haven't been notified yet.
								// Log the error to help developers understand why rendering stopped.
								EglHelper.logEglErrorAsWarning("GLThread", "eglSwapBuffers", swapError);

								synchronized (sGLThreadManager) {
									mSurfaceIsBad = true;
									sGLThreadManager.notifyAll();
								}
								break;
						}
					}
					// -- GODOT end --

					if (wantRenderNotification) {
						doRenderNotification = true;
						wantRenderNotification = false;
					}
				}

			} finally {
				/*
				 * clean-up everything...
				 */
				synchronized (sGLThreadManager) {
					Log.d("GLThread", "Exiting render thread");
					GLSurfaceView view = mGLSurfaceViewWeakRef.get();
					if (view != null) {
						view.mRenderer.onRenderThreadExiting();
					}

					stopEglSurfaceLocked();
					stopEglContextLocked();
				}
			}
		}

		public boolean ableToDraw() {
			return mHaveEglContext && mHaveEglSurface && readyToDraw();
		}

		private boolean readyToDraw() {
			return (!mPaused) && mHasSurface && (!mSurfaceIsBad)
					&& (mWidth > 0) && (mHeight > 0)
					&& (mRequestRender || (mRenderMode == RENDERMODE_CONTINUOUSLY));
		}

		public void setRenderMode(int renderMode) {
			if ( !((RENDERMODE_WHEN_DIRTY <= renderMode) && (renderMode <= RENDERMODE_CONTINUOUSLY)) ) {
				throw new IllegalArgumentException("renderMode");
			}
			synchronized(sGLThreadManager) {
				mRenderMode = renderMode;
				sGLThreadManager.notifyAll();
			}
		}

		public int getRenderMode() {
			synchronized(sGLThreadManager) {
				return mRenderMode;
			}
		}

		public void requestRender() {
			synchronized(sGLThreadManager) {
				mRequestRender = true;
				sGLThreadManager.notifyAll();
			}
		}

		public void requestRenderAndNotify(Runnable finishDrawing) {
			synchronized(sGLThreadManager) {
				// If we are already on the GL thread, this means a client callback
				// has caused reentrancy, for example via updating the SurfaceView parameters.
				// We will return to the client rendering code, so here we don't need to
				// do anything.
				if (Thread.currentThread() == this) {
					return;
				}

				mWantRenderNotification = true;
				mRequestRender = true;
				mRenderComplete = false;

				// fix lost old callback when continuous call requestRenderAndNotify
				//
				// If continuous call requestRenderAndNotify before trigger old
				// callback, old callback will lose, cause VRI will wait for SV's
				// draw to finish forever not calling finishDraw.
				// https://android.googlesource.com/platform/frameworks/base/+/044fce0b826f2da3a192aac56785b5089143e693%5E%21/
				//+++++++++++++++++++++++++++++++++++++++++++++++++++
				final Runnable oldCallback = mFinishDrawingRunnable;
				mFinishDrawingRunnable = () -> {
					if (oldCallback != null) {
						oldCallback.run();
					}
					if (finishDrawing != null) {
						finishDrawing.run();
					}
				};
				//----------------------------------------------------

				sGLThreadManager.notifyAll();
			}
		}

		public void surfaceCreated() {
			synchronized(sGLThreadManager) {
				if (LOG_THREADS) {
					Log.i("GLThread", "surfaceCreated tid=" + getId());
				}
				mHasSurface = true;
				mFinishedCreatingEglSurface = false;
				sGLThreadManager.notifyAll();
			}
		}

		public void surfaceDestroyed() {
			synchronized(sGLThreadManager) {
				if (LOG_THREADS) {
					Log.i("GLThread", "surfaceDestroyed tid=" + getId());
				}
				mHasSurface = false;
				sGLThreadManager.notifyAll();
			}
		}

		public void onPause() {
			synchronized (sGLThreadManager) {
				if (LOG_PAUSE_RESUME) {
					Log.i("GLThread", "onPause tid=" + getId());
				}
				mRequestPaused = true;
				sGLThreadManager.notifyAll();
			}
		}

		public void onResume() {
			synchronized (sGLThreadManager) {
				if (LOG_PAUSE_RESUME) {
					Log.i("GLThread", "onResume tid=" + getId());
				}
				mRequestPaused = false;
				mRequestRender = true;
				mRenderComplete = false;
				sGLThreadManager.notifyAll();
			}
		}

		public void onWindowResize(int w, int h) {
			synchronized (sGLThreadManager) {
				mWidth = w;
				mHeight = h;
				mSizeChanged = true;
				mRequestRender = true;
				mRenderComplete = false;

				// If we are already on the GL thread, this means a client callback
				// has caused reentrancy, for example via updating the SurfaceView parameters.
				// We need to process the size change eventually though and update our EGLSurface.
				// So we set the parameters and return so they can be processed on our
				// next iteration.
				if (Thread.currentThread() == this) {
					return;
				}

				sGLThreadManager.notifyAll();
			}
		}

		public void requestExitAndWait() {
			// don't call this from GLThread thread or it is a guaranteed
			// deadlock!
			synchronized(sGLThreadManager) {
				mShouldExit = true;
				sGLThreadManager.notifyAll();
				while (! mExited) {
					try {
						sGLThreadManager.wait();
					} catch (InterruptedException ex) {
						Thread.currentThread().interrupt();
					}
				}
			}
		}

		public boolean requestExitAndWait(long timeInMs) {
			// don't call this from GLThread thread or it is a guaranteed
			// deadlock!
			synchronized(sGLThreadManager) {
				mShouldExit = true;
				sGLThreadManager.notifyAll();
				if (!mExited) {
					try {
						sGLThreadManager.wait(timeInMs);
					} catch (InterruptedException ex) {
						Thread.currentThread().interrupt();
					}
				}

				return mExited;
			}
		}

		public void requestReleaseEglContextLocked() {
			mShouldReleaseEglContext = true;
			sGLThreadManager.notifyAll();
		}

		/**
		 * Queue an "event" to be run on the GL rendering thread.
		 * @param r the runnable to be run on the GL rendering thread.
		 */
		public void queueEvent(Runnable r) {
			if (r == null) {
				throw new IllegalArgumentException("r must not be null");
			}
			synchronized(sGLThreadManager) {
				mEventQueue.add(r);
				sGLThreadManager.notifyAll();
			}
		}

		// Once the thread is started, all accesses to the following member
		// variables are protected by the sGLThreadManager monitor
		private boolean mShouldExit;
		private boolean mExited;
		private boolean mRequestPaused;
		private boolean mPaused;
		private boolean mHasSurface;
		private boolean mSurfaceIsBad;
		private boolean mWaitingForSurface;
		private boolean mHaveEglContext;
		private boolean mHaveEglSurface;
		private boolean mFinishedCreatingEglSurface;
		private boolean mShouldReleaseEglContext;
		private int mWidth;
		private int mHeight;
		private int mRenderMode;
		private boolean mRequestRender;
		private boolean mWantRenderNotification;
		private boolean mRenderComplete;
		private ArrayList<Runnable> mEventQueue = new ArrayList<Runnable>();
		private boolean mSizeChanged = true;
		private Runnable mFinishDrawingRunnable = null;

		// End of member variables protected by the sGLThreadManager monitor.

		private EglHelper mEglHelper;

		/**
		 * Set once at thread construction time, nulled out when the parent view is garbage
		 * called. This weak reference allows the GLSurfaceView to be garbage collected while
		 * the GLThread is still alive.
		 */
		private WeakReference<GLSurfaceView> mGLSurfaceViewWeakRef;

	}

	static class LogWriter extends Writer {

		@Override public void close() {
			flushBuilder();
		}

		@Override public void flush() {
			flushBuilder();
		}

		@Override public void write(char[] buf, int offset, int count) {
			for(int i = 0; i < count; i++) {
				char c = buf[offset + i];
				if ( c == '\n') {
					flushBuilder();
				}
				else {
					mBuilder.append(c);
				}
			}
		}

		private void flushBuilder() {
			if (mBuilder.length() > 0) {
				Log.v("GLSurfaceView", mBuilder.toString());
				mBuilder.delete(0, mBuilder.length());
			}
		}

		private StringBuilder mBuilder = new StringBuilder();
	}


	private void checkRenderThreadState() {
		if (mGLThread != null) {
			throw new IllegalStateException(
					"setRenderer has already been called for this instance.");
		}
	}

	private static class GLThreadManager {
		private static String TAG = "GLThreadManager";

		public synchronized void threadExiting(GLThread thread) {
			if (LOG_THREADS) {
				Log.i("GLThread", "exiting tid=" +  thread.getId());
			}
			thread.mExited = true;
			notifyAll();
		}

		/*
		 * Releases the EGL context. Requires that we are already in the
		 * sGLThreadManager monitor when this is called.
		 */
		public void releaseEglContextLocked(GLThread thread) {
			notifyAll();
		}
	}

	private static final GLThreadManager sGLThreadManager = new GLThreadManager();

	private final WeakReference<GLSurfaceView> mThisWeakRef =
			new WeakReference<GLSurfaceView>(this);
	private GLThread mGLThread;
	private Renderer mRenderer;
	private boolean mDetached;
	private EGLConfigChooser mEGLConfigChooser;
	private EGLContextFactory mEGLContextFactory;
	private EGLWindowSurfaceFactory mEGLWindowSurfaceFactory;
	private GLWrapper mGLWrapper;
	private int mDebugFlags;
	private int mEGLContextClientVersion;
	private boolean mPreserveEGLContextOnPause;
}
