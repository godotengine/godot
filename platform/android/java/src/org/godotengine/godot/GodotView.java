/*************************************************************************/
/*  GodotView.java                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
import android.content.Context;
import android.graphics.PixelFormat;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.content.ContextWrapper;
import android.view.InputDevice;
import android.hardware.input.InputManager;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.opengles.GL10;

import org.godotengine.godot.input.InputManagerCompat;
import org.godotengine.godot.input.InputManagerCompat.InputDeviceListener;
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
public class GodotView extends GLSurfaceView implements InputDeviceListener {

	private static String TAG = "GodotView";
	private static final boolean DEBUG = false;
	private static Context ctx;

	private static GodotIO io;
	private static boolean firsttime=true;
	private static boolean use_gl3=false;
	private static boolean use_32=false;

	private Godot activity;


	private InputManagerCompat mInputManager;
	public GodotView(Context context,GodotIO p_io,boolean p_use_gl3, boolean p_use_32_bits, Godot p_activity) {
		super(context);
		ctx=context;
		io=p_io;
		use_gl3=p_use_gl3;
		use_32=p_use_32_bits;

		activity = p_activity;

		if (!p_io.needsReloadHooks()) {
			//will only work on SDK 11+!!
			setPreserveEGLContextOnPause(true);
		}
		mInputManager = InputManagerCompat.Factory.getInputManager(this.getContext());
		mInputManager.registerInputDeviceListener(this, null);
		init(false, 16, 0);
    }

    public GodotView(Context context, boolean translucent, int depth, int stencil) {
		super(context);
		init(translucent, depth, stencil);
    }

	@Override public boolean onTouchEvent (MotionEvent event) {

		return activity.gotTouchEvent(event);
	};

	public int get_godot_button(int keyCode) {

		int button = 0;
		switch (keyCode) {
			case KeyEvent.KEYCODE_BUTTON_A: // Android A is SNES B
				button = 0;
				break;
			case KeyEvent.KEYCODE_BUTTON_B:
				button = 1;
				break;
			case KeyEvent.KEYCODE_BUTTON_X: // Android X is SNES Y
				button = 2;
				break;
			case KeyEvent.KEYCODE_BUTTON_Y:
				button = 3;
				break;
			case KeyEvent.KEYCODE_BUTTON_L1:
				button = 9;
				break;
			case KeyEvent.KEYCODE_BUTTON_L2:
				button = 15;
				break;
			case KeyEvent.KEYCODE_BUTTON_R1:
				button = 10;
				break;
			case KeyEvent.KEYCODE_BUTTON_R2:
				button = 16;
				break;
			case KeyEvent.KEYCODE_BUTTON_SELECT:
				button = 4;
				break;
			case KeyEvent.KEYCODE_BUTTON_START:
				button = 6;
				break;
			case KeyEvent.KEYCODE_BUTTON_THUMBL:
				button = 7;
				break;
			case KeyEvent.KEYCODE_BUTTON_THUMBR:
				button = 8;
				break;
			case KeyEvent.KEYCODE_DPAD_UP:
				button = 11;
				break;
			case KeyEvent.KEYCODE_DPAD_DOWN:
				button = 12;
				break;
			case KeyEvent.KEYCODE_DPAD_LEFT:
				button = 13;
				break;
			case KeyEvent.KEYCODE_DPAD_RIGHT:
				button = 14;
				break;
			case KeyEvent.KEYCODE_BUTTON_C:
				button = 17;
				break;
			case KeyEvent.KEYCODE_BUTTON_Z:
				button = 18;
				break;

			default:
				button = keyCode - KeyEvent.KEYCODE_BUTTON_1 + 20;
				break;
		};
		return button;
	};

	private static class joystick {
		public int device_id;
		public String name;
		public ArrayList<InputDevice.MotionRange> axes;
		public ArrayList<InputDevice.MotionRange> hats;
	}

	private static class RangeComparator implements Comparator<InputDevice.MotionRange> {
		@Override
		public int compare(InputDevice.MotionRange arg0, InputDevice.MotionRange arg1) {
			return arg0.getAxis() - arg1.getAxis();
		}
	}

	ArrayList<joystick> joy_devices = new ArrayList<joystick>();

	private int find_joy_device(int device_id) {
		for (int i=0; i<joy_devices.size(); i++) {
			if (joy_devices.get(i).device_id == device_id) {
					return i;
			}
		}
		onInputDeviceAdded(device_id);
		return joy_devices.size() - 1;
	}

	@Override public void onInputDeviceAdded(int deviceId) {
		joystick joy = new joystick();
		joy.device_id = deviceId;
		final int id = joy_devices.size();
		InputDevice device = mInputManager.getInputDevice(deviceId);
		final String name = device.getName();
		joy.name = device.getName();
		joy.axes = new ArrayList<InputDevice.MotionRange>();
		joy.hats = new ArrayList<InputDevice.MotionRange>();
		List<InputDevice.MotionRange> ranges = device.getMotionRanges();
		Collections.sort(ranges, new RangeComparator());
		for (InputDevice.MotionRange range : ranges) {
			if (range.getAxis() == MotionEvent.AXIS_HAT_X || range.getAxis() == MotionEvent.AXIS_HAT_Y) {
				joy.hats.add(range);
			}
			else {
				joy.axes.add(range);
			}
		}
		joy_devices.add(joy);
		queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.joyconnectionchanged(id, true, name);
			}
		});
  }

	@Override public void onInputDeviceRemoved(int deviceId) {
		final int id = find_joy_device(deviceId);
		joy_devices.remove(id);
		queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.joyconnectionchanged(id, false, "");
			}
		});
	}

	@Override public void onInputDeviceChanged(int deviceId) {

	}
	@Override public boolean onKeyUp(final int keyCode, KeyEvent event) {

		if (keyCode == KeyEvent.KEYCODE_BACK) {
			return true;
		}

		if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
			return super.onKeyUp(keyCode, event);
		};

		int source = event.getSource();
		if ((source & InputDevice.SOURCE_JOYSTICK) != 0 || (source & InputDevice.SOURCE_DPAD) != 0 || (source & InputDevice.SOURCE_GAMEPAD) != 0) {

			final int button = get_godot_button(keyCode);
			final int device = find_joy_device(event.getDeviceId());

			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.joybutton(device, button, false);
				}
			});
			return true;
		} else {
			final int chr = event.getUnicodeChar(0);
			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.key(keyCode, chr, false);
				}
			});
		};
		return super.onKeyUp(keyCode, event);
	};

	@Override public boolean onKeyDown(final int keyCode, KeyEvent event) {

		if (keyCode == KeyEvent.KEYCODE_BACK) {
			activity.onBackPressed();
			// press 'back' button should not terminate program
			//normal handle 'back' event in game logic
			return true;
		}

		if (keyCode == KeyEvent.KEYCODE_VOLUME_UP || keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
			return super.onKeyDown(keyCode, event);
		};

		int source = event.getSource();
		//Log.e(TAG, String.format("Key down! source %d, device %d, joystick %d, %d, %d", event.getDeviceId(), source, (source & InputDevice.SOURCE_JOYSTICK), (source & InputDevice.SOURCE_DPAD), (source & InputDevice.SOURCE_GAMEPAD)));

		if ((source & InputDevice.SOURCE_JOYSTICK) != 0 || (source & InputDevice.SOURCE_DPAD) != 0 || (source & InputDevice.SOURCE_GAMEPAD) != 0) {

			if (event.getRepeatCount() > 0) // ignore key echo
				return true;
			final int button = get_godot_button(keyCode);
			final int device = find_joy_device(event.getDeviceId());

			//Log.e(TAG, String.format("joy button down! button %x, %d, device %d", keyCode, button, device));
			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.joybutton(device, button, true);
				}
			});
			return true;

		} else {
			final int chr = event.getUnicodeChar(0);
			queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.key(keyCode, chr, true);
				}
			});
		};
		return super.onKeyDown(keyCode, event);
	}

	@Override public boolean onGenericMotionEvent(MotionEvent event) {

		if ((event.getSource() & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK && event.getAction() == MotionEvent.ACTION_MOVE) {

			final int device_id = find_joy_device(event.getDeviceId());
			joystick joy = joy_devices.get(device_id);

			for (int i = 0; i < joy.axes.size(); i++) {
				InputDevice.MotionRange range = joy.axes.get(i);
				final float value = (event.getAxisValue(range.getAxis()) - range.getMin() ) / range.getRange() * 2.0f - 1.0f;
				//Log.e(TAG, String.format("axis event: %d, value %f", i, value));
				final int idx = i;
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.joyaxis(device_id, idx, value);
					}
				});
			}

			for (int i = 0; i < joy.hats.size(); i+=2) {
				final int hatX = Math.round(event.getAxisValue(joy.hats.get(i).getAxis()));
				final int hatY = Math.round(event.getAxisValue(joy.hats.get(i+1).getAxis()));
				//Log.e(TAG, String.format("HAT EVENT %d, %d", hatX, hatY));
				queueEvent(new Runnable() {
					@Override
					public void run() {
						GodotLib.joyhat(device_id, hatX, hatY);
					}
				});
			}
			return true;
		};

		return super.onGenericMotionEvent(event);
	};


    private void init(boolean translucent, int depth, int stencil) {

		this.setFocusableInTouchMode(true);
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
		setEGLContextFactory(new ContextFactory());

		/* We need to choose an EGLConfig that matches the format of
		 * our surface exactly. This is going to be done in our
		 * custom config chooser. See ConfigChooser class definition
		 * below.
		 */

		if (use_32) {
			setEGLConfigChooser( translucent ?
						new FallbackConfigChooser(8, 8, 8, 8, 24, stencil, new ConfigChooser(8, 8, 8, 8, 16, stencil)) :
						new FallbackConfigChooser(8, 8, 8, 8, 24, stencil, new ConfigChooser(5, 6, 5, 0, 16, stencil)) );

		} else {
			setEGLConfigChooser( translucent ?
						new ConfigChooser(8, 8, 8, 8, 16, stencil) :
						new ConfigChooser(5, 6, 5, 0, 16, stencil) );
		}

		/* Set the renderer responsible for frame rendering */
		setRenderer(new Renderer());
	}

	private static class ContextFactory implements GLSurfaceView.EGLContextFactory {
	private static int EGL_CONTEXT_CLIENT_VERSION = 0x3098;
	public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig eglConfig) {
		if (use_gl3)
			Log.w(TAG, "creating OpenGL ES 3.0 context :");
		else
			Log.w(TAG, "creating OpenGL ES 2.0 context :");

		checkEglError("Before eglCreateContext", egl);
		int[] attrib_list2 = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL10.EGL_NONE };
		int[] attrib_list3 = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL10.EGL_NONE };
		EGLContext context = egl.eglCreateContext(display, eglConfig, EGL10.EGL_NO_CONTEXT, use_gl3?attrib_list3:attrib_list2);
		checkEglError("After eglCreateContext", egl);
		return context;
	}

	public void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context) {
	    egl.eglDestroyContext(display, context);
	}
    }

    private static void checkEglError(String prompt, EGL10 egl) {
	int error;
	while ((error = egl.eglGetError()) != EGL10.EGL_SUCCESS) {
	    Log.e(TAG, String.format("%s: EGL error: 0x%x", prompt, error));
	}
    }
    	/* Fallback if 32bit View is not supported*/
	private static class FallbackConfigChooser extends ConfigChooser {
		private ConfigChooser fallback;

		public FallbackConfigChooser(int r, int g, int b, int a, int depth, int stencil, ConfigChooser fallback) {
			super(r, g, b, a, depth, stencil);
			this.fallback = fallback;
		}

      		@Override
		public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display, EGLConfig[] configs) {
			EGLConfig ec = super.chooseConfig(egl, display, configs);
			if (ec == null) {
	  			Log.w(TAG, "Trying ConfigChooser fallback");
	  			ec = fallback.chooseConfig(egl, display, configs);
				use_32=false;
			}
			return ec;
      		}
    	}

	private static class ConfigChooser implements GLSurfaceView.EGLConfigChooser {

		public ConfigChooser(int r, int g, int b, int a, int depth, int stencil) {
			mRedSize = r;
			mGreenSize = g;
			mBlueSize = b;
			mAlphaSize = a;
			mDepthSize = depth;
			mStencilSize = stencil;
		}

		/* This EGL config specification is used to specify 2.0 rendering.
		 * We use a minimum size of 4 bits for red/green/blue, but will
		 * perform actual matching in chooseConfig() below.
		 */
		private static int EGL_OPENGL_ES2_BIT = 4;
		private static int[] s_configAttribs2 =
		{
			EGL10.EGL_RED_SIZE, 4,
			EGL10.EGL_GREEN_SIZE, 4,
			EGL10.EGL_BLUE_SIZE, 4,
		  //  EGL10.EGL_DEPTH_SIZE,     16,
		   // EGL10.EGL_STENCIL_SIZE,   EGL10.EGL_DONT_CARE,
			EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
			EGL10.EGL_NONE
		};
		private static int[] s_configAttribs3 =
		{
			EGL10.EGL_RED_SIZE, 4,
			EGL10.EGL_GREEN_SIZE, 4,
			EGL10.EGL_BLUE_SIZE, 4,
		   // EGL10.EGL_DEPTH_SIZE,     16,
		  //  EGL10.EGL_STENCIL_SIZE,   EGL10.EGL_DONT_CARE,
			EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, //apparently there is no EGL_OPENGL_ES3_BIT
			EGL10.EGL_NONE
		};

		public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display) {

			/* Get the number of minimally matching EGL configurations
			 */
			int[] num_config = new int[1];
			egl.eglChooseConfig(display, use_gl3?s_configAttribs3:s_configAttribs2, null, 0, num_config);

			int numConfigs = num_config[0];

			if (numConfigs <= 0) {
			throw new IllegalArgumentException("No configs match configSpec");
			}

			/* Allocate then read the array of minimally matching EGL configs
			 */
			EGLConfig[] configs = new EGLConfig[numConfigs];
			egl.eglChooseConfig(display, use_gl3?s_configAttribs3:s_configAttribs2, configs, numConfigs, num_config);

			if (DEBUG) {
			 printConfigs(egl, display, configs);
			}
			/* Now return the "best" one
			 */
			return chooseConfig(egl, display, configs);
		}

		public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display,
			EGLConfig[] configs) {
			for(EGLConfig config : configs) {
			int d = findConfigAttrib(egl, display, config,
				EGL10.EGL_DEPTH_SIZE, 0);
			int s = findConfigAttrib(egl, display, config,
				EGL10.EGL_STENCIL_SIZE, 0);

			// We need at least mDepthSize and mStencilSize bits
			if (d < mDepthSize || s < mStencilSize)
				continue;

			// We want an *exact* match for red/green/blue/alpha
			int r = findConfigAttrib(egl, display, config,
				EGL10.EGL_RED_SIZE, 0);
			int g = findConfigAttrib(egl, display, config,
					EGL10.EGL_GREEN_SIZE, 0);
			int b = findConfigAttrib(egl, display, config,
					EGL10.EGL_BLUE_SIZE, 0);
			int a = findConfigAttrib(egl, display, config,
				EGL10.EGL_ALPHA_SIZE, 0);

			if (r == mRedSize && g == mGreenSize && b == mBlueSize && a == mAlphaSize)
				return config;
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

		private void printConfigs(EGL10 egl, EGLDisplay display,
			EGLConfig[] configs) {
			int numConfigs = configs.length;
			Log.w(TAG, String.format("%d configurations", numConfigs));
			for (int i = 0; i < numConfigs; i++) {
			Log.w(TAG, String.format("Configuration %d:\n", i));
			printConfig(egl, display, configs[i]);
			}
		}

		private void printConfig(EGL10 egl, EGLDisplay display,
			EGLConfig config) {
			int[] attributes = {
				EGL10.EGL_BUFFER_SIZE,
				EGL10.EGL_ALPHA_SIZE,
				EGL10.EGL_BLUE_SIZE,
				EGL10.EGL_GREEN_SIZE,
				EGL10.EGL_RED_SIZE,
				EGL10.EGL_DEPTH_SIZE,
				EGL10.EGL_STENCIL_SIZE,
				EGL10.EGL_CONFIG_CAVEAT,
				EGL10.EGL_CONFIG_ID,
				EGL10.EGL_LEVEL,
				EGL10.EGL_MAX_PBUFFER_HEIGHT,
				EGL10.EGL_MAX_PBUFFER_PIXELS,
				EGL10.EGL_MAX_PBUFFER_WIDTH,
				EGL10.EGL_NATIVE_RENDERABLE,
				EGL10.EGL_NATIVE_VISUAL_ID,
				EGL10.EGL_NATIVE_VISUAL_TYPE,
				0x3030, // EGL10.EGL_PRESERVED_RESOURCES,
				EGL10.EGL_SAMPLES,
				EGL10.EGL_SAMPLE_BUFFERS,
				EGL10.EGL_SURFACE_TYPE,
				EGL10.EGL_TRANSPARENT_TYPE,
				EGL10.EGL_TRANSPARENT_RED_VALUE,
				EGL10.EGL_TRANSPARENT_GREEN_VALUE,
				EGL10.EGL_TRANSPARENT_BLUE_VALUE,
				0x3039, // EGL10.EGL_BIND_TO_TEXTURE_RGB,
				0x303A, // EGL10.EGL_BIND_TO_TEXTURE_RGBA,
				0x303B, // EGL10.EGL_MIN_SWAP_INTERVAL,
				0x303C, // EGL10.EGL_MAX_SWAP_INTERVAL,
				EGL10.EGL_LUMINANCE_SIZE,
				EGL10.EGL_ALPHA_MASK_SIZE,
				EGL10.EGL_COLOR_BUFFER_TYPE,
				EGL10.EGL_RENDERABLE_TYPE,
				0x3042 // EGL10.EGL_CONFORMANT
			};
			String[] names = {
				"EGL_BUFFER_SIZE",
				"EGL_ALPHA_SIZE",
				"EGL_BLUE_SIZE",
				"EGL_GREEN_SIZE",
				"EGL_RED_SIZE",
				"EGL_DEPTH_SIZE",
				"EGL_STENCIL_SIZE",
				"EGL_CONFIG_CAVEAT",
				"EGL_CONFIG_ID",
				"EGL_LEVEL",
				"EGL_MAX_PBUFFER_HEIGHT",
				"EGL_MAX_PBUFFER_PIXELS",
				"EGL_MAX_PBUFFER_WIDTH",
				"EGL_NATIVE_RENDERABLE",
				"EGL_NATIVE_VISUAL_ID",
				"EGL_NATIVE_VISUAL_TYPE",
				"EGL_PRESERVED_RESOURCES",
				"EGL_SAMPLES",
				"EGL_SAMPLE_BUFFERS",
				"EGL_SURFACE_TYPE",
				"EGL_TRANSPARENT_TYPE",
				"EGL_TRANSPARENT_RED_VALUE",
				"EGL_TRANSPARENT_GREEN_VALUE",
				"EGL_TRANSPARENT_BLUE_VALUE",
				"EGL_BIND_TO_TEXTURE_RGB",
				"EGL_BIND_TO_TEXTURE_RGBA",
				"EGL_MIN_SWAP_INTERVAL",
				"EGL_MAX_SWAP_INTERVAL",
				"EGL_LUMINANCE_SIZE",
				"EGL_ALPHA_MASK_SIZE",
				"EGL_COLOR_BUFFER_TYPE",
				"EGL_RENDERABLE_TYPE",
				"EGL_CONFORMANT"
			};
			int[] value = new int[1];
			for (int i = 0; i < attributes.length; i++) {
			int attribute = attributes[i];
			String name = names[i];
			if ( egl.eglGetConfigAttrib(display, config, attribute, value)) {
				Log.w(TAG, String.format("  %s: %d\n", name, value[0]));
			} else {
				// Log.w(TAG, String.format("  %s: failed\n", name));
				while (egl.eglGetError() != EGL10.EGL_SUCCESS);
			}
			}
		}

		// Subclasses can adjust these values:
		protected int mRedSize;
		protected int mGreenSize;
		protected int mBlueSize;
		protected int mAlphaSize;
		protected int mDepthSize;
		protected int mStencilSize;
		private int[] mValue = new int[1];
	}

	private static class Renderer implements GLSurfaceView.Renderer {


		public void onDrawFrame(GL10 gl) {
			GodotLib.step();
			for(int i=0;i<Godot.singleton_count;i++) {
				Godot.singletons[i].onGLDrawFrame(gl);
			}
		}

		public void onSurfaceChanged(GL10 gl, int width, int height) {

			GodotLib.resize(width, height,!firsttime);
			firsttime=false;
			for(int i=0;i<Godot.singleton_count;i++) {
				Godot.singletons[i].onGLSurfaceChanged(gl, width, height);
			}
		}

		public void onSurfaceCreated(GL10 gl, EGLConfig config) {
			GodotLib.newcontext(use_32);
		}
	}
}
