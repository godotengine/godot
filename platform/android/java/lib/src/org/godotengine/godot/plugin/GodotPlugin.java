/*************************************************************************/
/*  GodotPlugin.java                                                     */
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

package org.godotengine.godot.plugin;

import org.godotengine.godot.BuildConfig;
import org.godotengine.godot.Godot;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Base class for the Godot Android plugins.
 * <p>
 * A Godot Android plugin is a regular Android library packaged as an aar archive file with the following caveats:
 * <p>
 * - The library must have a dependency on the Godot Android library (godot-lib.aar).
 * A stable version is available for each release.
 * <p>
 * - The library must include a <meta-data> tag in its manifest file setup as follow:
 * <meta-data android:name="org.godotengine.plugin.v1.[PluginName]" android:value="[plugin.init.ClassFullName]" />
 * Where:
 * - 'PluginName' is the name of the plugin.
 * - 'plugin.init.ClassFullName' is the full name (package + class name) of the plugin class
 * extending {@link GodotPlugin}.
 *
 * A plugin can also define and provide c/c++ gdnative libraries and nativescripts for the target
 * app/game to leverage.
 * The shared library for the gdnative library will be automatically bundled by the aar build
 * system.
 * Godot '*.gdnlib' and '*.gdns' resource files must however be manually defined in the project
 * 'assets' directory. The recommended path for these resources in the 'assets' directory should be:
 * 'godot/plugin/v1/[PluginName]/'
 */
public abstract class GodotPlugin {
	private static final String TAG = GodotPlugin.class.getSimpleName();

	private final Godot godot;
	private final ConcurrentHashMap<String, SignalInfo> registeredSignals = new ConcurrentHashMap<>();

	public GodotPlugin(Godot godot) {
		this.godot = godot;
	}

	/**
	 * Provides access to the Godot engine.
	 */
	protected Godot getGodot() {
		return godot;
	}

	/**
	 * Provides access to the underlying {@link Activity}.
	 */
	@Nullable
	protected Activity getActivity() {
		return godot.getActivity();
	}

	/**
	 * Register the plugin with Godot native code.
	 *
	 * This method is invoked on the render thread.
	 */
	public final void onRegisterPluginWithGodotNative() {
		nativeRegisterSingleton(getPluginName());

		Class clazz = getClass();
		Method[] methods = clazz.getDeclaredMethods();
		for (Method method : methods) {
			boolean found = false;

			for (String s : getPluginMethods()) {
				if (s.equals(method.getName())) {
					found = true;
					break;
				}
			}
			if (!found)
				continue;

			List<String> ptr = new ArrayList<String>();

			Class[] paramTypes = method.getParameterTypes();
			for (Class c : paramTypes) {
				ptr.add(c.getName());
			}

			String[] pt = new String[ptr.size()];
			ptr.toArray(pt);

			nativeRegisterMethod(getPluginName(), method.getName(), method.getReturnType().getName(), pt);
		}

		// Register the signals for this plugin.
		for (SignalInfo signalInfo : getPluginSignals()) {
			String signalName = signalInfo.getName();
			nativeRegisterSignal(getPluginName(), signalName, signalInfo.getParamTypesNames());
			registeredSignals.put(signalName, signalInfo);
		}

		// Get the list of gdnative libraries to register.
		Set<String> gdnativeLibrariesPaths = getPluginGDNativeLibrariesPaths();
		if (!gdnativeLibrariesPaths.isEmpty()) {
			nativeRegisterGDNativeLibraries(gdnativeLibrariesPaths.toArray(new String[0]));
		}
	}

	/**
	 * Invoked once during the Godot Android initialization process after creation of the
	 * {@link org.godotengine.godot.GodotView} view.
	 * <p>
	 * The plugin can return a non-null {@link View} layout in order to add it to the Godot view
	 * hierarchy.
	 *
	 * @see Activity#onCreate(Bundle)
	 * @return the plugin's view to be included; null if no views should be included.
	 */
	@Nullable
	public View onMainCreate(Activity activity) {
		return null;
	}

	/**
	 * @see Activity#onActivityResult(int, int, Intent)
	 */
	public void onMainActivityResult(int requestCode, int resultCode, Intent data) {
	}

	/**
	 * @see Activity#onRequestPermissionsResult(int, String[], int[])
	 */
	public void onMainRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
	}

	/**
	 * @see Activity#onPause()
	 */
	public void onMainPause() {}

	/**
	 * @see Activity#onResume()
	 */
	public void onMainResume() {}

	/**
	 * @see Activity#onDestroy()
	 */
	public void onMainDestroy() {}

	/**
	 * @see Activity#onBackPressed()
	 */
	public boolean onMainBackPressed() { return false; }

	/**
	 * Invoked on the render thread when the Godot main loop has started.
	 */
	public void onGodotMainLoopStarted() {}

	/**
	 * Invoked once per frame on the GL thread after the frame is drawn.
	 */
	public void onGLDrawFrame(GL10 gl) {}

	/**
	 * Called on the GL thread after the surface is created and whenever the OpenGL ES surface size
	 * changes.
	 */
	public void onGLSurfaceChanged(GL10 gl, int width, int height) {}

	/**
	 * Called on the GL thread when the surface is created or recreated.
	 */
	public void onGLSurfaceCreated(GL10 gl, EGLConfig config) {}

	/**
	 * Invoked once per frame on the Vulkan thread after the frame is drawn.
	 */
	public void onVkDrawFrame() {}

	/**
	 * Called on the Vulkan thread after the surface is created and whenever the surface size
	 * changes.
	 */
	public void onVkSurfaceChanged(Surface surface, int width, int height) {}

	/**
	 * Called on the Vulkan thread when the surface is created or recreated.
	 */
	public void onVkSurfaceCreated(Surface surface) {}

	/**
	 * Returns the name of the plugin.
	 * <p>
	 * This value must match the one listed in the plugin '<meta-data>' manifest entry.
	 */
	@NonNull
	public abstract String getPluginName();

	/**
	 * Returns the list of methods to be exposed to Godot.
	 */
	@NonNull
	public List<String> getPluginMethods() {
		return Collections.emptyList();
	}

	/**
	 * Returns the list of signals to be exposed to Godot.
	 */
	@NonNull
	public Set<SignalInfo> getPluginSignals() {
		return Collections.emptySet();
	}

	/**
	 * Returns the paths for the plugin's gdnative libraries.
	 *
	 * The paths must be relative to the 'assets' directory and point to a '*.gdnlib' file.
	 */
	@NonNull
	protected Set<String> getPluginGDNativeLibrariesPaths() {
		return Collections.emptySet();
	}

	/**
	 * Runs the specified action on the UI thread. If the current thread is the UI
	 * thread, then the action is executed immediately. If the current thread is
	 * not the UI thread, the action is posted to the event queue of the UI thread.
	 *
	 * @param action the action to run on the UI thread
	 */
	protected void runOnUiThread(Runnable action) {
		godot.runOnUiThread(action);
	}

	/**
	 * Queue the specified action to be run on the render thread.
	 *
	 * @param action the action to run on the render thread
	 */
	protected void runOnRenderThread(Runnable action) {
		godot.runOnRenderThread(action);
	}

	/**
	 * Emit a registered Godot signal.
	 * @param signalName
	 * @param signalArgs
	 */
	protected void emitSignal(final String signalName, final Object... signalArgs) {
		try {
			// Check that the given signal is among the registered set.
			SignalInfo signalInfo = registeredSignals.get(signalName);
			if (signalInfo == null) {
				throw new IllegalArgumentException(
						"Signal " + signalName + " is not registered for this plugin.");
			}

			// Validate the arguments count.
			Class<?>[] signalParamTypes = signalInfo.getParamTypes();
			if (signalArgs.length != signalParamTypes.length) {
				throw new IllegalArgumentException(
						"Invalid arguments count. Should be " + signalParamTypes.length + "  but is " + signalArgs.length);
			}

			// Validate the argument's types.
			for (int i = 0; i < signalParamTypes.length; i++) {
				if (!signalParamTypes[i].isInstance(signalArgs[i])) {
					throw new IllegalArgumentException(
							"Invalid type for argument #" + i + ". Should be of type " + signalParamTypes[i].getName());
				}
			}

			runOnRenderThread(new Runnable() {
				@Override
				public void run() {
					nativeEmitSignal(getPluginName(), signalName, signalArgs);
				}
			});
		} catch (IllegalArgumentException exception) {
			Log.w(TAG, exception.getMessage());
			if (BuildConfig.DEBUG) {
				throw exception;
			}
		}
	}

	/**
	 * Used to setup a {@link GodotPlugin} instance.
	 * @param p_name Name of the instance.
	 */
	private native void nativeRegisterSingleton(String p_name);

	/**
	 * Used to complete registration of the {@link GodotPlugin} instance's methods.
	 * @param p_sname Name of the instance
	 * @param p_name Name of the method to register
	 * @param p_ret Return type of the registered method
	 * @param p_params Method parameters types
	 */
	private native void nativeRegisterMethod(String p_sname, String p_name, String p_ret, String[] p_params);

	/**
	 * Used to register gdnative libraries bundled by the plugin.
	 * @param gdnlibPaths Paths to the libraries relative to the 'assets' directory.
	 */
	private native void nativeRegisterGDNativeLibraries(String[] gdnlibPaths);

	/**
	 * Used to complete registration of the {@link GodotPlugin} instance's methods.
	 * @param pluginName Name of the plugin
	 * @param signalName Name of the signal to register
	 * @param signalParamTypes Signal parameters types
	 */
	private native void nativeRegisterSignal(String pluginName, String signalName, String[] signalParamTypes);

	/**
	 * Used to emit signal by {@link GodotPlugin} instance.
	 * @param pluginName Name of the plugin
	 * @param signalName Name of the signal to emit
	 * @param signalParams Signal parameters
	 */
	private native void nativeEmitSignal(String pluginName, String signalName, Object[] signalParams);
}
