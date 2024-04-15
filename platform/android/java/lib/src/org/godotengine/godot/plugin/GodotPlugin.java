/**************************************************************************/
/*  GodotPlugin.java                                                      */
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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Base class for Godot Android plugins.
 * <p>
 * A Godot Android plugin is an Android library with the following requirements:
 * <p>
 * - The plugin must have a dependency on the Godot Android library: `implementation "org.godotengine:godot:<godotLibVersion>"`
 * <p>
 * - The plugin must include a <meta-data> tag in its Android manifest with the following format:
 * <meta-data android:name="org.godotengine.plugin.v2.[PluginName]" android:value="[plugin.init.ClassFullName]" />
 * <p>
 * Where:
 * <p>
 * - 'PluginName' is the name of the plugin.
 * <p>
 * - 'plugin.init.ClassFullName' is the full name (package + class name) of the plugin init class
 * extending {@link GodotPlugin}.
 * <p>
 * A Godot Android plugin can also define and provide c/c++ gdextension libraries, which will be
 * automatically bundled by the aar build system.
 * GDExtension ('*.gdextension') config files must be located in the project 'assets' directory and
 * their paths specified by {@link GodotPlugin#getPluginGDExtensionLibrariesPaths()}.
 *
 * @see <a href="https://docs.godotengine.org/en/stable/tutorials/platform/android/index.html">Android plugins</a>
 */
public abstract class GodotPlugin {
	private static final String TAG = GodotPlugin.class.getSimpleName();

	private final Godot godot;
	private final ConcurrentHashMap<String, SignalInfo> registeredSignals = new ConcurrentHashMap<>();

	/**
	 * Base constructor passing a {@link Godot} instance through which the plugin can access Godot's
	 * APIs and lifecycle events.
	 */
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
	 * Provides access to the hosting {@link Activity}.
	 */
	@Nullable
	protected Activity getActivity() {
		return godot.getActivity();
	}

	/**
	 * Register the plugin with Godot native code.
	 * <p>
	 * This method is invoked by the Godot Engine on the render thread.
	 */
	public final void onRegisterPluginWithGodotNative() {
		registeredSignals.putAll(
				registerPluginWithGodotNative(this, getPluginName(), getPluginMethods(), getPluginSignals()));
	}

	private static Map<String, SignalInfo> registerPluginWithGodotNative(Object pluginObject,
			String pluginName, List<String> pluginMethods, Set<SignalInfo> pluginSignals) {
		nativeRegisterSingleton(pluginName, pluginObject);

		Set<Method> filteredMethods = new HashSet<>();
		Class<?> clazz = pluginObject.getClass();

		Method[] methods = clazz.getDeclaredMethods();
		for (Method method : methods) {
			// Check if the method is annotated with {@link UsedByGodot}.
			if (method.getAnnotation(UsedByGodot.class) != null) {
				filteredMethods.add(method);
			} else {
				// For backward compatibility, process the methods from the given <pluginMethods> argument.
				for (String methodName : pluginMethods) {
					if (methodName.equals(method.getName())) {
						filteredMethods.add(method);
						break;
					}
				}
			}
		}

		for (Method method : filteredMethods) {
			List<String> ptr = new ArrayList<>();

			Class<?>[] paramTypes = method.getParameterTypes();
			for (Class<?> c : paramTypes) {
				ptr.add(c.getName());
			}

			String[] pt = new String[ptr.size()];
			ptr.toArray(pt);

			nativeRegisterMethod(pluginName, method.getName(), method.getReturnType().getName(), pt);
		}

		// Register the signals for this plugin.
		Map<String, SignalInfo> registeredSignals = new HashMap<>();
		for (SignalInfo signalInfo : pluginSignals) {
			String signalName = signalInfo.getName();
			nativeRegisterSignal(pluginName, signalName, signalInfo.getParamTypesNames());
			registeredSignals.put(signalName, signalInfo);
		}

		return registeredSignals;
	}

	/**
	 * Invoked once during the initialization process after creation of the
	 * {@link org.godotengine.godot.GodotRenderView} view.
	 * <p>
	 * The plugin can return a non-null {@link View} layout which will be added to the Godot view
	 * hierarchy.
	 * <p>
	 * Use {@link GodotPlugin#shouldBeOnTop()} to specify whether the plugin's {@link View} should
	 * be added on top or behind the main Godot view.
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
	 * Invoked on the render thread when set up of the Godot engine is complete.
	 * <p>
	 * This is invoked before {@link GodotPlugin#onGodotMainLoopStarted()}.
	 */
	public void onGodotSetupCompleted() {}

	/**
	 * Invoked on the render thread when the Godot main loop has started.
	 *
	 * This is invoked after {@link GodotPlugin#onGodotSetupCompleted()}.
	 */
	public void onGodotMainLoopStarted() {}

	/**
	 * When using the OpenGL renderer, this is invoked once per frame on the GL thread after the
	 * frame is drawn.
	 */
	public void onGLDrawFrame(GL10 gl) {}

	/**
	 * When using the OpenGL renderer, this is called on the GL thread after the surface is created
	 * and whenever the OpenGL ES surface size changes.
	 */
	public void onGLSurfaceChanged(GL10 gl, int width, int height) {}

	/**
	 * When using the OpenGL renderer, this is called on the GL thread when the surface is created
	 * or recreated.
	 */
	public void onGLSurfaceCreated(GL10 gl, EGLConfig config) {}

	/**
	 * When using the Vulkan renderer, this is invoked once per frame on the Vulkan thread after
	 * the frame is drawn.
	 */
	public void onVkDrawFrame() {}

	/**
	 * When using the Vulkan renderer, this is called on the Vulkan thread after the surface is
	 * created and whenever the surface size changes.
	 */
	public void onVkSurfaceChanged(Surface surface, int width, int height) {}

	/**
	 * When using the Vulkan renderer, this is called on the Vulkan thread when the surface is
	 * created or recreated.
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
	 *
	 * @deprecated Use the {@link UsedByGodot} annotation instead.
	 */
	@NonNull
	@Deprecated
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
	 * Returns the paths for the plugin's gdextension libraries (if any).
	 * <p>
	 * Each returned path must be relative to the 'assets' directory and point to a '*.gdextension' file.
	 */
	@NonNull
	public Set<String> getPluginGDExtensionLibrariesPaths() {
		return Collections.emptySet();
	}

	/**
	 * Returns whether the plugin's {@link View} returned in
	 * {@link GodotPlugin#onMainCreate(Activity)} should be placed on top of the main Godot view.
	 * <p>
	 * Returning false causes the plugin's {@link View} to be placed behind, which can be useful
	 * when used with transparency in order to let the Godot view handle inputs.
	 */
	public boolean shouldBeOnTop() {
		return true;
	}

	/**
	 * Returns whether the plugin supports the given feature tag.
	 *
	 * @see <a href="https://docs.godotengine.org/en/stable/tutorials/export/feature_tags.html">Feature tags</a>
	 */
	public boolean supportsFeature(String featureTag) {
		return false;
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
	 * @param signalName Name of the signal to emit. It will be validated against the set of registered signals.
	 * @param signalArgs Arguments used to populate the emitted signal. The arguments will be validated against the {@link SignalInfo} matching the registered signalName parameter.
	 */
	protected void emitSignal(final String signalName, final Object... signalArgs) {
		try {
			// Check that the given signal is among the registered set.
			SignalInfo signalInfo = registeredSignals.get(signalName);
			if (signalInfo == null) {
				throw new IllegalArgumentException(
						"Signal " + signalName + " is not registered for this plugin.");
			}
			emitSignal(getGodot(), getPluginName(), signalInfo, signalArgs);
		} catch (IllegalArgumentException exception) {
			Log.w(TAG, exception);
			if (BuildConfig.DEBUG) {
				throw exception;
			}
		}
	}

	/**
	 * Emit a Godot signal.
	 * @param godot Godot instance
	 * @param pluginName Name of the Godot plugin the signal will be emitted from. The plugin must already be registered with the Godot engine.
	 * @param signalInfo Information about the signal to emit.
	 * @param signalArgs Arguments used to populate the emitted signal. The arguments will be validated against the given {@link SignalInfo} parameter.
	 */
	public static void emitSignal(Godot godot, String pluginName, SignalInfo signalInfo, final Object... signalArgs) {
		try {
			if (signalInfo == null) {
				throw new IllegalArgumentException("Signal must be non null.");
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

			godot.runOnRenderThread(() -> nativeEmitSignal(pluginName, signalInfo.getName(), signalArgs));

		} catch (IllegalArgumentException exception) {
			Log.w(TAG, exception);
			if (BuildConfig.DEBUG) {
				throw exception;
			}
		}
	}

	/**
	 * Used to setup a {@link GodotPlugin} instance.
	 * @param p_name Name of the instance.
	 */
	private static native void nativeRegisterSingleton(String p_name, Object object);

	/**
	 * Used to complete registration of the {@link GodotPlugin} instance's methods.
	 * @param p_sname Name of the instance
	 * @param p_name Name of the method to register
	 * @param p_ret Return type of the registered method
	 * @param p_params Method parameters types
	 */
	private static native void nativeRegisterMethod(String p_sname, String p_name, String p_ret, String[] p_params);

	/**
	 * Used to complete registration of the {@link GodotPlugin} instance's signals.
	 * @param pluginName Name of the plugin
	 * @param signalName Name of the signal to register
	 * @param signalParamTypes Signal parameters types
	 */
	private static native void nativeRegisterSignal(String pluginName, String signalName, String[] signalParamTypes);

	/**
	 * Used to emit signal by {@link GodotPlugin} instance.
	 * @param pluginName Name of the plugin
	 * @param signalName Name of the signal to emit
	 * @param signalParams Signal parameters
	 */
	private static native void nativeEmitSignal(String pluginName, String signalName, Object[] signalParams);
}
