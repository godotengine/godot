/**************************************************************************/
/*  GodotHost.java                                                        */
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

import org.godotengine.godot.plugin.GodotPlugin;

import android.app.Activity;

import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Denotate a component (e.g: Activity, Fragment) that hosts the {@link Godot} engine.
 */
public interface GodotHost {
	/**
	 * Provides a set of command line parameters to setup the {@link Godot} engine.
	 */
	default List<String> getCommandLine() {
		return Collections.emptyList();
	}

	/**
	 * Invoked on the render thread when setup of the {@link Godot} engine is complete.
	 */
	default void onGodotSetupCompleted() {}

	/**
	 * Invoked on the render thread when the {@link Godot} engine main loop has started.
	 */
	default void onGodotMainLoopStarted() {}

	/**
	 * Invoked on the render thread to terminate the given {@link Godot} engine instance.
	 */
	default void onGodotForceQuit(Godot instance) {}

	/**
	 * Invoked on the render thread to terminate the {@link Godot} engine instance with the given id.
	 * @param godotInstanceId id of the Godot instance to terminate. See {@code onNewGodotInstanceRequested}
	 *
	 * @return true if successful, false otherwise.
	 */
	default boolean onGodotForceQuit(int godotInstanceId) {
		return false;
	}

	/**
	 * Invoked on the render thread when the Godot instance wants to be restarted. It's up to the host
	 * to perform the appropriate action(s).
	 */
	default void onGodotRestartRequested(Godot instance) {}

	/**
	 * Invoked on the render thread when a new Godot instance is requested. It's up to the host to
	 * perform the appropriate action(s).
	 *
	 * @param args Arguments used to initialize the new instance.
	 *
	 * @return the id of the new instance. See {@code onGodotForceQuit}
	 */
	default int onNewGodotInstanceRequested(String[] args) {
		return 0;
	}

	/**
	 * Provide access to the Activity hosting the {@link Godot} engine.
	 */
	Activity getActivity();

	/**
	 * Provide access to the hosted {@link Godot} engine.
	 */
	Godot getGodot();

	/**
	 * Returns a set of {@link GodotPlugin} to be registered with the hosted {@link Godot} engine.
	 */
	default Set<GodotPlugin> getHostPlugins(Godot engine) {
		return Collections.emptySet();
	}
}
