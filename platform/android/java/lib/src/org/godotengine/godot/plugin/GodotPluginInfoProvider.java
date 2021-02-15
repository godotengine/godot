/*************************************************************************/
/*  GodotPluginInfoProvider.java                                         */
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

package org.godotengine.godot.plugin;

import androidx.annotation.NonNull;

import java.util.List;
import java.util.Set;

/**
 * Provides the set of information expected from a Godot plugin.
 */
public interface GodotPluginInfoProvider {

	/**
	 * Returns the name of the plugin.
	 */
	@NonNull
	String getPluginName();

	/**
	 * Returns the list of methods to be exposed to Godot.
	 */
	@NonNull
	List<String> getPluginMethods();

	/**
	 * Returns the list of signals to be exposed to Godot.
	 */
	@NonNull
	Set<SignalInfo> getPluginSignals();

	/**
	 * Returns the paths for the plugin's gdnative libraries (if any).
	 *
	 * The paths must be relative to the 'assets' directory and point to a '*.gdnlib' file.
	 */
	@NonNull
	Set<String> getPluginGDNativeLibrariesPaths();
}
