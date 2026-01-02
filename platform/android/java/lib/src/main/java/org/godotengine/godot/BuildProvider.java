/**************************************************************************/
/*  BuildProvider.java                                                    */
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

import org.godotengine.godot.variant.Callable;

import androidx.annotation.NonNull;

/**
 * Provides an environment for executing build commands.
 */
public interface BuildProvider {
	/**
	 * Connects to the build environment.
	 *
	 * @param callback The callback to call when connected
	 * @return Whether or not connecting is possible
	 */
	boolean buildEnvConnect(@NonNull Callable callback);

	/**
	 * Disconnects from the build environment.
	 */
	void buildEnvDisconnect();

	/**
	 * Executes a command via the build environment.
	 *
	 * @param buildTool      The build tool to execute (for example, "gradle")
	 * @param arguments      The argument for the command
	 * @param projectPath    The working directory to use when executing the command
	 * @param buildDir       The build directory within the project
	 * @param outputCallback The callback to call for each line of output from the command
	 * @param resultCallback The callback to call when the command is finished running
	 * @return A positive job id, if successful; otherwise, a negative number
	 */
	int buildEnvExecute(String buildTool, @NonNull String[] arguments, @NonNull String projectPath, @NonNull String buildDir, @NonNull Callable outputCallback, @NonNull Callable resultCallback);

	/**
	 * Cancels a command executed via the build environment.
	 *
	 * @param jobId The job id returned from buildEnvExecute()
	 */
	void buildEnvCancel(int jobId);

	/**
	 * Requests that a project be cleaned up via the build environment.
	 *
	 * @param projectPath The working directory to use when executing the command
	 * @param buildDir    The build directory within the project
	 */
	void buildEnvCleanProject(@NonNull String projectPath, @NonNull String buildDir, @NonNull Callable callback);
}
