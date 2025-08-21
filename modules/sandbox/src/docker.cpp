/**************************************************************************/
/*  docker.cpp                                                            */
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

#include "docker.h"

#include "sandbox_project_settings.h"
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/project_settings.hpp>
//#define ENABLE_TIMINGS 1
#ifdef ENABLE_TIMINGS
#include <time.h>
#endif

static constexpr bool VERBOSE_CMD = true;
using namespace godot;

static bool ContainerIsAlreadyRunning(String container_name) {
	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "container", "inspect", "-f", "{{.State.Running}}", container_name };
	Array output;
	if constexpr (VERBOSE_CMD) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	const int res = OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);
	if (res != 0) {
		return false;
	}
	const String running = output[0];
	return running.contains("true");
}

bool Docker::ContainerPullLatest(String image_name, Array &output) {
	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "pull", image_name };
	if constexpr (VERBOSE_CMD) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	const int res = OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);
	return res == 0;
}

String Docker::ContainerGetMountPath(String container_name) {
	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "inspect", "-f", "{{ (index .Mounts 0).Source }}", container_name };
	Array output;
	if constexpr (VERBOSE_CMD) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	const int res = OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);
	if (res != 0) {
		return "";
	}
	return String(output[0]).replace("\n", "");
}

bool Docker::ContainerStart(String container_name, String image_name, Array &output) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return true;
	}
	if (ContainerIsAlreadyRunning(container_name)) {
		ProjectSettings *project_settings = ProjectSettings::get_singleton();
		// If the container mount path does not match the current project path, stop the container.
		String path = ContainerGetMountPath(container_name);
		String project_path = project_settings->globalize_path("res://");
		//printf("Container mount path: %s\n", path.utf8().get_data());
		//printf("Current project path: %s\n", project_path.utf8().get_data());
		if (!path.is_empty() && !project_path.begins_with(path)) {
			UtilityFunctions::print("Container mount path (", path, ") does not match the current project path (", project_path, "). Stopping the container.");
			Docker::ContainerStop(container_name);
		} else {
			// The container is already running and the mount path matches the current project path.
			UtilityFunctions::print("Container ", container_name, " was already running.");
			return true;
		}
	}
	// The container is not running. Try to pull the latest image.
	Array dont_care; // We don't care about the output of the image pull (for now).
	if (ContainerPullLatest(image_name, dont_care)) {
		// Delete the container if it exists. It's not running, but it might be stopped.
		ContainerDelete(container_name, dont_care);
	} else {
		WARN_PRINT("Sandbox: Failed to pull the latest container image: " + image_name);
	}
	// Start the container, even if the image pull failed. It might be locally available.
	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "run", "--name", container_name, "-dv", ".:/usr/src", image_name };
	if constexpr (VERBOSE_CMD) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	const int res = OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);
	return res == 0;
}

Array Docker::ContainerStop(String container_name) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return Array();
	}
	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "stop", container_name, "--time", "0" };
	if constexpr (VERBOSE_CMD) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	Array output;
	OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);
	return output;
}

bool Docker::ContainerExecute(String container_name, const PackedStringArray &p_arguments, Array &output, bool verbose) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return false;
	}
#ifdef ENABLE_TIMINGS
	timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif

	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "exec", "-t", container_name, "bash" };
	for (int i = 0; i < p_arguments.size(); i++) {
		arguments.push_back(p_arguments[i]);
	}
	if (VERBOSE_CMD && verbose) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	const int res = OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);

#ifdef ENABLE_TIMINGS
	timespec end;
	clock_gettime(CLOCK_MONOTONIC, &end);
	const double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	fprintf(stderr, "Docker::ContainerExecute: %f seconds\n", elapsed);
#endif

	return res == 0;
}

int Docker::ContainerVersion(String container_name, const PackedStringArray &p_arguments) {
	// Execute --version in the container.
	Array output;
	if (ContainerExecute(container_name, p_arguments, output)) {
		// Docker container responds with a number, eg "1" (ASCII)
		return output[0].operator String().to_int();
	}
	return -1;
}

bool Docker::ContainerDelete(String container_name, Array &output) {
	godot::OS *OS = godot::OS::get_singleton();
	PackedStringArray arguments = { "rm", container_name };
	if constexpr (VERBOSE_CMD) {
		UtilityFunctions::print(SandboxProjectSettings::get_docker_path(), arguments);
	}
	const int res = OS->execute(SandboxProjectSettings::get_docker_path(), arguments, output);
	return res == 0;
}
