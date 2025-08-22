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

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "sandbox_project_settings.h"
//#define ENABLE_TIMINGS 1
#ifdef ENABLE_TIMINGS
#include <time.h>
#endif

static constexpr bool VERBOSE_CMD = true;

static bool ContainerIsAlreadyRunning(String container_name) {
	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "container", "inspect", "-f", "{{.State.Running}}", container_name };

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if constexpr (VERBOSE_CMD) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output;
	int exit_code;
	Error error = OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output, &exit_code);
	if (error != OK || exit_code != 0) {
		return false;
	}
	return output.contains("true");
}

bool Docker::ContainerPullLatest(String image_name, Array &output) {
	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "pull", image_name };

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if constexpr (VERBOSE_CMD) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output_str;
	int exit_code;
	Error error = OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output_str, &exit_code);

	// Convert string output back to Array for compatibility
	output.clear();
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

	return (error == OK && exit_code == 0);
}

String Docker::ContainerGetMountPath(String container_name) {
	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "inspect", "-f", "{{ (index .Mounts 0).Source }}", container_name };

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if constexpr (VERBOSE_CMD) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output;
	int exit_code;
	Error error = OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output, &exit_code);
	if (error != OK || exit_code != 0) {
		return "";
	}
	return output.replace("\n", "");
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
			print_line("Container mount path (" + path + ") does not match the current project path (" + project_path + "). Stopping the container.");
			Docker::ContainerStop(container_name);
		} else {
			// The container is already running and the mount path matches the current project path.
			print_line("Container " + container_name + " was already running.");
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
	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "run", "--name", container_name, "-dv", ".:/usr/src", image_name };

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if constexpr (VERBOSE_CMD) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output_str;
	int exit_code;
	Error error = OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output_str, &exit_code);

	// Convert string output back to Array for compatibility
	output.clear();
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

	return (error == OK && exit_code == 0);
}

Array Docker::ContainerStop(String container_name) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return Array();
	}
	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "stop", container_name, "--time", "0" };

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if constexpr (VERBOSE_CMD) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output_str;
	int exit_code;
	OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output_str, &exit_code);

	// Convert string output back to Array for compatibility
	Array output;
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

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

	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "exec", "-t", container_name, "bash" };
	for (int i = 0; i < p_arguments.size(); i++) {
		arguments.push_back(p_arguments[i]);
	}

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if (VERBOSE_CMD && verbose) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output_str;
	int exit_code;
	Error error = OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output_str, &exit_code);

	// Convert string output back to Array for compatibility
	output.clear();
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

#ifdef ENABLE_TIMINGS
	timespec end;
	clock_gettime(CLOCK_MONOTONIC, &end);
	const double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	fprintf(stderr, "Docker::ContainerExecute: %f seconds\n", elapsed);
#endif

	return (error == OK && exit_code == 0);
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
	::OS *OS = ::OS::get_singleton();
	PackedStringArray arguments = { "rm", container_name };

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	if constexpr (VERBOSE_CMD) {
		String args_str = "";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line("Docker: " + SandboxProjectSettings::get_docker_path() + " " + args_str);
	}

	String output_str;
	int exit_code;
	Error error = OS->execute(SandboxProjectSettings::get_docker_path(), args_list, &output_str, &exit_code);

	// Convert string output back to Array for compatibility
	output.clear();
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

	return (error == OK && exit_code == 0);
}
