/**************************************************************************/
/*  modding_security.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                          QUAIL MODDING SDK                             */
/**************************************************************************/
/* Security module for safe modding - restricts dangerous operations     */
/**************************************************************************/

#pragma once

#include "core/string/ustring.h"

class ModdingSecurity {
public:
	// Check if a file path is allowed for reading
	static bool is_path_allowed_for_read(const String &p_path) {
		// Only allow reading from user://mods/ directory
		if (p_path.begins_with("user://mods/")) {
			return true;
		}

		// Block all other paths
		return false;
	}

	// Check if ANY write operation is allowed (always false for modding security)
	static bool is_write_allowed() {
		return false; // NO file writing allowed for mods
	}

	// Check if directory operations are allowed (always false)
	static bool is_directory_operation_allowed() {
		return false; // NO directory operations allowed
	}

	// Check if OS commands are allowed (always false)
	static bool is_os_command_allowed() {
		return false; // NO OS commands allowed
	}

	// Check if network operations are allowed (always false)
	static bool is_network_allowed() {
		return false; // NO network access allowed
	}

	// Check if native code loading is allowed (always false)
	static bool is_native_code_loading_allowed() {
		return false; // NO native code loading allowed
	}
};
