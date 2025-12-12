/**************************************************************************/
/*  openxr_select_runtime.cpp                                             */
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

#include "openxr_select_runtime.h"

#ifdef WINDOWS_ENABLED
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/os/os.h"
#include "editor/settings/editor_settings.h"

constexpr char GENERIC_PREFIX[] = "Unknown OpenXR Runtime";

void OpenXRSelectRuntime::_update_items() {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	OS *os = OS::get_singleton();
	Dictionary runtimes = EDITOR_GET("xr/openxr/runtime_paths");

	int current_runtime = 0;
	int index = 0;
	String current_path = os->get_environment("XR_RUNTIME_JSON");

	// Parse the user's home folder.
	String home_folder = os->get_environment("HOME");
	if (home_folder.is_empty()) {
		home_folder = os->get_environment("HOMEDRIVE") + os->get_environment("HOMEPATH");
	}

	clear();
	add_item("Default", -1);
	set_item_metadata(index, "");
	index++;

	for (const KeyValue<Variant, Variant> &kv : runtimes) {
		const String &key = kv.key;
		const String &path = kv.value;
		String adj_path = path.replace("~", home_folder);

		if (da->file_exists(adj_path)) {
			add_item(key, index);
			set_item_metadata(index, adj_path);

			if (current_path == adj_path) {
				current_runtime = index;
			}
			index++;
		}
	}

	select(current_runtime);
}

void OpenXRSelectRuntime::_on_item_selected(int p_which) {
	OS *os = OS::get_singleton();

	if (p_which == 0) {
		// Return to default runtime
		os->set_environment("XR_RUNTIME_JSON", "");
	} else {
		// Select the runtime we want
		String runtime_path = get_item_metadata(p_which);
		os->set_environment("XR_RUNTIME_JSON", runtime_path);
	}
}

void OpenXRSelectRuntime::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
			// Update dropdown
			_update_items();

			// Connect signal
			connect(SceneStringName(item_selected), callable_mp(this, &OpenXRSelectRuntime::_on_item_selected));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			// Disconnect signal
			disconnect(SceneStringName(item_selected), callable_mp(this, &OpenXRSelectRuntime::_on_item_selected));
		} break;
	}
}

String OpenXRSelectRuntime::_try_and_get_runtime_name(const String &p_config_file) {
	if constexpr (!GD_IS_CLASS_ENABLED(JSON)) {
		return "";
	}

	// Attempt to get a valid runtime name from the json file
	String file_contents = FileAccess::get_file_as_string(p_config_file);
	Dictionary root_node = JSON::parse_string(file_contents);
	if (!root_node.has("runtime")) {
		return "";
	}
	Dictionary api_layer = root_node["runtime"];
	if (!api_layer.has("name") || api_layer["name"].get_type() != Variant::STRING) {
		return "";
	}
	return api_layer["name"];
}

void OpenXRSelectRuntime::_add_runtime(Dictionary &r_runtimes, const String &p_config_file) {
	if (r_runtimes.values().has(p_config_file)) {
		// config file already in the list of runtimes, do not add a duplicate
		return;
	}

	String runtime_name = _try_and_get_runtime_name(p_config_file);
	if (runtime_name.is_empty()) {
		runtime_name = GENERIC_PREFIX;
	}

	if (r_runtimes.keys().has(runtime_name)) {
		// Highly unlikely, performance is not critical
		unsigned int index = 1;
		while (r_runtimes.keys().has(runtime_name + " " + uitos(index))) {
			index++;
		}
		runtime_name = runtime_name + " " + uitos(index);
	}
	r_runtimes[runtime_name] = p_config_file;
}

Dictionary OpenXRSelectRuntime::_enumerate_runtimes() {
	Dictionary default_runtimes;

#if defined(WINDOWS_ENABLED)
	// Add known common runtimes in case they are not populated in registry
	default_runtimes["Meta"] = "C:\\Program Files\\Oculus\\Support\\oculus-runtime\\oculus_openxr_64.json";
	default_runtimes["SteamVR"] = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR\\steamxr_win64.json";
	default_runtimes["Varjo"] = "C:\\Program Files\\Varjo\\varjo-openxr\\VarjoOpenXR.json";
	default_runtimes["WMR"] = "C:\\WINDOWS\\system32\\MixedRealityRuntime.json";

	// Hard code openxr version 1.
	LPCWSTR runtimes_key = L"SOFTWARE\\Khronos\\OpenXR\\1\\AvailableRuntimes";
	HKEY hkey = nullptr;
	LSTATUS result = RegOpenKeyExW(HKEY_LOCAL_MACHINE, runtimes_key, 0, KEY_READ | KEY_QUERY_VALUE, &hkey);
	if (result != ERROR_SUCCESS) {
		return default_runtimes;
	}

	DWORD max_value_len, value_count;
	result = RegQueryInfoKeyW(
			hkey, // hKey
			nullptr, // lpClass
			nullptr, // lpcchClass
			nullptr, // lpReserved
			nullptr, // lpcSubKeys
			nullptr, // lpcbMaxSubKeyLen
			nullptr, // lpcbMaxClassLen
			&value_count, // lpcValues
			&max_value_len, // lpcbMaxValueNameLen
			nullptr, // lpcbMaxValueLen
			nullptr, // lpcbSecurityDescriptor
			nullptr // lpftLastWriteTime
	);
	if (result != ERROR_SUCCESS) {
		RegCloseKey(hkey);
		return default_runtimes;
	}

	Char16String value_name;
	value_name.resize_uninitialized(max_value_len + 1);
	DWORD value_len, value_type;

	for (DWORD i = 0; i < value_count; i++) {
		value_len = max_value_len + 1;
		result = RegEnumValueW(
				hkey, // hKey
				i, // dwIndex
				(LPWSTR)value_name.get_data(), // lpValueName
				&value_len, // lpcchValueName
				nullptr, // lpReserved
				&value_type, // lpType
				nullptr, // lpData
				nullptr // lpcbData
		);
		if (result != ERROR_SUCCESS || value_type != REG_DWORD) {
			continue;
		}

		_add_runtime(default_runtimes, String::utf16((const char16_t *)value_name.get_data()));
	}

	// Cleanup, close the key we opened
	RegCloseKey(hkey);

#elif defined(LINUXBSD_ENABLED)
	default_runtimes["Monado"] = "/usr/share/openxr/1/openxr_monado.json";
	default_runtimes["SteamVR"] = "~/.steam/steam/steamapps/common/SteamVR/steamxr_linux64.json";
#endif
	return default_runtimes;
}

OpenXRSelectRuntime::OpenXRSelectRuntime() {
	// TODO: Move to editor_settings.cpp
	EDITOR_DEF_RST("xr/openxr/runtime_paths", _enumerate_runtimes());

	set_flat(true);
	set_theme_type_variation("TopBarOptionButton");
	set_fit_to_longest_item(false);
	set_focus_mode(Control::FOCUS_NONE);
	set_tooltip_text(TTR("Choose an XR runtime."));
}
