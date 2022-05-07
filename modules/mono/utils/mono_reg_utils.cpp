/*************************************************************************/
/*  mono_reg_utils.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "mono_reg_utils.h"
#include "core/os/dir_access.h"

#ifdef WINDOWS_ENABLED

#include "core/os/os.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace MonoRegUtils {

template <int>
REGSAM bitness_sam_impl();

template <>
REGSAM bitness_sam_impl<4>() {
	return KEY_WOW64_64KEY;
}

template <>
REGSAM bitness_sam_impl<8>() {
	return KEY_WOW64_32KEY;
}

REGSAM _get_bitness_sam() {
	return bitness_sam_impl<sizeof(size_t)>();
}

LONG _RegOpenKey(HKEY hKey, LPCWSTR lpSubKey, PHKEY phkResult) {
	LONG res = RegOpenKeyExW(hKey, lpSubKey, 0, KEY_READ, phkResult);

	if (res != ERROR_SUCCESS)
		res = RegOpenKeyExW(hKey, lpSubKey, 0, KEY_READ | _get_bitness_sam(), phkResult);

	return res;
}

LONG _RegKeyQueryString(HKEY hKey, const String &p_value_name, String &r_value) {
	Vector<WCHAR> buffer;
	buffer.resize(512);
	DWORD dwBufferSize = buffer.size();

	LONG res = RegQueryValueExW(hKey, p_value_name.c_str(), 0, NULL, (LPBYTE)buffer.ptr(), &dwBufferSize);

	if (res == ERROR_MORE_DATA) {
		// dwBufferSize now contains the actual size
		buffer.resize(dwBufferSize);
		res = RegQueryValueExW(hKey, p_value_name.c_str(), 0, NULL, (LPBYTE)buffer.ptr(), &dwBufferSize);
	}

	if (res == ERROR_SUCCESS) {
		r_value = String(buffer.ptr(), buffer.size());
	} else {
		r_value = String();
	}

	return res;
}

LONG _find_mono_in_reg(const String &p_subkey, MonoRegInfo &r_info, bool p_old_reg = false) {
	HKEY hKey;
	LONG res = _RegOpenKey(HKEY_LOCAL_MACHINE, p_subkey.c_str(), &hKey);

	if (res != ERROR_SUCCESS)
		goto cleanup;

	if (!p_old_reg) {
		res = _RegKeyQueryString(hKey, "Version", r_info.version);
		if (res != ERROR_SUCCESS)
			goto cleanup;
	}

	res = _RegKeyQueryString(hKey, "SdkInstallRoot", r_info.install_root_dir);
	if (res != ERROR_SUCCESS)
		goto cleanup;

	res = _RegKeyQueryString(hKey, "FrameworkAssemblyDirectory", r_info.assembly_dir);
	if (res != ERROR_SUCCESS)
		goto cleanup;

	res = _RegKeyQueryString(hKey, "MonoConfigDir", r_info.config_dir);
	if (res != ERROR_SUCCESS)
		goto cleanup;

	if (r_info.install_root_dir.ends_with("\\"))
		r_info.bin_dir = r_info.install_root_dir + "bin";
	else
		r_info.bin_dir = r_info.install_root_dir + "\\bin";

cleanup:
	RegCloseKey(hKey);
	return res;
}

LONG _find_mono_in_reg_old(const String &p_subkey, MonoRegInfo &r_info) {
	String default_clr;

	HKEY hKey;
	LONG res = _RegOpenKey(HKEY_LOCAL_MACHINE, p_subkey.c_str(), &hKey);

	if (res != ERROR_SUCCESS)
		goto cleanup;

	res = _RegKeyQueryString(hKey, "DefaultCLR", default_clr);

	if (res == ERROR_SUCCESS && default_clr.length()) {
		r_info.version = default_clr;
		res = _find_mono_in_reg(p_subkey + "\\" + default_clr, r_info, true);
	}

cleanup:
	RegCloseKey(hKey);
	return res;
}

MonoRegInfo find_mono() {
	MonoRegInfo info;

	if (_find_mono_in_reg("Software\\Mono", info) == ERROR_SUCCESS)
		return info;

	if (_find_mono_in_reg_old("Software\\Novell\\Mono", info) == ERROR_SUCCESS)
		return info;

	return MonoRegInfo();
}

String find_msbuild_tools_path() {
	String msbuild_tools_path;

	// Try to find 15.0 with vswhere

	String vswhere_path = OS::get_singleton()->get_environment(sizeof(size_t) == 8 ? "ProgramFiles(x86)" : "ProgramFiles");
	vswhere_path += "\\Microsoft Visual Studio\\Installer\\vswhere.exe";

	List<String> vswhere_args;
	vswhere_args.push_back("-latest");
	vswhere_args.push_back("-products");
	vswhere_args.push_back("*");
	vswhere_args.push_back("-requires");
	vswhere_args.push_back("Microsoft.Component.MSBuild");

	String output;
	int exit_code;
	OS::get_singleton()->execute(vswhere_path, vswhere_args, true, NULL, &output, &exit_code);

	if (exit_code == 0) {
		Vector<String> lines = output.split("\n");

		for (int i = 0; i < lines.size(); i++) {
			const String &line = lines[i];
			int sep_idx = line.find(":");

			if (sep_idx > 0) {
				String key = line.substr(0, sep_idx); // No need to trim

				if (key == "installationPath") {
					String val = line.substr(sep_idx + 1, line.length()).strip_edges();

					ERR_BREAK(val.empty());

					if (!val.ends_with("\\")) {
						val += "\\";
					}

					// Since VS2019, the directory is simply named "Current"
					String msbuild_dir = val + "MSBuild\\Current\\Bin";
					if (DirAccess::exists(msbuild_dir)) {
						return msbuild_dir;
					}

					// Directory name "15.0" is used in VS 2017
					return val + "MSBuild\\15.0\\Bin";
				}
			}
		}
	}

	// Try to find 14.0 in the Registry

	HKEY hKey;
	LONG res = _RegOpenKey(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\MSBuild\\ToolsVersions\\14.0", &hKey);

	if (res != ERROR_SUCCESS)
		goto cleanup;

	res = _RegKeyQueryString(hKey, "MSBuildToolsPath", msbuild_tools_path);

	if (res != ERROR_SUCCESS)
		goto cleanup;

cleanup:
	RegCloseKey(hKey);

	return msbuild_tools_path;
}
} // namespace MonoRegUtils

#endif // WINDOWS_ENABLED
