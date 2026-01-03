/**************************************************************************/
/*  os_windows.cpp                                                        */
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

#include "os_windows.h"

#include "display_server_windows.h"
#include "lang_table.h"
#include "windows_terminal_logger.h"
#include "windows_utils.h"

#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/io/marshalls.h"
#include "core/os/main_loop.h"
#include "core/profiling/profiling.h"
#include "core/version_generated.gen.h"
#include "drivers/windows/dir_access_windows.h"
#include "drivers/windows/file_access_windows.h"
#include "drivers/windows/file_access_windows_pipe.h"
#include "drivers/windows/ip_windows.h"
#include "drivers/windows/net_socket_winsock.h"
#include "drivers/windows/thread_windows.h"
#include "main/main.h"
#include "servers/audio/audio_server.h"
#include "servers/rendering/rendering_server_default.h"
#include "servers/text/text_server.h"

#include <avrt.h>
#include <bcrypt.h>
#include <direct.h>
#include <hidsdi.h>
#include <knownfolders.h>
#include <process.h>
#include <psapi.h>
#include <regstr.h>
#include <shlobj.h>
#include <wbemcli.h>
#include <wincrypt.h>
#include <winternl.h>

#if defined(RD_ENABLED)
#include "servers/rendering/rendering_device.h"
#endif

#if defined(GLES3_ENABLED)
#include "gl_manager_windows_native.h"
#endif

#if defined(VULKAN_ENABLED)
#include "rendering_context_driver_vulkan_windows.h"
#endif
#if defined(D3D12_ENABLED)
#include "drivers/d3d12/rendering_context_driver_d3d12.h"
#endif
#if defined(GLES3_ENABLED)
#include "drivers/gles3/rasterizer_gles3.h"
#endif

#ifdef DEBUG_ENABLED
#pragma pack(push, before_imagehlp, 8)
#include <imagehlp.h>
#pragma pack(pop, before_imagehlp)
#endif

extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
__declspec(dllexport) void NoHotPatch() {} // Disable Nahimic code injection.
}

// Workaround mingw-w64 < 4.0 bug
#ifndef WM_TOUCH
#define WM_TOUCH 576
#endif

#ifndef WM_POINTERUPDATE
#define WM_POINTERUPDATE 0x0245
#endif

// Missing in MinGW headers before 8.0.
#ifndef DWRITE_FONT_WEIGHT_SEMI_LIGHT
#define DWRITE_FONT_WEIGHT_SEMI_LIGHT (DWRITE_FONT_WEIGHT)350
#endif

static String fix_path(const String &p_path) {
	String path = p_path;
	if (p_path.is_relative_path()) {
		Char16String current_dir_name;
		size_t str_len = GetCurrentDirectoryW(0, nullptr);
		current_dir_name.resize_uninitialized(str_len + 1);
		GetCurrentDirectoryW(current_dir_name.size(), (LPWSTR)current_dir_name.ptrw());
		path = String::utf16((const char16_t *)current_dir_name.get_data()).trim_prefix(R"(\\?\)").replace_char('\\', '/').path_join(path);
	}
	path = path.simplify_path();
	path = path.replace_char('/', '\\');
	if (path.size() >= MAX_PATH && !path.is_network_share_path() && !path.begins_with(R"(\\?\)")) {
		path = R"(\\?\)" + path;
	}
	return path;
}

static String format_error_message(DWORD id) {
	LPWSTR messageBuffer = nullptr;
	size_t size = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&messageBuffer, 0, nullptr);

	String msg = "Error " + itos(id) + ": " + String::utf16((const char16_t *)messageBuffer, size);

	LocalFree(messageBuffer);

	return msg.remove_chars("\r\n");
}

void RedirectStream(const char *p_file_name, const char *p_mode, FILE *p_cpp_stream, const DWORD p_std_handle) {
	const HANDLE h_existing = GetStdHandle(p_std_handle);
	if (h_existing != INVALID_HANDLE_VALUE) { // Redirect only if attached console have a valid handle.
		const HANDLE h_cpp = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(p_cpp_stream)));
		if (h_cpp == INVALID_HANDLE_VALUE) { // Redirect only if it's not already redirected to the pipe or file.
			FILE *fp = p_cpp_stream;
			freopen_s(&fp, p_file_name, p_mode, p_cpp_stream); // Redirect stream.
			setvbuf(p_cpp_stream, nullptr, _IONBF, 0); // Disable stream buffering.
		}
	}
}

void RedirectIOToConsole() {
	// Save current handles.
	HANDLE h_stdin = GetStdHandle(STD_INPUT_HANDLE);
	HANDLE h_stdout = GetStdHandle(STD_OUTPUT_HANDLE);
	HANDLE h_stderr = GetStdHandle(STD_ERROR_HANDLE);

	if (AttachConsole(ATTACH_PARENT_PROCESS)) {
		// Restore redirection (Note: if not redirected it's NULL handles not INVALID_HANDLE_VALUE).
		if (h_stdin != nullptr) {
			SetStdHandle(STD_INPUT_HANDLE, h_stdin);
		}
		if (h_stdout != nullptr) {
			SetStdHandle(STD_OUTPUT_HANDLE, h_stdout);
		}
		if (h_stderr != nullptr) {
			SetStdHandle(STD_ERROR_HANDLE, h_stderr);
		}

		// Update file handles.
		RedirectStream("CONIN$", "r", stdin, STD_INPUT_HANDLE);
		RedirectStream("CONOUT$", "w", stdout, STD_OUTPUT_HANDLE);
		RedirectStream("CONOUT$", "w", stderr, STD_ERROR_HANDLE);
	}
}

bool OS_Windows::is_using_con_wrapper() const {
	static String exe_renames[] = {
		".console.exe",
		"_console.exe",
		" console.exe",
		"console.exe",
		String(),
	};

	bool found_exe = false;
	bool found_conwrap_exe = false;
	String exe_name = get_executable_path().to_lower();
	String exe_dir = exe_name.get_base_dir();
	String exe_fname = exe_name.get_file().get_basename();

	DWORD pids[256];
	DWORD count = GetConsoleProcessList(&pids[0], 256);
	for (DWORD i = 0; i < count; i++) {
		HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pids[i]);
		if (process != NULL) {
			WCHAR proc_name[MAX_PATH];
			DWORD len = MAX_PATH;
			if (QueryFullProcessImageNameW(process, 0, &proc_name[0], &len)) {
				String name = String::utf16((const char16_t *)&proc_name[0], len).replace_char('\\', '/').to_lower();
				if (name == exe_name) {
					found_exe = true;
				}
				for (int j = 0; !exe_renames[j].is_empty(); j++) {
					if (name == exe_dir.path_join(exe_fname + exe_renames[j])) {
						found_conwrap_exe = true;
					}
				}
			}
			CloseHandle(process);
			if (found_conwrap_exe && found_exe) {
				break;
			}
		}
	}
	if (!found_exe) {
		return true; // Unable to read console info, assume true.
	}

	return found_conwrap_exe;
}

BOOL WINAPI HandlerRoutine(_In_ DWORD dwCtrlType) {
	if (!EngineDebugger::is_active()) {
		return FALSE;
	}

	switch (dwCtrlType) {
		case CTRL_C_EVENT:
			EngineDebugger::get_script_debugger()->set_depth(-1);
			EngineDebugger::get_script_debugger()->set_lines_left(1);
			return TRUE;
		default:
			return FALSE;
	}
}

void OS_Windows::alert(const String &p_alert, const String &p_title) {
	MessageBoxW(nullptr, (LPCWSTR)(p_alert.utf16().get_data()), (LPCWSTR)(p_title.utf16().get_data()), MB_OK | MB_ICONEXCLAMATION | MB_TASKMODAL);
}

void OS_Windows::initialize_debugging() {
	SetConsoleCtrlHandler(HandlerRoutine, TRUE);
}

#ifdef WINDOWS_DEBUG_OUTPUT_ENABLED
static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type) {
	String err_str;
	if (p_errorexp && p_errorexp[0]) {
		err_str = String::utf8(p_errorexp) + "\n";
	} else {
		err_str = String::utf8(p_file) + ":" + itos(p_line) + " - " + String::utf8(p_error) + "\n";
	}

	OutputDebugStringW((LPCWSTR)err_str.utf16().ptr());
}
#endif

void OS_Windows::initialize() {
	crash_handler.initialize();

#ifdef WINDOWS_DEBUG_OUTPUT_ENABLED
	error_handlers.errfunc = _error_handler;
	error_handlers.userdata = this;
	add_error_handler(&error_handlers);
#endif

#ifdef THREADS_ENABLED
	init_thread_win();
#endif

	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_FILESYSTEM);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_EDITOR_RESOURCES);
	FileAccess::make_default<FileAccessWindowsPipe>(FileAccess::ACCESS_PIPE);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_FILESYSTEM);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_EDITOR_RESOURCES);

	NetSocketWinSock::make_default();

	// We need to know how often the clock is updated
	QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_second);
	QueryPerformanceCounter((LARGE_INTEGER *)&ticks_start);

#if WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP
	// set minimum resolution for periodic timers, otherwise Sleep(n) may wait at least as
	//  long as the windows scheduler resolution (~16-30ms) even for calls like Sleep(1)
	TIMECAPS time_caps;
	if (timeGetDevCaps(&time_caps, sizeof(time_caps)) == MMSYSERR_NOERROR) {
		delay_resolution = time_caps.wPeriodMin * 1000;
		timeBeginPeriod(time_caps.wPeriodMin);
	} else {
		ERR_PRINT("Unable to detect sleep timer resolution.");
		delay_resolution = 1000;
		timeBeginPeriod(1);
	}
#else
	delay_resolution = 1000;
#endif

	process_map = memnew((HashMap<ProcessID, ProcessInfo>));

	// Add current Godot PID to the list of known PIDs
	ProcessInfo current_pi = {};
	PROCESS_INFORMATION current_pi_pi = {};
	current_pi.pi = current_pi_pi;
	current_pi.pi.hProcess = GetCurrentProcess();
	process_map->insert(GetCurrentProcessId(), current_pi);

	IPWindows::make_default();
	main_loop = nullptr;

	HRESULT hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown **>(&dwrite_factory));
	if (SUCCEEDED(hr)) {
		hr = dwrite_factory->GetSystemFontCollection(&font_collection, false);
		if (SUCCEEDED(hr)) {
			dwrite_init = true;
			hr = dwrite_factory->QueryInterface(&dwrite_factory2);
			if (SUCCEEDED(hr)) {
				hr = dwrite_factory2->GetSystemFontFallback(&system_font_fallback);
				if (SUCCEEDED(hr)) {
					dwrite2_init = true;
				}
			}
		}
	}
	if (!dwrite_init) {
		print_verbose("Unable to load IDWriteFactory, system font support is disabled.");
	} else if (!dwrite2_init) {
		print_verbose("Unable to load IDWriteFactory2, automatic system font fallback is disabled.");
	}

	FileAccessWindows::initialize();
}

void OS_Windows::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}
	main_loop = nullptr;
}

void OS_Windows::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

void OS_Windows::finalize() {
	if (dwrite_factory2) {
		dwrite_factory2->Release();
		dwrite_factory2 = nullptr;
	}
	if (font_collection) {
		font_collection->Release();
		font_collection = nullptr;
	}
	if (system_font_fallback) {
		system_font_fallback->Release();
		system_font_fallback = nullptr;
	}
	if (dwrite_factory) {
		dwrite_factory->Release();
		dwrite_factory = nullptr;
	}
#ifdef WINMIDI_ENABLED
	driver_midi.close();
#endif

	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = nullptr;
}

void OS_Windows::finalize_core() {
	while (!temp_libraries.is_empty()) {
		_remove_temp_library(temp_libraries.last()->key);
	}

	FileAccessWindows::finalize();

#if WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP
	timeEndPeriod(1);
#endif

	memdelete(process_map);
	NetSocketWinSock::cleanup();

#ifdef WINDOWS_DEBUG_OUTPUT_ENABLED
	remove_error_handler(&error_handlers);
#endif
}

Error OS_Windows::get_entropy(uint8_t *r_buffer, int p_bytes) {
	NTSTATUS status = BCryptGenRandom(nullptr, r_buffer, p_bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
	ERR_FAIL_COND_V(status, FAILED);
	return OK;
}

#ifdef DEBUG_ENABLED
void debug_dynamic_library_check_dependencies(const String &p_path, HashSet<String> &r_checked, HashSet<String> &r_missing) {
	if (r_checked.has(p_path)) {
		return;
	}
	r_checked.insert(p_path);

	LOADED_IMAGE loaded_image;
	HANDLE file = CreateFileW((LPCWSTR)p_path.utf16().get_data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
	if (file != INVALID_HANDLE_VALUE) {
		HANDLE file_mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY | SEC_COMMIT, 0, 0, nullptr);
		if (file_mapping != INVALID_HANDLE_VALUE) {
			PVOID mapping = MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, 0);
			if (mapping) {
				PIMAGE_DOS_HEADER dos_header = (PIMAGE_DOS_HEADER)mapping;
				PIMAGE_NT_HEADERS nt_header = nullptr;
				if (dos_header->e_magic == IMAGE_DOS_SIGNATURE) {
					PCHAR nt_header_ptr;
					nt_header_ptr = ((PCHAR)mapping) + dos_header->e_lfanew;
					nt_header = (PIMAGE_NT_HEADERS)nt_header_ptr;
					if (nt_header->Signature != IMAGE_NT_SIGNATURE) {
						nt_header = nullptr;
					}
				}
				if (nt_header) {
					loaded_image.ModuleName = nullptr;
					loaded_image.hFile = file;
					loaded_image.MappedAddress = (PUCHAR)mapping;
					loaded_image.FileHeader = nt_header;
					loaded_image.Sections = (PIMAGE_SECTION_HEADER)((LPBYTE)&nt_header->OptionalHeader + nt_header->FileHeader.SizeOfOptionalHeader);
					loaded_image.NumberOfSections = nt_header->FileHeader.NumberOfSections;
					loaded_image.SizeOfImage = GetFileSize(file, nullptr);
					loaded_image.Characteristics = nt_header->FileHeader.Characteristics;
					loaded_image.LastRvaSection = loaded_image.Sections;
					loaded_image.fSystemImage = false;
					loaded_image.fDOSImage = false;
					loaded_image.Links.Flink = &loaded_image.Links;
					loaded_image.Links.Blink = &loaded_image.Links;

					ULONG size = 0;
					const IMAGE_IMPORT_DESCRIPTOR *import_desc = (const IMAGE_IMPORT_DESCRIPTOR *)ImageDirectoryEntryToData((HMODULE)loaded_image.MappedAddress, false, IMAGE_DIRECTORY_ENTRY_IMPORT, &size);
					if (import_desc) {
						for (; import_desc->Name && import_desc->FirstThunk; import_desc++) {
							char16_t full_name_wc[32767];
							const char *name_cs = (const char *)ImageRvaToVa(loaded_image.FileHeader, loaded_image.MappedAddress, import_desc->Name, nullptr);
							String name = String(name_cs);
							if (name.begins_with("api-ms-win-")) {
								r_checked.insert(name);
							} else if (SearchPathW(nullptr, (LPCWSTR)name.utf16().get_data(), nullptr, 32767, (LPWSTR)full_name_wc, nullptr)) {
								debug_dynamic_library_check_dependencies(String::utf16(full_name_wc), r_checked, r_missing);
							} else if (SearchPathW((LPCWSTR)(p_path.get_base_dir().utf16().get_data()), (LPCWSTR)name.utf16().get_data(), nullptr, 32767, (LPWSTR)full_name_wc, nullptr)) {
								debug_dynamic_library_check_dependencies(String::utf16(full_name_wc), r_checked, r_missing);
							} else {
								r_missing.insert(name);
							}
						}
					}
				}
				UnmapViewOfFile(mapping);
			}
			CloseHandle(file_mapping);
		}
		CloseHandle(file);
	}
}
#endif

Error OS_Windows::open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data) {
	String path = p_path;

	if (!FileAccess::exists(path)) {
		//this code exists so gdextension can load .dll files from within the executable path
		path = get_executable_path().get_base_dir().path_join(p_path.get_file());
	}
	// Path to load from may be different from original if we make copies.
	String load_path = path;

	ERR_FAIL_COND_V(!FileAccess::exists(path), ERR_FILE_NOT_FOUND);

	// Here we want a copy to be loaded.
	// This is so the original file isn't locked and can be updated by a compiler.
	if (p_data != nullptr && p_data->generate_temp_files) {
		// Copy the file to the same directory as the original with a prefix in the name.
		// This is so relative path to dependencies are satisfied.
		load_path = path.get_base_dir().path_join("~" + path.get_file());

		// If there's a left-over copy (possibly from a crash) then delete it first.
		if (FileAccess::exists(load_path)) {
			DirAccess::remove_absolute(load_path);
		}

		Error copy_err = DirAccess::copy_absolute(path, load_path);
		if (copy_err) {
			ERR_PRINT("Error copying library: " + path);
			return ERR_CANT_CREATE;
		}

		FileAccess::set_hidden_attribute(load_path, true);

		Error pdb_err = WindowsUtils::copy_and_rename_pdb(load_path);
		if (pdb_err != OK && pdb_err != ERR_SKIP) {
			WARN_PRINT(vformat("Failed to rename the PDB file. The original PDB file for '%s' will be loaded.", path));
		}
	}

	DLL_DIRECTORY_COOKIE cookie = nullptr;

	String dll_path = fix_path(load_path);
	String dll_dir = fix_path(ProjectSettings::get_singleton()->globalize_path(load_path.get_base_dir()));
	if (p_data != nullptr && p_data->also_set_library_path) {
		cookie = AddDllDirectory((LPCWSTR)(dll_dir.utf16().get_data()));
	}

	p_library_handle = (void *)LoadLibraryExW((LPCWSTR)(dll_path.utf16().get_data()), nullptr, (p_data != nullptr && p_data->also_set_library_path) ? LOAD_LIBRARY_SEARCH_DEFAULT_DIRS : 0);
	if (!p_library_handle) {
		if (p_data != nullptr && p_data->generate_temp_files) {
			DirAccess::remove_absolute(load_path);
		}

#ifdef DEBUG_ENABLED
		DWORD err_code = GetLastError();

		HashSet<String> checked_libs;
		HashSet<String> missing_libs;
		debug_dynamic_library_check_dependencies(dll_path, checked_libs, missing_libs);
		if (!missing_libs.is_empty()) {
			String missing;
			for (const String &E : missing_libs) {
				if (!missing.is_empty()) {
					missing += ", ";
				}
				missing += E;
			}
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Missing dependencies: %s. Error: %s.", p_path, missing, format_error_message(err_code)));
		} else {
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, format_error_message(err_code)));
		}
#endif
	}

#ifndef DEBUG_ENABLED
	ERR_FAIL_NULL_V_MSG(p_library_handle, ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, format_error_message(GetLastError())));
#endif

	if (cookie) {
		RemoveDllDirectory(cookie);
	}

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = path;
	}

	if (p_data != nullptr && p_data->generate_temp_files) {
		// Save the copied path so it can be deleted later.
		temp_libraries[p_library_handle] = load_path;
	}

	return OK;
}

Error OS_Windows::close_dynamic_library(void *p_library_handle) {
	if (!FreeLibrary((HMODULE)p_library_handle)) {
		return FAILED;
	}

	// Delete temporary copy of library if it exists.
	_remove_temp_library(p_library_handle);

	return OK;
}

void OS_Windows::_remove_temp_library(void *p_library_handle) {
	if (temp_libraries.has(p_library_handle)) {
		String path = temp_libraries[p_library_handle];
		DirAccess::remove_absolute(path);
		WindowsUtils::remove_temp_pdbs(path);
		temp_libraries.erase(p_library_handle);
	}
}

Error OS_Windows::get_dynamic_library_symbol_handle(void *p_library_handle, const String &p_name, void *&p_symbol_handle, bool p_optional) {
	p_symbol_handle = (void *)GetProcAddress((HMODULE)p_library_handle, p_name.utf8().get_data());
	if (!p_symbol_handle) {
		if (!p_optional) {
			ERR_FAIL_V_MSG(ERR_CANT_RESOLVE, vformat("Can't resolve symbol %s, error: \"%s\".", p_name, format_error_message(GetLastError())));
		} else {
			return ERR_CANT_RESOLVE;
		}
	}
	return OK;
}

String OS_Windows::get_name() const {
	return "Windows";
}

String OS_Windows::get_distribution_name() const {
	return get_name();
}

String OS_Windows::get_version() const {
	RtlGetVersionPtr version_ptr = (RtlGetVersionPtr)(void *)GetProcAddress(GetModuleHandle("ntdll.dll"), "RtlGetVersion");
	if (version_ptr != nullptr) {
		RTL_OSVERSIONINFOEXW fow;
		ZeroMemory(&fow, sizeof(fow));
		fow.dwOSVersionInfoSize = sizeof(fow);
		if (version_ptr(&fow) == 0x00000000) {
			return vformat("%d.%d.%d", (int64_t)fow.dwMajorVersion, (int64_t)fow.dwMinorVersion, (int64_t)fow.dwBuildNumber);
		}
	}
	return "";
}

String OS_Windows::get_version_alias() const {
	RtlGetVersionPtr version_ptr = (RtlGetVersionPtr)(void *)GetProcAddress(GetModuleHandle("ntdll.dll"), "RtlGetVersion");
	if (version_ptr != nullptr) {
		RTL_OSVERSIONINFOEXW fow;
		ZeroMemory(&fow, sizeof(fow));
		fow.dwOSVersionInfoSize = sizeof(fow);
		if (version_ptr(&fow) == 0x00000000) {
			String windows_string;
			if (fow.wProductType != VER_NT_WORKSTATION && fow.dwMajorVersion == 10 && fow.dwBuildNumber >= 26100) {
				windows_string = "Server 2025";
			} else if (fow.dwMajorVersion == 10 && fow.dwBuildNumber >= 20348) {
				// Builds above 20348 correspond to Windows 11 / Windows Server 2022.
				// Their major version numbers are still 10 though, not 11.
				if (fow.wProductType != VER_NT_WORKSTATION) {
					windows_string += "Server 2022";
				} else {
					windows_string += "11";
				}
			} else if (fow.dwMajorVersion == 10) {
				if (fow.wProductType != VER_NT_WORKSTATION && fow.dwBuildNumber >= 17763) {
					windows_string += "Server 2019";
				} else {
					if (fow.wProductType != VER_NT_WORKSTATION) {
						windows_string += "Server 2016";
					} else {
						windows_string += "10";
					}
				}
			} else {
				windows_string += "Unknown";
			}
			// Windows versions older than 10 cannot run Godot.

			return vformat("%s (build %d)", windows_string, (int64_t)fow.dwBuildNumber);
		}
	}

	return "";
}

Vector<String> OS_Windows::_get_video_adapter_driver_info_reg(const String &p_name) const {
	Vector<String> info;

	String subkey = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";
	HKEY hkey = nullptr;
	LSTATUS result = RegOpenKeyExW(HKEY_LOCAL_MACHINE, (LPCWSTR)subkey.utf16().get_data(), 0, KEY_READ, &hkey);
	if (result != ERROR_SUCCESS) {
		return Vector<String>();
	}

	DWORD subkeys = 0;
	result = RegQueryInfoKeyW(hkey, nullptr, nullptr, nullptr, &subkeys, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
	if (result != ERROR_SUCCESS) {
		RegCloseKey(hkey);
		return Vector<String>();
	}
	for (DWORD i = 0; i < subkeys; i++) {
		WCHAR key_name[MAX_PATH] = L"";
		DWORD key_name_size = MAX_PATH;
		result = RegEnumKeyExW(hkey, i, key_name, &key_name_size, nullptr, nullptr, nullptr, nullptr);
		if (result != ERROR_SUCCESS) {
			continue;
		}
		String id = String::utf16((const char16_t *)key_name, key_name_size);
		if (!id.is_empty()) {
			HKEY sub_hkey = nullptr;
			result = RegOpenKeyExW(HKEY_LOCAL_MACHINE, (LPCWSTR)(subkey + "\\" + id).utf16().get_data(), 0, KEY_QUERY_VALUE, &sub_hkey);
			if (result != ERROR_SUCCESS) {
				continue;
			}

			WCHAR buffer[4096];
			DWORD buffer_len = 4096;
			DWORD vtype = REG_SZ;
			if (RegQueryValueExW(sub_hkey, L"DriverDesc", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) != ERROR_SUCCESS || buffer_len == 0) {
				buffer_len = 4096;
				if (RegQueryValueExW(sub_hkey, L"HardwareInformation.AdapterString", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) != ERROR_SUCCESS || buffer_len == 0) {
					RegCloseKey(sub_hkey);
					continue;
				}
			}

			String driver_name = String::utf16((const char16_t *)buffer, buffer_len).strip_edges();
			if (driver_name == p_name) {
				String driver_provider = driver_name;
				String driver_version;

				buffer_len = 4096;
				if (RegQueryValueExW(sub_hkey, L"ProviderName", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) == ERROR_SUCCESS && buffer_len != 0) {
					driver_provider = String::utf16((const char16_t *)buffer, buffer_len).strip_edges();
				}
				buffer_len = 4096;
				if (RegQueryValueExW(sub_hkey, L"DriverVersion", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) == ERROR_SUCCESS && buffer_len != 0) {
					driver_version = String::utf16((const char16_t *)buffer, buffer_len).strip_edges();
				}
				if (!driver_version.is_empty()) {
					info.push_back(driver_provider);
					info.push_back(driver_version);

					RegCloseKey(sub_hkey);
					break;
				}
			}
			RegCloseKey(sub_hkey);
		}
	}
	RegCloseKey(hkey);
	return info;
}

Vector<String> OS_Windows::_get_video_adapter_driver_info_wmi(const String &p_name) const {
	Vector<String> info;

	REFCLSID clsid = CLSID_WbemLocator; // Unmarshaler CLSID
	REFIID uuid = IID_IWbemLocator; // Interface UUID
	IWbemLocator *wbemLocator = nullptr; // to get the services
	IWbemServices *wbemServices = nullptr; // to get the class
	IEnumWbemClassObject *iter = nullptr;
	IWbemClassObject *pnpSDriverObject[1]; // contains driver name, version, etc.
	String driver_name;
	String driver_version;

	HRESULT hr = CoCreateInstance(clsid, nullptr, CLSCTX_INPROC_SERVER, uuid, (LPVOID *)&wbemLocator);
	if (hr != S_OK) {
		return Vector<String>();
	}
	BSTR resource_name = SysAllocString(L"root\\CIMV2");
	hr = wbemLocator->ConnectServer(resource_name, nullptr, nullptr, nullptr, 0, nullptr, nullptr, &wbemServices);
	SysFreeString(resource_name);

	SAFE_RELEASE(wbemLocator) // from now on, use `wbemServices`
	if (hr != S_OK) {
		SAFE_RELEASE(wbemServices)
		return Vector<String>();
	}

	const String gpu_device_class_query = vformat("SELECT * FROM Win32_PnPSignedDriver WHERE DeviceName = \"%s\"", p_name);
	BSTR query = SysAllocString((const WCHAR *)gpu_device_class_query.utf16().get_data());
	BSTR query_lang = SysAllocString(L"WQL");
	hr = wbemServices->ExecQuery(query_lang, query, WBEM_FLAG_RETURN_IMMEDIATELY | WBEM_FLAG_FORWARD_ONLY, nullptr, &iter);
	SysFreeString(query_lang);
	SysFreeString(query);
	if (hr == S_OK) {
		ULONG resultCount;
		hr = iter->Next(5000, 1, pnpSDriverObject, &resultCount); // Get exactly 1. Wait max 5 seconds.

		if (hr == S_OK && resultCount > 0) {
			VARIANT dn;
			VariantInit(&dn);

			BSTR object_name = SysAllocString(L"DriverName");
			hr = pnpSDriverObject[0]->Get(object_name, 0, &dn, nullptr, nullptr);
			SysFreeString(object_name);
			if (hr == S_OK && dn.vt == VT_BSTR) {
				String d_name = String(V_BSTR(&dn));
				if (d_name.is_empty()) {
					object_name = SysAllocString(L"DriverProviderName");
					hr = pnpSDriverObject[0]->Get(object_name, 0, &dn, nullptr, nullptr);
					SysFreeString(object_name);
					if (hr == S_OK) {
						driver_name = String(V_BSTR(&dn));
					}
				} else {
					driver_name = d_name;
				}
			} else {
				object_name = SysAllocString(L"DriverProviderName");
				hr = pnpSDriverObject[0]->Get(object_name, 0, &dn, nullptr, nullptr);
				SysFreeString(object_name);
				if (hr == S_OK && dn.vt == VT_BSTR) {
					driver_name = String(V_BSTR(&dn));
				} else {
					driver_name = "Unknown";
				}
			}

			VARIANT dv;
			VariantInit(&dv);
			object_name = SysAllocString(L"DriverVersion");
			hr = pnpSDriverObject[0]->Get(object_name, 0, &dv, nullptr, nullptr);
			SysFreeString(object_name);
			if (hr == S_OK && dv.vt == VT_BSTR) {
				driver_version = String(V_BSTR(&dv));
			} else {
				driver_version = "Unknown";
			}
			for (ULONG i = 0; i < resultCount; i++) {
				SAFE_RELEASE(pnpSDriverObject[i])
			}
		}
	}

	SAFE_RELEASE(wbemServices)
	SAFE_RELEASE(iter)

	info.push_back(driver_name);
	info.push_back(driver_version);

	return info;
}

Vector<String> OS_Windows::get_video_adapter_driver_info() const {
	if (RenderingServer::get_singleton() == nullptr) {
		return Vector<String>();
	}

	static Vector<String> info;
	if (!info.is_empty()) {
		return info;
	}

	const String device_name = RenderingServer::get_singleton()->get_video_adapter_name();
	if (device_name.is_empty()) {
		return Vector<String>();
	}

	info = _get_video_adapter_driver_info_reg(device_name);
	if (info.is_empty()) {
		info = _get_video_adapter_driver_info_wmi(device_name);
	}
	return info;
}

bool OS_Windows::get_user_prefers_integrated_gpu() const {
	// On Windows 10, the preferred GPU configured in Windows Settings is
	// stored in the registry under the key
	// `HKEY_CURRENT_USER\SOFTWARE\Microsoft\DirectX\UserGpuPreferences`
	// with the name being the app ID or EXE path. The value is in the form of
	// `GpuPreference=1;`, with the value being 1 for integrated GPU and 2
	// for discrete GPU. On Windows 11, there may be more flags, separated
	// by semicolons.

	// If this is a packaged app, use the "application user model ID".
	// Otherwise, use the EXE path.
	WCHAR value_name[32768];
	bool is_packaged = false;
	{
		HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
		if (kernel32) {
			using GetCurrentApplicationUserModelIdPtr = LONG(WINAPI *)(UINT32 * length, PWSTR id);
			GetCurrentApplicationUserModelIdPtr GetCurrentApplicationUserModelId = (GetCurrentApplicationUserModelIdPtr)(void *)GetProcAddress(kernel32, "GetCurrentApplicationUserModelId");

			if (GetCurrentApplicationUserModelId) {
				UINT32 length = std_size(value_name);
				LONG result = GetCurrentApplicationUserModelId(&length, value_name);
				if (result == ERROR_SUCCESS) {
					is_packaged = true;
				}
			}
		}
	}
	if (!is_packaged && GetModuleFileNameW(nullptr, value_name, sizeof(value_name) / sizeof(value_name[0])) >= sizeof(value_name) / sizeof(value_name[0])) {
		// Paths should never be longer than 32767, but just in case.
		return false;
	}

	LPCWSTR subkey = L"SOFTWARE\\Microsoft\\DirectX\\UserGpuPreferences";
	HKEY hkey = nullptr;
	LSTATUS result = RegOpenKeyExW(HKEY_CURRENT_USER, subkey, 0, KEY_READ, &hkey);
	if (result != ERROR_SUCCESS) {
		return false;
	}

	DWORD size = 0;
	result = RegGetValueW(hkey, nullptr, value_name, RRF_RT_REG_SZ, nullptr, nullptr, &size);
	if (result != ERROR_SUCCESS || size == 0) {
		RegCloseKey(hkey);
		return false;
	}

	Vector<WCHAR> buffer;
	buffer.resize(size / sizeof(WCHAR));
	result = RegGetValueW(hkey, nullptr, value_name, RRF_RT_REG_SZ, nullptr, (LPBYTE)buffer.ptrw(), &size);
	if (result != ERROR_SUCCESS) {
		RegCloseKey(hkey);
		return false;
	}

	RegCloseKey(hkey);
	const String flags = String::utf16((const char16_t *)buffer.ptr(), size / sizeof(WCHAR));

	for (const String &flag : flags.split(";", false)) {
		if (flag == "GpuPreference=1") {
			return true;
		}
	}
	return false;
}

OS::DateTime OS_Windows::get_datetime(bool p_utc) const {
	SYSTEMTIME systemtime;
	if (p_utc) {
		GetSystemTime(&systemtime);
	} else {
		GetLocalTime(&systemtime);
	}

	//Get DST information from Windows, but only if p_utc is false.
	TIME_ZONE_INFORMATION info;
	bool is_daylight = false;
	if (!p_utc && GetTimeZoneInformation(&info) == TIME_ZONE_ID_DAYLIGHT) {
		is_daylight = true;
	}

	DateTime dt;
	dt.year = systemtime.wYear;
	dt.month = Month(systemtime.wMonth);
	dt.day = systemtime.wDay;
	dt.weekday = Weekday(systemtime.wDayOfWeek);
	dt.hour = systemtime.wHour;
	dt.minute = systemtime.wMinute;
	dt.second = systemtime.wSecond;
	dt.dst = is_daylight;
	return dt;
}

OS::TimeZoneInfo OS_Windows::get_time_zone_info() const {
	TIME_ZONE_INFORMATION info;
	bool is_daylight = false;
	if (GetTimeZoneInformation(&info) == TIME_ZONE_ID_DAYLIGHT) {
		is_daylight = true;
	}

	// Daylight Bias needs to be added to the bias if DST is in effect, or else it will not properly update.
	TimeZoneInfo ret;
	if (is_daylight) {
		ret.name = info.DaylightName;
		ret.bias = info.Bias + info.DaylightBias;
	} else {
		ret.name = info.StandardName;
		ret.bias = info.Bias + info.StandardBias;
	}

	// Bias value returned by GetTimeZoneInformation is inverted of what we expect
	// For example, on GMT-3 GetTimeZoneInformation return a Bias of 180, so invert the value to get -180
	ret.bias = -ret.bias;
	return ret;
}

double OS_Windows::get_unix_time() const {
	// 1 Windows tick is 100ns
	const uint64_t WINDOWS_TICKS_PER_SECOND = 10000000;
	const uint64_t TICKS_TO_UNIX_EPOCH = 116444736000000000LL;

	SYSTEMTIME st;
	GetSystemTime(&st);
	FILETIME ft;
	SystemTimeToFileTime(&st, &ft);
	uint64_t ticks_time;
	ticks_time = ft.dwHighDateTime;
	ticks_time <<= 32;
	ticks_time |= ft.dwLowDateTime;

	return (double)(ticks_time - TICKS_TO_UNIX_EPOCH) / WINDOWS_TICKS_PER_SECOND;
}

void OS_Windows::delay_usec(uint32_t p_usec) const {
	if (p_usec < 1000) {
		Sleep(1);
	} else {
		Sleep(p_usec / 1000);
	}
}

uint64_t OS_Windows::get_ticks_usec() const {
	uint64_t ticks;

	// This is the number of clock ticks since start
	QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
	// Subtract the ticks at game start to get
	// the ticks since the game started
	ticks -= ticks_start;

	// Divide by frequency to get the time in seconds
	// original calculation shown below is subject to overflow
	// with high ticks_per_second and a number of days since the last reboot.
	// time = ticks * 1000000L / ticks_per_second;

	// we can prevent this by either using 128 bit math
	// or separating into a calculation for seconds, and the fraction
	uint64_t seconds = ticks / ticks_per_second;

	// compiler will optimize these two into one divide
	uint64_t leftover = ticks % ticks_per_second;

	// remainder
	uint64_t time = (leftover * 1000000L) / ticks_per_second;

	// seconds
	time += seconds * 1000000L;

	return time;
}

String OS_Windows::_quote_command_line_argument(const String &p_text) const {
	for (int i = 0; i < p_text.size(); i++) {
		char32_t c = p_text[i];
		if (c == ' ' || c == '&' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '^' || c == '=' || c == ';' || c == '!' || c == '\'' || c == '+' || c == ',' || c == '`' || c == '~') {
			return "\"" + p_text + "\"";
		}
	}
	return p_text;
}

static void _append_to_pipe(char *p_bytes, int p_size, String *r_pipe, Mutex *p_pipe_mutex) {
	// Try to convert from default ANSI code page to Unicode.
	LocalVector<wchar_t> wchars;
	int total_wchars = MultiByteToWideChar(CP_ACP, 0, p_bytes, p_size, nullptr, 0);
	if (total_wchars > 0) {
		wchars.resize(total_wchars);
		if (MultiByteToWideChar(CP_ACP, 0, p_bytes, p_size, wchars.ptr(), total_wchars) == 0) {
			wchars.clear();
		}
	}

	if (p_pipe_mutex) {
		p_pipe_mutex->lock();
	}
	if (wchars.is_empty()) {
		// Let's hope it's compatible with UTF-8.
		(*r_pipe) += String::utf8(p_bytes, p_size);
	} else {
		(*r_pipe) += String::utf16((char16_t *)wchars.ptr(), total_wchars);
	}
	if (p_pipe_mutex) {
		p_pipe_mutex->unlock();
	}
}

void OS_Windows::_init_encodings() {
	encodings[""] = 0;
	encodings["CP_ACP"] = 0;
	encodings["CP_OEMCP"] = 1;
	encodings["CP_MACCP"] = 2;
	encodings["CP_THREAD_ACP"] = 3;
	encodings["CP_SYMBOL"] = 42;
	encodings["IBM037"] = 37;
	encodings["IBM437"] = 437;
	encodings["IBM500"] = 500;
	encodings["ASMO-708"] = 708;
	encodings["ASMO-449"] = 709;
	encodings["DOS-710"] = 710;
	encodings["DOS-720"] = 720;
	encodings["IBM737"] = 737;
	encodings["IBM775"] = 775;
	encodings["IBM850"] = 850;
	encodings["IBM852"] = 852;
	encodings["IBM855"] = 855;
	encodings["IBM857"] = 857;
	encodings["IBM00858"] = 858;
	encodings["IBM860"] = 860;
	encodings["IBM861"] = 861;
	encodings["DOS-862"] = 862;
	encodings["IBM863"] = 863;
	encodings["IBM864"] = 864;
	encodings["IBM865"] = 865;
	encodings["CP866"] = 866;
	encodings["IBM869"] = 869;
	encodings["IBM870"] = 870;
	encodings["WINDOWS-874"] = 874;
	encodings["CP875"] = 875;
	encodings["SHIFT_JIS"] = 932;
	encodings["GB2312"] = 936;
	encodings["KS_C_5601-1987"] = 949;
	encodings["BIG5"] = 950;
	encodings["IBM1026"] = 1026;
	encodings["IBM01047"] = 1047;
	encodings["IBM01140"] = 1140;
	encodings["IBM01141"] = 1141;
	encodings["IBM01142"] = 1142;
	encodings["IBM01143"] = 1143;
	encodings["IBM01144"] = 1144;
	encodings["IBM01145"] = 1145;
	encodings["IBM01146"] = 1146;
	encodings["IBM01147"] = 1147;
	encodings["IBM01148"] = 1148;
	encodings["IBM01149"] = 1149;
	encodings["UTF-16"] = 1200;
	encodings["UNICODEFFFE"] = 1201;
	encodings["WINDOWS-1250"] = 1250;
	encodings["WINDOWS-1251"] = 1251;
	encodings["WINDOWS-1252"] = 1252;
	encodings["WINDOWS-1253"] = 1253;
	encodings["WINDOWS-1254"] = 1254;
	encodings["WINDOWS-1255"] = 1255;
	encodings["WINDOWS-1256"] = 1256;
	encodings["WINDOWS-1257"] = 1257;
	encodings["WINDOWS-1258"] = 1258;
	encodings["JOHAB"] = 1361;
	encodings["MACINTOSH"] = 10000;
	encodings["X-MAC-JAPANESE"] = 10001;
	encodings["X-MAC-CHINESETRAD"] = 10002;
	encodings["X-MAC-KOREAN"] = 10003;
	encodings["X-MAC-ARABIC"] = 10004;
	encodings["X-MAC-HEBREW"] = 10005;
	encodings["X-MAC-GREEK"] = 10006;
	encodings["X-MAC-CYRILLIC"] = 10007;
	encodings["X-MAC-CHINESESIMP"] = 10008;
	encodings["X-MAC-ROMANIAN"] = 10010;
	encodings["X-MAC-UKRAINIAN"] = 10017;
	encodings["X-MAC-THAI"] = 10021;
	encodings["X-MAC-CE"] = 10029;
	encodings["X-MAC-ICELANDIC"] = 10079;
	encodings["X-MAC-TURKISH"] = 10081;
	encodings["X-MAC-CROATIAN"] = 10082;
	encodings["UTF-32"] = 12000;
	encodings["UTF-32BE"] = 12001;
	encodings["X-CHINESE_CNS"] = 20000;
	encodings["X-CP20001"] = 20001;
	encodings["X_CHINESE-ETEN"] = 20002;
	encodings["X-CP20003"] = 20003;
	encodings["X-CP20004"] = 20004;
	encodings["X-CP20005"] = 20005;
	encodings["X-IA5"] = 20105;
	encodings["X-IA5-GERMAN"] = 20106;
	encodings["X-IA5-SWEDISH"] = 20107;
	encodings["X-IA5-NORWEGIAN"] = 20108;
	encodings["US-ASCII"] = 20127;
	encodings["X-CP20261"] = 20261;
	encodings["X-CP20269"] = 20269;
	encodings["IBM273"] = 20273;
	encodings["IBM277"] = 20277;
	encodings["IBM278"] = 20278;
	encodings["IBM280"] = 20280;
	encodings["IBM284"] = 20284;
	encodings["IBM285"] = 20285;
	encodings["IBM290"] = 20290;
	encodings["IBM297"] = 20297;
	encodings["IBM420"] = 20420;
	encodings["IBM423"] = 20423;
	encodings["IBM424"] = 20424;
	encodings["X-EBCDIC-KOREANEXTENDED"] = 20833;
	encodings["IBM-THAI"] = 20838;
	encodings["KOI8-R"] = 20866;
	encodings["IBM871"] = 20871;
	encodings["IBM880"] = 20880;
	encodings["IBM905"] = 20905;
	encodings["IBM00924"] = 20924;
	encodings["EUC-JP"] = 20932;
	encodings["X-CP20936"] = 20936;
	encodings["X-CP20949"] = 20949;
	encodings["CP1025"] = 21025;
	encodings["KOI8-U"] = 21866;
	encodings["ISO-8859-1"] = 28591;
	encodings["ISO-8859-2"] = 28592;
	encodings["ISO-8859-3"] = 28593;
	encodings["ISO-8859-4"] = 28594;
	encodings["ISO-8859-5"] = 28595;
	encodings["ISO-8859-6"] = 28596;
	encodings["ISO-8859-7"] = 28597;
	encodings["ISO-8859-8"] = 28598;
	encodings["ISO-8859-9"] = 28599;
	encodings["ISO-8859-13"] = 28603;
	encodings["ISO-8859-15"] = 28605;
	encodings["X-EUROPA"] = 29001;
	encodings["ISO-8859-8-I"] = 38598;
	encodings["ISO-2022-JP"] = 50220;
	encodings["CSISO2022JP"] = 50221;
	encodings["ISO-2022-JP"] = 50222;
	encodings["ISO-2022-KR"] = 50225;
	encodings["X-CP50227"] = 50227;
	encodings["EBCDIC-JP"] = 50930;
	encodings["EBCDIC-US-JP"] = 50931;
	encodings["EBCDIC-KR"] = 50933;
	encodings["EBCDIC-CN-eXT"] = 50935;
	encodings["EBCDIC-CN"] = 50936;
	encodings["EBCDIC-US-CN"] = 50937;
	encodings["EBCDIC-JP-EXT"] = 50939;
	encodings["EUC-JP"] = 51932;
	encodings["EUC-CN"] = 51936;
	encodings["EUC-KR"] = 51949;
	encodings["HZ-GB-2312"] = 52936;
	encodings["GB18030"] = 54936;
	encodings["X-ISCII-DE"] = 57002;
	encodings["X-ISCII-BE"] = 57003;
	encodings["X-ISCII-TA"] = 57004;
	encodings["X-ISCII-TE"] = 57005;
	encodings["X-ISCII-AS"] = 57006;
	encodings["X-ISCII-OR"] = 57007;
	encodings["X-ISCII-KA"] = 57008;
	encodings["X-ISCII-MA"] = 57009;
	encodings["X-ISCII-GU"] = 57010;
	encodings["X-ISCII-PA"] = 57011;
	encodings["UTF-7"] = 65000;
	encodings["UTF-8"] = 65001;
}

String OS_Windows::multibyte_to_string(const String &p_encoding, const PackedByteArray &p_array) const {
	const int *encoding = encodings.getptr(p_encoding.to_upper());
	ERR_FAIL_NULL_V_MSG(encoding, String(), "Conversion failed: Unknown encoding");

	LocalVector<wchar_t> wchars;
	int total_wchars = MultiByteToWideChar(*encoding, 0, (const char *)p_array.ptr(), p_array.size(), nullptr, 0);
	if (total_wchars == 0) {
		DWORD err_code = GetLastError();
		ERR_FAIL_V_MSG(String(), vformat("Conversion failed: %s", format_error_message(err_code)));
	}
	wchars.resize(total_wchars);
	if (MultiByteToWideChar(*encoding, 0, (const char *)p_array.ptr(), p_array.size(), wchars.ptr(), total_wchars) == 0) {
		DWORD err_code = GetLastError();
		ERR_FAIL_V_MSG(String(), vformat("Conversion failed: %s", format_error_message(err_code)));
	}

	return String::utf16((const char16_t *)wchars.ptr(), wchars.size());
}

PackedByteArray OS_Windows::string_to_multibyte(const String &p_encoding, const String &p_string) const {
	const int *encoding = encodings.getptr(p_encoding.to_upper());
	ERR_FAIL_NULL_V_MSG(encoding, PackedByteArray(), "Conversion failed: Unknown encoding");

	Char16String charstr = p_string.utf16();
	PackedByteArray ret;
	int total_mbchars = WideCharToMultiByte(*encoding, 0, (const wchar_t *)charstr.ptr(), charstr.size(), nullptr, 0, nullptr, nullptr);
	if (total_mbchars == 0) {
		DWORD err_code = GetLastError();
		ERR_FAIL_V_MSG(PackedByteArray(), vformat("Conversion failed: %s", format_error_message(err_code)));
	}

	ret.resize(total_mbchars);
	if (WideCharToMultiByte(*encoding, 0, (const wchar_t *)charstr.ptr(), charstr.size(), (char *)ret.ptrw(), ret.size(), nullptr, nullptr) == 0) {
		DWORD err_code = GetLastError();
		ERR_FAIL_V_MSG(PackedByteArray(), vformat("Conversion failed: %s", format_error_message(err_code)));
	}

	return ret;
}

Dictionary OS_Windows::get_memory_info() const {
	Dictionary meminfo;

	meminfo["physical"] = -1;
	meminfo["free"] = -1;
	meminfo["available"] = -1;
	meminfo["stack"] = -1;

	PERFORMANCE_INFORMATION pref_info;
	pref_info.cb = sizeof(pref_info);
	GetPerformanceInfo(&pref_info, sizeof(pref_info));

	typedef void(WINAPI * PGetCurrentThreadStackLimits)(PULONG_PTR, PULONG_PTR);
	PGetCurrentThreadStackLimits GetCurrentThreadStackLimits = (PGetCurrentThreadStackLimits)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetCurrentThreadStackLimits");

	ULONG_PTR LowLimit = 0;
	ULONG_PTR HighLimit = 0;
	if (GetCurrentThreadStackLimits) {
		GetCurrentThreadStackLimits(&LowLimit, &HighLimit);
	}

	if (pref_info.PhysicalTotal * pref_info.PageSize != 0) {
		meminfo["physical"] = static_cast<int64_t>(pref_info.PhysicalTotal * pref_info.PageSize);
	}
	if (pref_info.PhysicalAvailable * pref_info.PageSize != 0) {
		meminfo["free"] = static_cast<int64_t>(pref_info.PhysicalAvailable * pref_info.PageSize);
	}
	if (pref_info.CommitLimit * pref_info.PageSize != 0) {
		meminfo["available"] = static_cast<int64_t>(pref_info.CommitLimit * pref_info.PageSize);
	}
	if (HighLimit - LowLimit != 0) {
		meminfo["stack"] = static_cast<int64_t>(HighLimit - LowLimit);
	}

	return meminfo;
}

Dictionary OS_Windows::execute_with_pipe(const String &p_path, const List<String> &p_arguments, bool p_blocking) {
#define CLEAN_PIPES               \
	if (pipe_in[0] != 0) {        \
		CloseHandle(pipe_in[0]);  \
	}                             \
	if (pipe_in[1] != 0) {        \
		CloseHandle(pipe_in[1]);  \
	}                             \
	if (pipe_out[0] != 0) {       \
		CloseHandle(pipe_out[0]); \
	}                             \
	if (pipe_out[1] != 0) {       \
		CloseHandle(pipe_out[1]); \
	}                             \
	if (pipe_err[0] != 0) {       \
		CloseHandle(pipe_err[0]); \
	}                             \
	if (pipe_err[1] != 0) {       \
		CloseHandle(pipe_err[1]); \
	}

	Dictionary ret;

	String path = p_path.is_absolute_path() ? fix_path(p_path) : p_path;
	String command = _quote_command_line_argument(path);
	for (const String &E : p_arguments) {
		command += " " + _quote_command_line_argument(E);
	}

	// Create pipes.
	HANDLE pipe_in[2] = { nullptr, nullptr };
	HANDLE pipe_out[2] = { nullptr, nullptr };
	HANDLE pipe_err[2] = { nullptr, nullptr };

	SECURITY_ATTRIBUTES sa;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.bInheritHandle = true;
	sa.lpSecurityDescriptor = nullptr;

	ERR_FAIL_COND_V(!CreatePipe(&pipe_in[0], &pipe_in[1], &sa, 0), ret);
	if (!CreatePipe(&pipe_out[0], &pipe_out[1], &sa, 0)) {
		CLEAN_PIPES
		ERR_FAIL_V(ret);
	}
	if (!CreatePipe(&pipe_err[0], &pipe_err[1], &sa, 0)) {
		CLEAN_PIPES
		ERR_FAIL_V(ret);
	}
	ERR_FAIL_COND_V(!SetHandleInformation(pipe_err[0], HANDLE_FLAG_INHERIT, 0), ret);

	// Create process.
	ProcessInfo pi;
	ZeroMemory(&pi.si, sizeof(pi.si));
	pi.si.StartupInfo.cb = sizeof(pi.si);
	ZeroMemory(&pi.pi, sizeof(pi.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&pi.si.StartupInfo;

	pi.si.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;
	pi.si.StartupInfo.hStdInput = pipe_in[0];
	pi.si.StartupInfo.hStdOutput = pipe_out[1];
	pi.si.StartupInfo.hStdError = pipe_err[1];

	SIZE_T attr_list_size = 0;
	InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_list_size);
	pi.si.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)alloca(attr_list_size);
	if (!InitializeProcThreadAttributeList(pi.si.lpAttributeList, 1, 0, &attr_list_size)) {
		CLEAN_PIPES
		ERR_FAIL_V(ret);
	}
	HANDLE handles_to_inherit[] = { pipe_in[0], pipe_out[1], pipe_err[1] };
	if (!UpdateProcThreadAttribute(
				pi.si.lpAttributeList,
				0,
				PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
				handles_to_inherit,
				sizeof(handles_to_inherit),
				nullptr,
				nullptr)) {
		CLEAN_PIPES
		DeleteProcThreadAttributeList(pi.si.lpAttributeList);
		ERR_FAIL_V(ret);
	}

	DWORD creation_flags = NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT;

	Char16String current_dir_name;
	size_t str_len = GetCurrentDirectoryW(0, nullptr);
	current_dir_name.resize_uninitialized(str_len + 1);
	GetCurrentDirectoryW(current_dir_name.size(), (LPWSTR)current_dir_name.ptrw());
	if (current_dir_name.size() >= MAX_PATH) {
		Char16String current_short_dir_name;
		str_len = GetShortPathNameW((LPCWSTR)current_dir_name.ptr(), nullptr, 0);
		current_short_dir_name.resize_uninitialized(str_len);
		GetShortPathNameW((LPCWSTR)current_dir_name.ptr(), (LPWSTR)current_short_dir_name.ptrw(), current_short_dir_name.size());
		current_dir_name = current_short_dir_name;
	}

	if (!CreateProcessW(nullptr, (LPWSTR)(command.utf16().ptrw()), nullptr, nullptr, true, creation_flags, nullptr, (LPWSTR)current_dir_name.ptr(), si_w, &pi.pi)) {
		CLEAN_PIPES
		DeleteProcThreadAttributeList(pi.si.lpAttributeList);
		ERR_FAIL_V_MSG(ret, "Could not create child process: " + command);
	}
	CloseHandle(pipe_in[0]);
	CloseHandle(pipe_out[1]);
	CloseHandle(pipe_err[1]);
	DeleteProcThreadAttributeList(pi.si.lpAttributeList);

	ProcessID pid = pi.pi.dwProcessId;
	process_map_mutex.lock();
	process_map->insert(pid, pi);
	process_map_mutex.unlock();

	Ref<FileAccessWindowsPipe> main_pipe;
	main_pipe.instantiate();
	main_pipe->open_existing(pipe_out[0], pipe_in[1], p_blocking);

	Ref<FileAccessWindowsPipe> err_pipe;
	err_pipe.instantiate();
	err_pipe->open_existing(pipe_err[0], nullptr, p_blocking);

	ret["stdio"] = main_pipe;
	ret["stderr"] = err_pipe;
	ret["pid"] = pid;

#undef CLEAN_PIPES
	return ret;
}

Error OS_Windows::execute(const String &p_path, const List<String> &p_arguments, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
	String path = p_path.is_absolute_path() ? fix_path(p_path) : p_path;
	String command = _quote_command_line_argument(path);
	for (const String &E : p_arguments) {
		command += " " + _quote_command_line_argument(E);
	}

	ProcessInfo pi;
	ZeroMemory(&pi.si, sizeof(pi.si));
	pi.si.StartupInfo.cb = sizeof(pi.si);
	ZeroMemory(&pi.pi, sizeof(pi.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&pi.si.StartupInfo;

	bool inherit_handles = false;
	HANDLE pipe[2] = { nullptr, nullptr };
	if (r_pipe) {
		// Create pipe for StdOut and StdErr.
		SECURITY_ATTRIBUTES sa;
		sa.nLength = sizeof(SECURITY_ATTRIBUTES);
		sa.bInheritHandle = true;
		sa.lpSecurityDescriptor = nullptr;

		ERR_FAIL_COND_V(!CreatePipe(&pipe[0], &pipe[1], &sa, 0), ERR_CANT_FORK);

		pi.si.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;
		pi.si.StartupInfo.hStdOutput = pipe[1];
		if (read_stderr) {
			pi.si.StartupInfo.hStdError = pipe[1];
		}

		SIZE_T attr_list_size = 0;
		InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_list_size);
		pi.si.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)alloca(attr_list_size);
		if (!InitializeProcThreadAttributeList(pi.si.lpAttributeList, 1, 0, &attr_list_size)) {
			CloseHandle(pipe[0]); // Cleanup pipe handles.
			CloseHandle(pipe[1]);
			ERR_FAIL_V(ERR_CANT_FORK);
		}
		if (!UpdateProcThreadAttribute(
					pi.si.lpAttributeList,
					0,
					PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
					&pipe[1],
					sizeof(HANDLE),
					nullptr,
					nullptr)) {
			CloseHandle(pipe[0]); // Cleanup pipe handles.
			CloseHandle(pipe[1]);
			DeleteProcThreadAttributeList(pi.si.lpAttributeList);
			ERR_FAIL_V(ERR_CANT_FORK);
		}
		inherit_handles = true;
	}
	DWORD creation_flags = NORMAL_PRIORITY_CLASS;
	if (inherit_handles) {
		creation_flags |= EXTENDED_STARTUPINFO_PRESENT;
	}
	if (p_open_console) {
		creation_flags |= CREATE_NEW_CONSOLE;
	} else {
		creation_flags |= CREATE_NO_WINDOW;
	}

	Char16String current_dir_name;
	size_t str_len = GetCurrentDirectoryW(0, nullptr);
	current_dir_name.resize_uninitialized(str_len + 1);
	GetCurrentDirectoryW(current_dir_name.size(), (LPWSTR)current_dir_name.ptrw());
	if (current_dir_name.size() >= MAX_PATH) {
		Char16String current_short_dir_name;
		str_len = GetShortPathNameW((LPCWSTR)current_dir_name.ptr(), nullptr, 0);
		current_short_dir_name.resize_uninitialized(str_len);
		GetShortPathNameW((LPCWSTR)current_dir_name.ptr(), (LPWSTR)current_short_dir_name.ptrw(), current_short_dir_name.size());
		current_dir_name = current_short_dir_name;
	}

	int ret = CreateProcessW(nullptr, (LPWSTR)(command.utf16().ptrw()), nullptr, nullptr, inherit_handles, creation_flags, nullptr, (LPWSTR)current_dir_name.ptr(), si_w, &pi.pi);
	if (!ret && r_pipe) {
		CloseHandle(pipe[0]); // Cleanup pipe handles.
		CloseHandle(pipe[1]);
		DeleteProcThreadAttributeList(pi.si.lpAttributeList);
	}
	ERR_FAIL_COND_V_MSG(ret == 0, ERR_CANT_FORK, "Could not create child process: " + command);

	if (r_pipe) {
		CloseHandle(pipe[1]); // Close pipe write handle (only child process is writing).

		LocalVector<char> bytes;
		int bytes_in_buffer = 0;

		const int CHUNK_SIZE = 4096;
		DWORD read = 0;
		for (;;) { // Read StdOut and StdErr from pipe.
			bytes.resize(bytes_in_buffer + CHUNK_SIZE);
			const bool success = ReadFile(pipe[0], bytes.ptr() + bytes_in_buffer, CHUNK_SIZE, &read, nullptr);
			if (!success || read == 0) {
				break;
			}

			// Assume that all possible encodings are ASCII-compatible.
			// Break at newline to allow receiving long output in portions.
			int newline_index = -1;
			for (int i = read - 1; i >= 0; i--) {
				if (bytes[bytes_in_buffer + i] == '\n') {
					newline_index = i;
					break;
				}
			}
			if (newline_index == -1) {
				bytes_in_buffer += read;
				continue;
			}

			const int bytes_to_convert = bytes_in_buffer + (newline_index + 1);
			_append_to_pipe(bytes.ptr(), bytes_to_convert, r_pipe, p_pipe_mutex);

			bytes_in_buffer = read - (newline_index + 1);
			memmove(bytes.ptr(), bytes.ptr() + bytes_to_convert, bytes_in_buffer);
		}

		if (bytes_in_buffer > 0) {
			_append_to_pipe(bytes.ptr(), bytes_in_buffer, r_pipe, p_pipe_mutex);
		}

		CloseHandle(pipe[0]); // Close pipe read handle.
	}
	WaitForSingleObject(pi.pi.hProcess, INFINITE);

	if (r_exitcode) {
		DWORD ret2;
		GetExitCodeProcess(pi.pi.hProcess, &ret2);
		*r_exitcode = ret2;
	}

	CloseHandle(pi.pi.hProcess);
	CloseHandle(pi.pi.hThread);
	if (r_pipe) {
		DeleteProcThreadAttributeList(pi.si.lpAttributeList);
	}

	return OK;
}

Error OS_Windows::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
	String path = p_path.is_absolute_path() ? fix_path(p_path) : p_path;
	String command = _quote_command_line_argument(path);
	for (const String &E : p_arguments) {
		command += " " + _quote_command_line_argument(E);
	}

	ProcessInfo pi;
	ZeroMemory(&pi.si, sizeof(pi.si));
	pi.si.StartupInfo.cb = sizeof(pi.si.StartupInfo);
	ZeroMemory(&pi.pi, sizeof(pi.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&pi.si.StartupInfo;

	DWORD creation_flags = NORMAL_PRIORITY_CLASS;
	if (p_open_console) {
		creation_flags |= CREATE_NEW_CONSOLE;
	} else {
		creation_flags |= CREATE_NO_WINDOW;
	}

	Char16String current_dir_name;
	size_t str_len = GetCurrentDirectoryW(0, nullptr);
	current_dir_name.resize_uninitialized(str_len + 1);
	GetCurrentDirectoryW(current_dir_name.size(), (LPWSTR)current_dir_name.ptrw());
	if (current_dir_name.size() >= MAX_PATH) {
		Char16String current_short_dir_name;
		str_len = GetShortPathNameW((LPCWSTR)current_dir_name.ptr(), nullptr, 0);
		current_short_dir_name.resize_uninitialized(str_len);
		GetShortPathNameW((LPCWSTR)current_dir_name.ptr(), (LPWSTR)current_short_dir_name.ptrw(), current_short_dir_name.size());
		current_dir_name = current_short_dir_name;
	}

	int ret = CreateProcessW(nullptr, (LPWSTR)(command.utf16().ptrw()), nullptr, nullptr, false, creation_flags, nullptr, (LPWSTR)current_dir_name.ptr(), si_w, &pi.pi);
	ERR_FAIL_COND_V_MSG(ret == 0, ERR_CANT_FORK, "Could not create child process: " + command);

	ProcessID pid = pi.pi.dwProcessId;
	if (r_child_id) {
		*r_child_id = pid;
	}
	process_map_mutex.lock();
	process_map->insert(pid, pi);
	process_map_mutex.unlock();

	return OK;
}

Error OS_Windows::kill(const ProcessID &p_pid) {
	int ret = 0;
	MutexLock lock(process_map_mutex);
	if (process_map->has(p_pid)) {
		const PROCESS_INFORMATION pi = (*process_map)[p_pid].pi;
		process_map->erase(p_pid);

		ret = TerminateProcess(pi.hProcess, 0);

		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);
	} else {
		HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, false, (DWORD)p_pid);
		if (hProcess != nullptr) {
			ret = TerminateProcess(hProcess, 0);

			CloseHandle(hProcess);
		}
	}

	return ret != 0 ? OK : FAILED;
}

int OS_Windows::get_process_id() const {
	return _getpid();
}

bool OS_Windows::is_process_running(const ProcessID &p_pid) const {
	MutexLock lock(process_map_mutex);
	if (!process_map->has(p_pid)) {
		return false;
	}

	const ProcessInfo &info = (*process_map)[p_pid];
	if (!info.is_running) {
		return false;
	}

	const PROCESS_INFORMATION &pi = info.pi;
	DWORD dw_exit_code = 0;
	if (!GetExitCodeProcess(pi.hProcess, &dw_exit_code)) {
		return false;
	}

	if (dw_exit_code != STILL_ACTIVE) {
		info.is_running = false;
		info.exit_code = dw_exit_code;
		return false;
	}

	return true;
}

int OS_Windows::get_process_exit_code(const ProcessID &p_pid) const {
	MutexLock lock(process_map_mutex);
	if (!process_map->has(p_pid)) {
		return -1;
	}

	const ProcessInfo &info = (*process_map)[p_pid];
	if (!info.is_running) {
		return info.exit_code;
	}

	const PROCESS_INFORMATION &pi = info.pi;

	DWORD dw_exit_code = 0;
	if (!GetExitCodeProcess(pi.hProcess, &dw_exit_code)) {
		return -1;
	}

	if (dw_exit_code == STILL_ACTIVE) {
		return -1;
	}

	info.is_running = false;
	info.exit_code = dw_exit_code;
	return dw_exit_code;
}

Error OS_Windows::set_cwd(const String &p_cwd) {
	if (_wchdir((LPCWSTR)(p_cwd.utf16().get_data())) != 0) {
		return ERR_CANT_OPEN;
	}

	return OK;
}

String OS_Windows::get_cwd() const {
	Char16String real_current_dir_name;
	size_t str_len = GetCurrentDirectoryW(0, nullptr);
	real_current_dir_name.resize_uninitialized(str_len + 1);
	GetCurrentDirectoryW(real_current_dir_name.size(), (LPWSTR)real_current_dir_name.ptrw());
	return String::utf16((const char16_t *)real_current_dir_name.get_data()).trim_prefix(R"(\\?\)").replace_char('\\', '/');
}

Vector<String> OS_Windows::get_system_fonts() const {
	if (!dwrite_init) {
		return Vector<String>();
	}

	Vector<String> ret;
	HashSet<String> font_names;

	UINT32 family_count = font_collection->GetFontFamilyCount();
	for (UINT32 i = 0; i < family_count; i++) {
		ComAutoreleaseRef<IDWriteFontFamily> family;
		HRESULT hr = font_collection->GetFontFamily(i, &family.reference);
		ERR_CONTINUE(FAILED(hr) || family.is_null());

		ComAutoreleaseRef<IDWriteLocalizedStrings> family_names;
		hr = family->GetFamilyNames(&family_names.reference);
		ERR_CONTINUE(FAILED(hr) || family_names.is_null());

		UINT32 index = 0;
		BOOL exists = false;
		UINT32 length = 0;
		Char16String name;

		hr = family_names->FindLocaleName(L"en-us", &index, &exists);
		ERR_CONTINUE(FAILED(hr));

		hr = family_names->GetStringLength(index, &length);
		ERR_CONTINUE(FAILED(hr));

		name.resize_uninitialized(length + 1);
		hr = family_names->GetString(index, (WCHAR *)name.ptrw(), length + 1);
		ERR_CONTINUE(FAILED(hr));

		font_names.insert(String::utf16(name.ptr(), length));
	}

	for (const String &E : font_names) {
		ret.push_back(E);
	}
	return ret;
}

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wnon-virtual-dtor") // Silence warning due to a COM API weirdness.

class FallbackTextAnalysisSource : public IDWriteTextAnalysisSource {
	LONG _cRef = 1;

	bool rtl = false;
	Char16String string;
	Char16String locale;
	IDWriteNumberSubstitution *n_sub = nullptr;

public:
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, VOID **ppvInterface) override {
		if (IID_IUnknown == riid) {
			AddRef();
			*ppvInterface = (IUnknown *)this;
		} else if (__uuidof(IMMNotificationClient) == riid) {
			AddRef();
			*ppvInterface = (IMMNotificationClient *)this;
		} else {
			*ppvInterface = nullptr;
			return E_NOINTERFACE;
		}
		return S_OK;
	}

	ULONG STDMETHODCALLTYPE AddRef() override {
		return InterlockedIncrement(&_cRef);
	}

	ULONG STDMETHODCALLTYPE Release() override {
		ULONG ulRef = InterlockedDecrement(&_cRef);
		if (0 == ulRef) {
			delete this;
		}
		return ulRef;
	}

	HRESULT STDMETHODCALLTYPE GetTextAtPosition(UINT32 p_text_position, WCHAR const **r_text_string, UINT32 *r_text_length) override {
		if (p_text_position >= (UINT32)string.length()) {
			*r_text_string = nullptr;
			*r_text_length = 0;
			return S_OK;
		}
		*r_text_string = reinterpret_cast<const wchar_t *>(string.get_data()) + p_text_position;
		*r_text_length = string.length() - p_text_position;
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE GetTextBeforePosition(UINT32 p_text_position, WCHAR const **r_text_string, UINT32 *r_text_length) override {
		if (p_text_position < 1 || p_text_position >= (UINT32)string.length()) {
			*r_text_string = nullptr;
			*r_text_length = 0;
			return S_OK;
		}
		*r_text_string = reinterpret_cast<const wchar_t *>(string.get_data());
		*r_text_length = p_text_position;
		return S_OK;
	}

	DWRITE_READING_DIRECTION STDMETHODCALLTYPE GetParagraphReadingDirection() override {
		return (rtl) ? DWRITE_READING_DIRECTION_RIGHT_TO_LEFT : DWRITE_READING_DIRECTION_LEFT_TO_RIGHT;
	}

	HRESULT STDMETHODCALLTYPE GetLocaleName(UINT32 p_text_position, UINT32 *r_text_length, WCHAR const **r_locale_name) override {
		*r_locale_name = reinterpret_cast<const wchar_t *>(locale.get_data());
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE GetNumberSubstitution(UINT32 p_text_position, UINT32 *r_text_length, IDWriteNumberSubstitution **r_number_substitution) override {
		*r_number_substitution = n_sub;
		return S_OK;
	}

	FallbackTextAnalysisSource(const Char16String &p_text, const Char16String &p_locale, bool p_rtl, IDWriteNumberSubstitution *p_nsub) {
		_cRef = 1;
		string = p_text;
		locale = p_locale;
		n_sub = p_nsub;
		rtl = p_rtl;
	}

	virtual ~FallbackTextAnalysisSource() {}
};

GODOT_GCC_WARNING_POP

String OS_Windows::_get_default_fontname(const String &p_font_name) const {
	String font_name = p_font_name;
	if (font_name.to_lower() == "sans-serif") {
		font_name = "Arial";
	} else if (font_name.to_lower() == "serif") {
		font_name = "Times New Roman";
	} else if (font_name.to_lower() == "monospace") {
		font_name = "Courier New";
	} else if (font_name.to_lower() == "cursive") {
		font_name = "Comic Sans MS";
	} else if (font_name.to_lower() == "fantasy") {
		font_name = "Gabriola";
	}
	return font_name;
}

DWRITE_FONT_WEIGHT OS_Windows::_weight_to_dw(int p_weight) const {
	if (p_weight < 150) {
		return DWRITE_FONT_WEIGHT_THIN;
	} else if (p_weight < 250) {
		return DWRITE_FONT_WEIGHT_EXTRA_LIGHT;
	} else if (p_weight < 325) {
		return DWRITE_FONT_WEIGHT_LIGHT;
	} else if (p_weight < 375) {
		return DWRITE_FONT_WEIGHT_SEMI_LIGHT;
	} else if (p_weight < 450) {
		return DWRITE_FONT_WEIGHT_NORMAL;
	} else if (p_weight < 550) {
		return DWRITE_FONT_WEIGHT_MEDIUM;
	} else if (p_weight < 650) {
		return DWRITE_FONT_WEIGHT_DEMI_BOLD;
	} else if (p_weight < 750) {
		return DWRITE_FONT_WEIGHT_BOLD;
	} else if (p_weight < 850) {
		return DWRITE_FONT_WEIGHT_EXTRA_BOLD;
	} else if (p_weight < 925) {
		return DWRITE_FONT_WEIGHT_BLACK;
	} else {
		return DWRITE_FONT_WEIGHT_EXTRA_BLACK;
	}
}

DWRITE_FONT_STRETCH OS_Windows::_stretch_to_dw(int p_stretch) const {
	if (p_stretch < 56) {
		return DWRITE_FONT_STRETCH_ULTRA_CONDENSED;
	} else if (p_stretch < 69) {
		return DWRITE_FONT_STRETCH_EXTRA_CONDENSED;
	} else if (p_stretch < 81) {
		return DWRITE_FONT_STRETCH_CONDENSED;
	} else if (p_stretch < 93) {
		return DWRITE_FONT_STRETCH_SEMI_CONDENSED;
	} else if (p_stretch < 106) {
		return DWRITE_FONT_STRETCH_NORMAL;
	} else if (p_stretch < 137) {
		return DWRITE_FONT_STRETCH_SEMI_EXPANDED;
	} else if (p_stretch < 144) {
		return DWRITE_FONT_STRETCH_EXPANDED;
	} else if (p_stretch < 162) {
		return DWRITE_FONT_STRETCH_EXTRA_EXPANDED;
	} else {
		return DWRITE_FONT_STRETCH_ULTRA_EXPANDED;
	}
}

Vector<String> OS_Windows::get_system_font_path_for_text(const String &p_font_name, const String &p_text, const String &p_locale, const String &p_script, int p_weight, int p_stretch, bool p_italic) const {
	// This may be called before TextServerManager has been created, which would cause a crash downstream if we do not check here
	if (!dwrite2_init || !TextServerManager::get_singleton()) {
		return Vector<String>();
	}

	String font_name = _get_default_fontname(p_font_name);

	bool rtl = TS->is_locale_right_to_left(p_locale);
	Char16String text = p_text.utf16();
	Char16String locale = p_locale.utf16();

	ComAutoreleaseRef<IDWriteNumberSubstitution> number_substitution;
	HRESULT hr = dwrite_factory->CreateNumberSubstitution(DWRITE_NUMBER_SUBSTITUTION_METHOD_NONE, reinterpret_cast<const wchar_t *>(locale.get_data()), true, &number_substitution.reference);
	ERR_FAIL_COND_V(FAILED(hr) || number_substitution.is_null(), Vector<String>());

	FallbackTextAnalysisSource fs = FallbackTextAnalysisSource(text, locale, rtl, number_substitution.reference);
	UINT32 mapped_length = 0;
	FLOAT scale = 0.0;
	ComAutoreleaseRef<IDWriteFont> dwrite_font;
	hr = system_font_fallback->MapCharacters(
			&fs,
			0,
			(UINT32)text.length(),
			font_collection,
			reinterpret_cast<const wchar_t *>(font_name.utf16().get_data()),
			_weight_to_dw(p_weight),
			p_italic ? DWRITE_FONT_STYLE_ITALIC : DWRITE_FONT_STYLE_NORMAL,
			_stretch_to_dw(p_stretch),
			&mapped_length,
			&dwrite_font.reference,
			&scale);

	if (FAILED(hr) || dwrite_font.is_null()) {
		return Vector<String>();
	}

	ComAutoreleaseRef<IDWriteFontFace> dwrite_face;
	hr = dwrite_font->CreateFontFace(&dwrite_face.reference);
	if (FAILED(hr) || dwrite_face.is_null()) {
		return Vector<String>();
	}

	UINT32 number_of_files = 0;
	hr = dwrite_face->GetFiles(&number_of_files, nullptr);
	if (FAILED(hr)) {
		return Vector<String>();
	}
	Vector<ComAutoreleaseRef<IDWriteFontFile>> files;
	files.resize(number_of_files);
	hr = dwrite_face->GetFiles(&number_of_files, (IDWriteFontFile **)files.ptrw());
	if (FAILED(hr)) {
		return Vector<String>();
	}

	Vector<String> ret;
	for (UINT32 i = 0; i < number_of_files; i++) {
		void const *reference_key = nullptr;
		UINT32 reference_key_size = 0;
		ComAutoreleaseRef<IDWriteLocalFontFileLoader> loader;

		hr = files.write[i]->GetLoader((IDWriteFontFileLoader **)&loader.reference);
		if (FAILED(hr) || loader.is_null()) {
			continue;
		}
		hr = files.write[i]->GetReferenceKey(&reference_key, &reference_key_size);
		if (FAILED(hr)) {
			continue;
		}

		WCHAR file_path[32767];
		hr = loader->GetFilePathFromKey(reference_key, reference_key_size, &file_path[0], 32767);
		if (FAILED(hr)) {
			continue;
		}
		String fpath = String::utf16((const char16_t *)&file_path[0]).replace_char('\\', '/');

		WIN32_FIND_DATAW d;
		HANDLE fnd = FindFirstFileW((LPCWSTR)&file_path[0], &d);
		if (fnd != INVALID_HANDLE_VALUE) {
			String fname = String::utf16((const char16_t *)d.cFileName);
			if (!fname.is_empty()) {
				fpath = fpath.get_base_dir().path_join(fname);
			}
			FindClose(fnd);
		}
		ret.push_back(fpath);
	}
	return ret;
}

String OS_Windows::get_system_font_path(const String &p_font_name, int p_weight, int p_stretch, bool p_italic) const {
	if (!dwrite_init) {
		return String();
	}

	String font_name = _get_default_fontname(p_font_name);

	UINT32 index = 0;
	BOOL exists = false;
	HRESULT hr = font_collection->FindFamilyName((const WCHAR *)font_name.utf16().get_data(), &index, &exists);
	if (FAILED(hr) || !exists) {
		return String();
	}

	ComAutoreleaseRef<IDWriteFontFamily> family;
	hr = font_collection->GetFontFamily(index, &family.reference);
	if (FAILED(hr) || family.is_null()) {
		return String();
	}

	ComAutoreleaseRef<IDWriteFont> dwrite_font;
	hr = family->GetFirstMatchingFont(_weight_to_dw(p_weight), _stretch_to_dw(p_stretch), p_italic ? DWRITE_FONT_STYLE_ITALIC : DWRITE_FONT_STYLE_NORMAL, &dwrite_font.reference);
	if (FAILED(hr) || dwrite_font.is_null()) {
		return String();
	}

	ComAutoreleaseRef<IDWriteFontFace> dwrite_face;
	hr = dwrite_font->CreateFontFace(&dwrite_face.reference);
	if (FAILED(hr) || dwrite_face.is_null()) {
		return String();
	}

	UINT32 number_of_files = 0;
	hr = dwrite_face->GetFiles(&number_of_files, nullptr);
	if (FAILED(hr)) {
		return String();
	}
	Vector<ComAutoreleaseRef<IDWriteFontFile>> files;
	files.resize(number_of_files);
	hr = dwrite_face->GetFiles(&number_of_files, (IDWriteFontFile **)files.ptrw());
	if (FAILED(hr)) {
		return String();
	}

	for (UINT32 i = 0; i < number_of_files; i++) {
		void const *reference_key = nullptr;
		UINT32 reference_key_size = 0;
		ComAutoreleaseRef<IDWriteLocalFontFileLoader> loader;

		hr = files.write[i]->GetLoader((IDWriteFontFileLoader **)&loader.reference);
		if (FAILED(hr) || loader.is_null()) {
			continue;
		}
		hr = files.write[i]->GetReferenceKey(&reference_key, &reference_key_size);
		if (FAILED(hr)) {
			continue;
		}

		WCHAR file_path[32767];
		hr = loader->GetFilePathFromKey(reference_key, reference_key_size, &file_path[0], 32767);
		if (FAILED(hr)) {
			continue;
		}
		String fpath = String::utf16((const char16_t *)&file_path[0]).replace_char('\\', '/');

		WIN32_FIND_DATAW d;
		HANDLE fnd = FindFirstFileW((LPCWSTR)&file_path[0], &d);
		if (fnd != INVALID_HANDLE_VALUE) {
			String fname = String::utf16((const char16_t *)d.cFileName);
			if (!fname.is_empty()) {
				fpath = fpath.get_base_dir().path_join(fname);
			}
			FindClose(fnd);
		}

		return fpath;
	}
	return String();
}

String OS_Windows::get_executable_path() const {
	WCHAR bufname[4096];
	GetModuleFileNameW(nullptr, bufname, 4096);
	String s = String::utf16((const char16_t *)bufname).replace_char('\\', '/');
	return s;
}

bool OS_Windows::has_environment(const String &p_var) const {
	return GetEnvironmentVariableW((LPCWSTR)(p_var.utf16().get_data()), nullptr, 0) > 0;
}

String OS_Windows::get_environment(const String &p_var) const {
	WCHAR wval[0x7fff]; // MSDN says 32767 char is the maximum
	int wlen = GetEnvironmentVariableW((LPCWSTR)(p_var.utf16().get_data()), wval, 0x7fff);
	if (wlen > 0) {
		return String::utf16((const char16_t *)wval);
	}
	return "";
}

void OS_Windows::set_environment(const String &p_var, const String &p_value) const {
	ERR_FAIL_COND_MSG(p_var.is_empty() || p_var.contains_char('='), vformat("Invalid environment variable name '%s', cannot be empty or include '='.", p_var));
	Char16String var = p_var.utf16();
	Char16String value = p_value.utf16();
	ERR_FAIL_COND_MSG(var.length() + value.length() + 2 > 32767, vformat("Invalid definition for environment variable '%s', cannot exceed 32767 characters.", p_var));
	SetEnvironmentVariableW((LPCWSTR)(var.get_data()), (LPCWSTR)(value.get_data()));
}

void OS_Windows::unset_environment(const String &p_var) const {
	ERR_FAIL_COND_MSG(p_var.is_empty() || p_var.contains_char('='), vformat("Invalid environment variable name '%s', cannot be empty or include '='.", p_var));
	SetEnvironmentVariableW((LPCWSTR)(p_var.utf16().get_data()), nullptr); // Null to delete.
}

String OS_Windows::get_stdin_string(int64_t p_buffer_size) {
	if (get_stdin_type() == STD_HANDLE_INVALID) {
		return String();
	}

	Vector<uint8_t> data;
	data.resize(p_buffer_size);
	DWORD count = 0;
	if (ReadFile(GetStdHandle(STD_INPUT_HANDLE), data.ptrw(), data.size(), &count, nullptr)) {
		return String::utf8((const char *)data.ptr(), count).replace("\r\n", "\n").rstrip("\n");
	}

	return String();
}

PackedByteArray OS_Windows::get_stdin_buffer(int64_t p_buffer_size) {
	Vector<uint8_t> data;
	data.resize(p_buffer_size);
	DWORD count = 0;
	if (ReadFile(GetStdHandle(STD_INPUT_HANDLE), data.ptrw(), data.size(), &count, nullptr)) {
		return data;
	}

	return PackedByteArray();
}

OS_Windows::StdHandleType OS_Windows::get_stdin_type() const {
	HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
	if (h == 0 || h == INVALID_HANDLE_VALUE) {
		return STD_HANDLE_INVALID;
	}
	DWORD ftype = GetFileType(h);
	if (ftype == FILE_TYPE_UNKNOWN && GetLastError() != ERROR_SUCCESS) {
		return STD_HANDLE_UNKNOWN;
	}
	ftype &= ~(FILE_TYPE_REMOTE);

	if (ftype == FILE_TYPE_DISK) {
		return STD_HANDLE_FILE;
	} else if (ftype == FILE_TYPE_PIPE) {
		return STD_HANDLE_PIPE;
	} else {
		DWORD conmode = 0;
		BOOL res = GetConsoleMode(h, &conmode);
		if (!res && (GetLastError() == ERROR_INVALID_HANDLE)) {
			return STD_HANDLE_UNKNOWN; // Unknown character device.
		} else {
#ifndef WINDOWS_SUBSYSTEM_CONSOLE
			if (!is_using_con_wrapper()) {
				return STD_HANDLE_INVALID; // Window app can't read stdin input without werapper.
			}
#endif
			return STD_HANDLE_CONSOLE;
		}
	}
}

OS_Windows::StdHandleType OS_Windows::get_stdout_type() const {
	HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
	if (h == 0 || h == INVALID_HANDLE_VALUE) {
		return STD_HANDLE_INVALID;
	}
	DWORD ftype = GetFileType(h);
	if (ftype == FILE_TYPE_UNKNOWN && GetLastError() != ERROR_SUCCESS) {
		return STD_HANDLE_UNKNOWN;
	}
	ftype &= ~(FILE_TYPE_REMOTE);

	if (ftype == FILE_TYPE_DISK) {
		return STD_HANDLE_FILE;
	} else if (ftype == FILE_TYPE_PIPE) {
		return STD_HANDLE_PIPE;
	} else {
		DWORD conmode = 0;
		BOOL res = GetConsoleMode(h, &conmode);
		if (!res && (GetLastError() == ERROR_INVALID_HANDLE)) {
			return STD_HANDLE_UNKNOWN; // Unknown character device.
		} else {
			return STD_HANDLE_CONSOLE;
		}
	}
}

OS_Windows::StdHandleType OS_Windows::get_stderr_type() const {
	HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
	if (h == 0 || h == INVALID_HANDLE_VALUE) {
		return STD_HANDLE_INVALID;
	}
	DWORD ftype = GetFileType(h);
	if (ftype == FILE_TYPE_UNKNOWN && GetLastError() != ERROR_SUCCESS) {
		return STD_HANDLE_UNKNOWN;
	}
	ftype &= ~(FILE_TYPE_REMOTE);

	if (ftype == FILE_TYPE_DISK) {
		return STD_HANDLE_FILE;
	} else if (ftype == FILE_TYPE_PIPE) {
		return STD_HANDLE_PIPE;
	} else {
		DWORD conmode = 0;
		BOOL res = GetConsoleMode(h, &conmode);
		if (!res && (GetLastError() == ERROR_INVALID_HANDLE)) {
			return STD_HANDLE_UNKNOWN; // Unknown character device.
		} else {
			return STD_HANDLE_CONSOLE;
		}
	}
}

Error OS_Windows::shell_open(const String &p_uri) {
	INT_PTR ret = (INT_PTR)ShellExecuteW(nullptr, nullptr, (LPCWSTR)(p_uri.utf16().get_data()), nullptr, nullptr, SW_SHOWNORMAL);
	if (ret > 32) {
		return OK;
	} else {
		switch (ret) {
			case ERROR_FILE_NOT_FOUND:
			case SE_ERR_DLLNOTFOUND:
				return ERR_FILE_NOT_FOUND;
			case ERROR_PATH_NOT_FOUND:
				return ERR_FILE_BAD_PATH;
			case ERROR_BAD_FORMAT:
				return ERR_FILE_CORRUPT;
			case SE_ERR_ACCESSDENIED:
				return ERR_UNAUTHORIZED;
			case 0:
			case SE_ERR_OOM:
				return ERR_OUT_OF_MEMORY;
			default:
				return FAILED;
		}
	}
}

Error OS_Windows::shell_show_in_file_manager(String p_path, bool p_open_folder) {
	bool open_folder = false;
	if (DirAccess::dir_exists_absolute(p_path) && p_open_folder) {
		open_folder = true;
	}

	if (!p_path.is_quoted()) {
		p_path = p_path.quote();
	}
	p_path = fix_path(p_path);

	INT_PTR ret = OK;
	if (open_folder) {
		ret = (INT_PTR)ShellExecuteW(nullptr, nullptr, L"explorer.exe", LPCWSTR(p_path.utf16().get_data()), nullptr, SW_SHOWNORMAL);
	} else {
		ret = (INT_PTR)ShellExecuteW(nullptr, nullptr, L"explorer.exe", LPCWSTR((String("/select,") + p_path).utf16().get_data()), nullptr, SW_SHOWNORMAL);
	}

	if (ret > 32) {
		return OK;
	} else {
		switch (ret) {
			case ERROR_FILE_NOT_FOUND:
			case SE_ERR_DLLNOTFOUND:
				return ERR_FILE_NOT_FOUND;
			case ERROR_PATH_NOT_FOUND:
				return ERR_FILE_BAD_PATH;
			case ERROR_BAD_FORMAT:
				return ERR_FILE_CORRUPT;
			case SE_ERR_ACCESSDENIED:
				return ERR_UNAUTHORIZED;
			case 0:
			case SE_ERR_OOM:
				return ERR_OUT_OF_MEMORY;
			default:
				return FAILED;
		}
	}
}

String OS_Windows::get_locale() const {
	const _WinLocale *wl = &_win_locales[0];

	LANGID langid = GetUserDefaultUILanguage();
	String neutral;
	int lang = PRIMARYLANGID(langid);
	int sublang = SUBLANGID(langid);

	while (wl->locale) {
		if (wl->main_lang == lang && wl->sublang == SUBLANG_NEUTRAL) {
			neutral = wl->locale;
		}

		if (lang == wl->main_lang && sublang == wl->sublang) {
			return String(wl->locale).replace_char('-', '_');
		}

		wl++;
	}

	if (!neutral.is_empty()) {
		return String(neutral).replace_char('-', '_');
	}

	return "en";
}

String OS_Windows::get_model_name() const {
	HKEY hkey;
	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"Hardware\\Description\\System\\BIOS", 0, KEY_QUERY_VALUE, &hkey) != ERROR_SUCCESS) {
		return OS::get_model_name();
	}

	String sys_name;
	String board_name;
	WCHAR buffer[256];
	DWORD buffer_len = 256;
	DWORD vtype = REG_SZ;
	if (RegQueryValueExW(hkey, L"SystemProductName", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) == ERROR_SUCCESS && buffer_len != 0) {
		sys_name = String::utf16((const char16_t *)buffer, buffer_len).strip_edges();
	}
	buffer_len = 256;
	if (RegQueryValueExW(hkey, L"BaseBoardProduct", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) == ERROR_SUCCESS && buffer_len != 0) {
		board_name = String::utf16((const char16_t *)buffer, buffer_len).strip_edges();
	}
	RegCloseKey(hkey);
	if (!sys_name.is_empty() && sys_name.to_lower() != "system product name") {
		return sys_name;
	}
	if (!board_name.is_empty() && board_name.to_lower() != "base board product") {
		return board_name;
	}
	return OS::get_model_name();
}

String OS_Windows::get_processor_name() const {
	const String id = "Hardware\\Description\\System\\CentralProcessor\\0";

	HKEY hkey;
	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, (LPCWSTR)(id.utf16().get_data()), 0, KEY_QUERY_VALUE, &hkey) != ERROR_SUCCESS) {
		ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
	}

	WCHAR buffer[256];
	DWORD buffer_len = 256;
	DWORD vtype = REG_SZ;
	if (RegQueryValueExW(hkey, L"ProcessorNameString", nullptr, &vtype, (LPBYTE)buffer, &buffer_len) == ERROR_SUCCESS) {
		RegCloseKey(hkey);
		return String::utf16((const char16_t *)buffer, buffer_len).strip_edges();
	} else {
		RegCloseKey(hkey);
		ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
	}
}

void OS_Windows::run() {
	if (!main_loop) {
		return;
	}

	main_loop->initialize();

	while (true) {
		GodotProfileFrameMark;
		GodotProfileZone("OS_Windows::run");
		DisplayServer::get_singleton()->process_events(); // get rid of pending events
		if (Main::iteration()) {
			break;
		}
	}

	main_loop->finalize();
}

MainLoop *OS_Windows::get_main_loop() const {
	return main_loop;
}

uint64_t OS_Windows::get_embedded_pck_offset() const {
	Ref<FileAccess> f = FileAccess::open(get_executable_path(), FileAccess::READ);
	if (f.is_null()) {
		return 0;
	}

	// Process header.
	{
		f->seek(0x3c);
		uint32_t pe_pos = f->get_32();

		f->seek(pe_pos);
		uint32_t magic = f->get_32();
		if (magic != 0x00004550) {
			return 0;
		}
	}

	int num_sections;
	{
		int64_t header_pos = f->get_position();

		f->seek(header_pos + 2);
		num_sections = f->get_16();
		f->seek(header_pos + 16);
		uint16_t opt_header_size = f->get_16();

		// Skip rest of header + optional header to go to the section headers.
		f->seek(f->get_position() + 2 + opt_header_size);
	}
	int64_t section_table_pos = f->get_position();

	// Search for the "pck" section.
	int64_t off = 0;
	for (int i = 0; i < num_sections; ++i) {
		int64_t section_header_pos = section_table_pos + i * 40;
		f->seek(section_header_pos);

		uint8_t section_name[9];
		f->get_buffer(section_name, 8);
		section_name[8] = '\0';

		if (strcmp((char *)section_name, "pck") == 0) {
			f->seek(section_header_pos + 20);
			off = f->get_32();
			break;
		}
	}

	return off;
}

String OS_Windows::get_config_path() const {
	if (has_environment("APPDATA")) {
		return get_environment("APPDATA").replace_char('\\', '/');
	}
	return ".";
}

String OS_Windows::get_data_path() const {
	return get_config_path();
}

String OS_Windows::get_cache_path() const {
	static String cache_path_cache;
	if (cache_path_cache.is_empty()) {
		if (has_environment("LOCALAPPDATA")) {
			cache_path_cache = get_environment("LOCALAPPDATA").replace_char('\\', '/');
		}
		if (cache_path_cache.is_empty()) {
			cache_path_cache = get_temp_path();
		}
	}
	return cache_path_cache;
}

String OS_Windows::get_temp_path() const {
	static String temp_path_cache;
	if (temp_path_cache.is_empty()) {
		{
			Vector<WCHAR> temp_path;
			// The maximum possible size is MAX_PATH+1 (261) + terminating null character.
			temp_path.resize(MAX_PATH + 2);
			DWORD temp_path_length = GetTempPathW(temp_path.size(), temp_path.ptrw());
			if (temp_path_length > 0 && temp_path_length < temp_path.size()) {
				temp_path_cache = String::utf16((const char16_t *)temp_path.ptr());
				// Let's try to get the long path instead of the short path (with tildes ~).
				DWORD temp_path_long_length = GetLongPathNameW(temp_path.ptr(), temp_path.ptrw(), temp_path.size());
				if (temp_path_long_length > 0 && temp_path_long_length < temp_path.size()) {
					temp_path_cache = String::utf16((const char16_t *)temp_path.ptr());
				}
			}
		}
		if (temp_path_cache.is_empty()) {
			temp_path_cache = get_config_path();
		}
	}
	return temp_path_cache.replace_char('\\', '/').trim_suffix("/");
}

// Get properly capitalized engine name for system paths
String OS_Windows::get_godot_dir_name() const {
	return String(GODOT_VERSION_SHORT_NAME).capitalize();
}

String OS_Windows::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	KNOWNFOLDERID id;

	switch (p_dir) {
		case SYSTEM_DIR_DESKTOP: {
			id = FOLDERID_Desktop;
		} break;
		case SYSTEM_DIR_DCIM: {
			id = FOLDERID_Pictures;
		} break;
		case SYSTEM_DIR_DOCUMENTS: {
			id = FOLDERID_Documents;
		} break;
		case SYSTEM_DIR_DOWNLOADS: {
			id = FOLDERID_Downloads;
		} break;
		case SYSTEM_DIR_MOVIES: {
			id = FOLDERID_Videos;
		} break;
		case SYSTEM_DIR_MUSIC: {
			id = FOLDERID_Music;
		} break;
		case SYSTEM_DIR_PICTURES: {
			id = FOLDERID_Pictures;
		} break;
		case SYSTEM_DIR_RINGTONES: {
			id = FOLDERID_Music;
		} break;
	}

	PWSTR szPath;
	HRESULT res = SHGetKnownFolderPath(id, 0, nullptr, &szPath);
	ERR_FAIL_COND_V(res != S_OK, String());
	String path = String::utf16((const char16_t *)szPath).replace_char('\\', '/');
	CoTaskMemFree(szPath);
	return path;
}

String OS_Windows::get_user_data_dir(const String &p_user_dir) const {
	return get_data_path().path_join(p_user_dir).replace_char('\\', '/');
}

String OS_Windows::get_unique_id() const {
	HW_PROFILE_INFOA HwProfInfo;
	ERR_FAIL_COND_V(!GetCurrentHwProfileA(&HwProfInfo), "");

	// Note: Windows API returns a GUID with null termination.
	return String::ascii(Span<char>(HwProfInfo.szHwProfileGuid, strnlen(HwProfInfo.szHwProfileGuid, HW_PROFILE_GUIDLEN)));
}

bool OS_Windows::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "system_fonts") {
		return dwrite_init;
	}
	if (p_feature == "pc") {
		return true;
	}

	return false;
}

void OS_Windows::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_Windows::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}

Error OS_Windows::move_to_trash(const String &p_path) {
	SHFILEOPSTRUCTW sf;

	Char16String utf16 = p_path.utf16();
	WCHAR *from = new WCHAR[utf16.length() + 2];
	wcscpy_s(from, utf16.length() + 1, (LPCWSTR)(utf16.get_data()));
	from[utf16.length() + 1] = 0;

	sf.hwnd = main_window;
	sf.wFunc = FO_DELETE;
	sf.pFrom = from;
	sf.pTo = nullptr;
	sf.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION;
	sf.fAnyOperationsAborted = FALSE;
	sf.hNameMappings = nullptr;
	sf.lpszProgressTitle = nullptr;

	int ret = SHFileOperationW(&sf);
	delete[] from;

	if (ret) {
		ERR_PRINT("SHFileOperation error: " + itos(ret));
		return FAILED;
	}

	return OK;
}

String OS_Windows::get_system_ca_certificates() {
	HCERTSTORE cert_store = CertOpenSystemStoreA(0, "ROOT");
	ERR_FAIL_NULL_V_MSG(cert_store, "", "Failed to read the root certificate store.");

	FILETIME curr_time;
	GetSystemTimeAsFileTime(&curr_time);

	String certs;
	PCCERT_CONTEXT curr = CertEnumCertificatesInStore(cert_store, nullptr);
	while (curr) {
		FILETIME ft;
		DWORD size = sizeof(ft);
		// Check if the certificate is disallowed.
		if (CertGetCertificateContextProperty(curr, CERT_DISALLOWED_FILETIME_PROP_ID, &ft, &size) && CompareFileTime(&curr_time, &ft) != -1) {
			curr = CertEnumCertificatesInStore(cert_store, curr);
			continue;
		}
		// Encode and add to certificate list.
		bool success = CryptBinaryToStringA(curr->pbCertEncoded, curr->cbCertEncoded, CRYPT_STRING_BASE64HEADER | CRYPT_STRING_NOCR, nullptr, &size);
		ERR_CONTINUE(!success);
		PackedByteArray pba;
		pba.resize(size + 1);
		CryptBinaryToStringA(curr->pbCertEncoded, curr->cbCertEncoded, CRYPT_STRING_BASE64HEADER | CRYPT_STRING_NOCR, (char *)pba.ptrw(), &size);
		pba.write[size] = 0;
		certs += String::ascii(Span((const char *)pba.ptr(), strlen((const char *)pba.ptr())));
		curr = CertEnumCertificatesInStore(cert_store, curr);
	}
	CertCloseStore(cert_store, 0);
	return certs;
}

void OS_Windows::add_frame_delay(bool p_can_draw, bool p_wake_for_events) {
	if (p_wake_for_events) {
		uint64_t delay = get_frame_delay(p_can_draw);
		if (delay == 0) {
			return;
		}

		DisplayServer *ds = DisplayServer::get_singleton();
		DisplayServerWindows *ds_win = Object::cast_to<DisplayServerWindows>(ds);
		if (ds_win) {
			MsgWaitForMultipleObjects(0, nullptr, false, Math::floor(double(delay) / 1000.0), QS_ALLINPUT);
			return;
		}
	}

	const uint32_t frame_delay = Engine::get_singleton()->get_frame_delay();
	if (frame_delay) {
		// Add fixed frame delay to decrease CPU/GPU usage. This doesn't take
		// the actual frame time into account.
		// Due to the high fluctuation of the actual sleep duration, it's not recommended
		// to use this as a FPS limiter.
		delay_usec(frame_delay * 1000);
	}

	// Add a dynamic frame delay to decrease CPU/GPU usage. This takes the
	// previous frame time into account for a smoother result.
	uint64_t dynamic_delay = 0;
	if (is_in_low_processor_usage_mode() || !p_can_draw) {
		dynamic_delay = get_low_processor_usage_mode_sleep_usec();
	}
	const int max_fps = Engine::get_singleton()->get_max_fps();
	if (max_fps > 0 && !Engine::get_singleton()->is_editor_hint()) {
		// Override the low processor usage mode sleep delay if the target FPS is lower.
		dynamic_delay = MAX(dynamic_delay, (uint64_t)(1000000 / max_fps));
	}

	if (dynamic_delay > 0) {
		target_ticks += dynamic_delay;
		uint64_t current_ticks = get_ticks_usec();

		if (!is_in_low_processor_usage_mode()) {
			if (target_ticks > current_ticks + delay_resolution) {
				uint64_t delay_time = target_ticks - current_ticks - delay_resolution;
				// Make sure we always sleep for a multiple of delay_resolution to avoid overshooting.
				// Refer to: https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-sleep#remarks
				delay_time = (delay_time / delay_resolution) * delay_resolution;
				if (delay_time > 0) {
					delay_usec(delay_time);
				}
			}
			// Busy wait for the remainder of time.
			while (get_ticks_usec() < target_ticks) {
				YieldProcessor();
			}
		} else {
			// Use a more relaxed approach for low processor usage mode.
			// This has worse frame pacing but is more power efficient.
			if (current_ticks < target_ticks) {
				delay_usec(target_ticks - current_ticks);
			}
		}

		current_ticks = get_ticks_usec();
		target_ticks = MIN(MAX(target_ticks, current_ticks - dynamic_delay), current_ticks + dynamic_delay);
	}
}

#ifdef TOOLS_ENABLED
bool OS_Windows::_test_create_rendering_device(const String &p_display_driver) const {
	// Tests Rendering Device creation.

	bool ok = false;
#if defined(RD_ENABLED)
	Error err;
	RenderingContextDriver *rcd = nullptr;

#if defined(VULKAN_ENABLED)
	rcd = memnew(RenderingContextDriverVulkan);
#endif
#ifdef D3D12_ENABLED
	if (rcd == nullptr) {
		rcd = memnew(RenderingContextDriverD3D12);
	}
#endif
	if (rcd != nullptr) {
		err = rcd->initialize();
		if (err == OK) {
			RenderingDevice *rd = memnew(RenderingDevice);
			err = rd->initialize(rcd);
			memdelete(rd);
			rd = nullptr;
			if (err == OK) {
				ok = true;
			}
		}
		memdelete(rcd);
		rcd = nullptr;
	}
#endif

	return ok;
}

bool OS_Windows::_test_create_rendering_device_and_gl(const String &p_display_driver) const {
	// Tests OpenGL context and Rendering Device simultaneous creation. This function is expected to crash on some NVIDIA drivers.

	WNDCLASSEXW wc_probe;
	memset(&wc_probe, 0, sizeof(WNDCLASSEXW));
	wc_probe.cbSize = sizeof(WNDCLASSEXW);
	wc_probe.style = CS_OWNDC | CS_DBLCLKS;
	wc_probe.lpfnWndProc = (WNDPROC)::DefWindowProcW;
	wc_probe.cbClsExtra = 0;
	wc_probe.cbWndExtra = 0;
	wc_probe.hInstance = GetModuleHandle(nullptr);
	wc_probe.hIcon = LoadIcon(nullptr, IDI_WINLOGO);
	wc_probe.hCursor = nullptr;
	wc_probe.hbrBackground = nullptr;
	wc_probe.lpszMenuName = nullptr;
	wc_probe.lpszClassName = L"Engine probe window";

	if (!RegisterClassExW(&wc_probe)) {
		return false;
	}

	HWND hWnd = CreateWindowExW(WS_EX_WINDOWEDGE, L"Engine probe window", L"", WS_OVERLAPPEDWINDOW, 0, 0, 800, 600, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
	if (!hWnd) {
		UnregisterClassW(L"Engine probe window", GetModuleHandle(nullptr));
		return false;
	}

	bool ok = true;
#ifdef GLES3_ENABLED
	GLManagerNative_Windows *test_gl_manager_native = memnew(GLManagerNative_Windows);
	if (test_gl_manager_native->window_create(DisplayServer::MAIN_WINDOW_ID, hWnd, GetModuleHandle(nullptr), 800, 600) == OK) {
		RasterizerGLES3::make_current(true);
	} else {
		ok = false;
	}
#endif

	MSG msg = {};
	while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}

	if (ok) {
		ok = _test_create_rendering_device(p_display_driver);
	}

#ifdef GLES3_ENABLED
	if (test_gl_manager_native) {
		memdelete(test_gl_manager_native);
	}
#endif

	DestroyWindow(hWnd);
	UnregisterClassW(L"Engine probe window", GetModuleHandle(nullptr));
	return ok;
}
#endif

using GetProcAddressType = FARPROC(__stdcall *)(HMODULE, LPCSTR);
GetProcAddressType Original_GetProcAddress = nullptr;

using HidD_GetProductStringType = BOOLEAN(__stdcall *)(HANDLE, void *, ULONG);
HidD_GetProductStringType Original_HidD_GetProductString = nullptr;

#ifndef HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER
#define HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER 0x08
#endif

bool _hid_is_controller(HANDLE p_hid_handle) {
	PHIDP_PREPARSED_DATA hid_preparsed = nullptr;
	BOOLEAN preparsed_res = HidD_GetPreparsedData(p_hid_handle, &hid_preparsed);
	if (!preparsed_res) {
		return false;
	}

	HIDP_CAPS hid_caps = {};
	NTSTATUS caps_res = HidP_GetCaps(hid_preparsed, &hid_caps);
	HidD_FreePreparsedData(hid_preparsed);
	if (caps_res != HIDP_STATUS_SUCCESS) {
		return false;
	}

	if (hid_caps.UsagePage != HID_USAGE_PAGE_GENERIC) {
		return false;
	}

	if (hid_caps.Usage == HID_USAGE_GENERIC_JOYSTICK || hid_caps.Usage == HID_USAGE_GENERIC_GAMEPAD || hid_caps.Usage == HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER) {
		return true;
	}

	return false;
}

BOOLEAN __stdcall Hook_HidD_GetProductString(HANDLE p_object, void *p_buffer, ULONG p_buffer_length) {
	constexpr const wchar_t unknown_product_string[] = L"Unknown HID Device";
	constexpr size_t unknown_product_length = sizeof(unknown_product_string);

	if (_hid_is_controller(p_object)) {
		return HidD_GetProductString(p_object, p_buffer, p_buffer_length);
	}

	// The HID is (probably) not a controller, so we don't care about returning its actual product string.
	// This avoids stalls on `EnumDevices` because DirectInput attempts to enumerate all HIDs, including some DACs
	// and other devices which take too long to respond to those requests, added to the lack of a shorter timeout.
	if (p_buffer_length >= unknown_product_length) {
		memcpy(p_buffer, unknown_product_string, unknown_product_length);
		return TRUE;
	}
	return FALSE;
}

FARPROC __stdcall Hook_GetProcAddress(HMODULE p_module, LPCSTR p_name) {
	if (String(p_name) == "HidD_GetProductString") {
		return (FARPROC)(LPVOID)Hook_HidD_GetProductString;
	}
	if (Original_GetProcAddress) {
		return Original_GetProcAddress(p_module, p_name);
	}
	return nullptr;
}

LPVOID install_iat_hook(const String &p_target, const String &p_module, const String &p_symbol, LPVOID p_hook_func) {
	LPVOID image_base = LoadLibraryA(p_target.ascii().get_data());
	if (image_base) {
		PIMAGE_NT_HEADERS nt_headers = (PIMAGE_NT_HEADERS)((DWORD_PTR)image_base + ((PIMAGE_DOS_HEADER)image_base)->e_lfanew);
		PIMAGE_IMPORT_DESCRIPTOR import_descriptor = (PIMAGE_IMPORT_DESCRIPTOR)((DWORD_PTR)image_base + nt_headers->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress);
		while (import_descriptor->Name != 0) {
			LPCSTR library_name = (LPCSTR)((DWORD_PTR)image_base + import_descriptor->Name);
			if (String(library_name).to_lower() == p_module) {
				PIMAGE_THUNK_DATA original_first_thunk = (PIMAGE_THUNK_DATA)((DWORD_PTR)image_base + import_descriptor->OriginalFirstThunk);
				PIMAGE_THUNK_DATA first_thunk = (PIMAGE_THUNK_DATA)((DWORD_PTR)image_base + import_descriptor->FirstThunk);

				while ((LPVOID)original_first_thunk->u1.AddressOfData != nullptr) {
					PIMAGE_IMPORT_BY_NAME function_import = (PIMAGE_IMPORT_BY_NAME)((DWORD_PTR)image_base + original_first_thunk->u1.AddressOfData);
					if (String(function_import->Name).to_lower() == p_symbol.to_lower()) {
						DWORD old_protect = 0;
						VirtualProtect((LPVOID)(&first_thunk->u1.Function), 8, PAGE_READWRITE, &old_protect);

						LPVOID old_func = (LPVOID)first_thunk->u1.Function;
						first_thunk->u1.Function = (DWORD_PTR)p_hook_func;

						VirtualProtect((LPVOID)(&first_thunk->u1.Function), 8, old_protect, nullptr);
						return old_func;
					}
					original_first_thunk++;
					first_thunk++;
				}
			}
			import_descriptor++;
		}
	}
	return nullptr;
}

OS_Windows::OS_Windows(HINSTANCE _hInstance) {
	hInstance = _hInstance;

	Original_GetProcAddress = (GetProcAddressType)install_iat_hook("dinput8.dll", "kernel32.dll", "GetProcAddress", (LPVOID)Hook_GetProcAddress);
	Original_HidD_GetProductString = (HidD_GetProductStringType)install_iat_hook("dinput8.dll", "hid.dll", "HidD_GetProductString", (LPVOID)Hook_HidD_GetProductString);

	_init_encodings();

	// Reset CWD to ensure long path is used.
	Char16String current_dir_name;
	size_t str_len = GetCurrentDirectoryW(0, nullptr);
	current_dir_name.resize_uninitialized(str_len + 1);
	GetCurrentDirectoryW(current_dir_name.size(), (LPWSTR)current_dir_name.ptrw());

	Char16String new_current_dir_name;
	str_len = GetLongPathNameW((LPCWSTR)current_dir_name.get_data(), nullptr, 0);
	new_current_dir_name.resize_uninitialized(str_len + 1);
	GetLongPathNameW((LPCWSTR)current_dir_name.get_data(), (LPWSTR)new_current_dir_name.ptrw(), new_current_dir_name.size());

	SetCurrentDirectoryW((LPCWSTR)new_current_dir_name.get_data());

#ifndef WINDOWS_SUBSYSTEM_CONSOLE
	RedirectIOToConsole();
#endif

	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);

	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

#ifdef WASAPI_ENABLED
	AudioDriverManager::add_driver(&driver_wasapi);
#endif
#ifdef XAUDIO2_ENABLED
	AudioDriverManager::add_driver(&driver_xaudio2);
#endif

	DisplayServerWindows::register_windows_driver();

	// Enable ANSI escape code support on Windows 10 v1607 (Anniversary Update) and later.
	// This lets the engine and projects use ANSI escape codes to color text just like on macOS and Linux.
	//
	// NOTE: The engine does not use ANSI escape codes to color error/warning messages; it uses Windows API calls instead.
	// Therefore, error/warning messages are still colored on Windows versions older than 10.
	HANDLE stdoutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD outMode = 0;
	GetConsoleMode(stdoutHandle, &outMode);
	outMode |= ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(stdoutHandle, outMode)) {
		// Windows 10 prior to Anniversary Update.
		print_verbose("Can't set the ENABLE_VIRTUAL_TERMINAL_PROCESSING Windows console mode. `print_rich()` will not work as expected.");
	}

	Vector<Logger *> loggers;
	loggers.push_back(memnew(WindowsTerminalLogger));
	_set_logger(memnew(CompositeLogger(loggers)));
}

OS_Windows::~OS_Windows() {
	CoUninitialize();
}
