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
#include "joypad_windows.h"
#include "lang_table.h"
#include "windows_terminal_logger.h"
#include "windows_utils.h"

#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/io/marshalls.h"
#include "core/version_generated.gen.h"
#include "drivers/unix/net_socket_posix.h"
#include "drivers/windows/dir_access_windows.h"
#include "drivers/windows/file_access_windows.h"
#include "drivers/windows/file_access_windows_pipe.h"
#include "main/main.h"
#include "servers/audio_server.h"
#include "servers/rendering/rendering_server_default.h"
#include "servers/text_server.h"

#include <avrt.h>
#include <bcrypt.h>
#include <direct.h>
#include <knownfolders.h>
#include <process.h>
#include <psapi.h>
#include <regstr.h>
#include <shlobj.h>
#include <wbemcli.h>
#include <wincrypt.h>

#ifdef DEBUG_ENABLED
#pragma pack(push, before_imagehlp, 8)
#include <imagehlp.h>
#pragma pack(pop, before_imagehlp)
#endif

extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
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

#if defined(__GNUC__)
// Workaround GCC warning from -Wcast-function-type.
#define GetProcAddress (void *)GetProcAddress
#endif

static String format_error_message(DWORD id) {
	LPWSTR messageBuffer = nullptr;
	size_t size = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&messageBuffer, 0, nullptr);

	String msg = "Error " + itos(id) + ": " + String::utf16((const char16_t *)messageBuffer, size);

	LocalFree(messageBuffer);

	return msg.replace("\r", "").replace("\n", "");
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
	if (AttachConsole(ATTACH_PARENT_PROCESS)) {
		RedirectStream("CONIN$", "r", stdin, STD_INPUT_HANDLE);
		RedirectStream("CONOUT$", "w", stdout, STD_OUTPUT_HANDLE);
		RedirectStream("CONOUT$", "w", stderr, STD_ERROR_HANDLE);
	}
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
		err_str = String::utf8(p_errorexp);
	} else {
		err_str = String::utf8(p_file) + ":" + itos(p_line) + " - " + String::utf8(p_error);
	}

	if (p_editor_notify) {
		err_str += " (User)\n";
	} else {
		err_str += "\n";
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

#ifndef WINDOWS_SUBSYSTEM_CONSOLE
	RedirectIOToConsole();
#endif

	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_FILESYSTEM);
	FileAccess::make_default<FileAccessWindowsPipe>(FileAccess::ACCESS_PIPE);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_FILESYSTEM);

	NetSocketPosix::make_default();

	// We need to know how often the clock is updated
	QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_second);
	QueryPerformanceCounter((LARGE_INTEGER *)&ticks_start);

	// set minimum resolution for periodic timers, otherwise Sleep(n) may wait at least as
	//  long as the windows scheduler resolution (~16-30ms) even for calls like Sleep(1)
	timeBeginPeriod(1);

	process_map = memnew((HashMap<ProcessID, ProcessInfo>));

	// Add current Godot PID to the list of known PIDs
	ProcessInfo current_pi = {};
	PROCESS_INFORMATION current_pi_pi = {};
	current_pi.pi = current_pi_pi;
	current_pi.pi.hProcess = GetCurrentProcess();
	process_map->insert(GetCurrentProcessId(), current_pi);

	IPUnix::make_default();
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

	timeEndPeriod(1);

	memdelete(process_map);
	NetSocketPosix::cleanup();

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
void debug_dynamic_library_check_dependencies(const String &p_root_path, const String &p_path, HashSet<String> &r_checked, HashSet<String> &r_missing) {
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
							char16_t full_name_wc[MAX_PATH];
							const char *name_cs = (const char *)ImageRvaToVa(loaded_image.FileHeader, loaded_image.MappedAddress, import_desc->Name, nullptr);
							String name = String(name_cs);
							if (name.begins_with("api-ms-win-")) {
								r_checked.insert(name);
							} else if (SearchPathW(nullptr, (LPCWSTR)name.utf16().get_data(), nullptr, MAX_PATH, (LPWSTR)full_name_wc, nullptr)) {
								debug_dynamic_library_check_dependencies(p_root_path, String::utf16(full_name_wc), r_checked, r_missing);
							} else if (SearchPathW((LPCWSTR)(p_path.get_base_dir().utf16().get_data()), (LPCWSTR)name.utf16().get_data(), nullptr, MAX_PATH, (LPWSTR)full_name_wc, nullptr)) {
								debug_dynamic_library_check_dependencies(p_root_path, String::utf16(full_name_wc), r_checked, r_missing);
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
	String path = p_path.replace("/", "\\");

	if (!FileAccess::exists(path)) {
		//this code exists so gdextension can load .dll files from within the executable path
		path = get_executable_path().get_base_dir().path_join(p_path.get_file());
	}

	ERR_FAIL_COND_V(!FileAccess::exists(path), ERR_FILE_NOT_FOUND);

	// Here we want a copy to be loaded.
	// This is so the original file isn't locked and can be updated by a compiler.
	if (p_data != nullptr && p_data->generate_temp_files) {
		// Copy the file to the same directory as the original with a prefix in the name.
		// This is so relative path to dependencies are satisfied.
		String copy_path = path.get_base_dir().path_join("~" + path.get_file());

		// If there's a left-over copy (possibly from a crash) then delete it first.
		if (FileAccess::exists(copy_path)) {
			DirAccess::remove_absolute(copy_path);
		}

		Error copy_err = DirAccess::copy_absolute(path, copy_path);
		if (copy_err) {
			ERR_PRINT("Error copying library: " + path);
			return ERR_CANT_CREATE;
		}

		FileAccess::set_hidden_attribute(copy_path, true);

		// Save the copied path so it can be deleted later.
		path = copy_path;

		Error pdb_err = WindowsUtils::copy_and_rename_pdb(path);
		if (pdb_err != OK && pdb_err != ERR_SKIP) {
			WARN_PRINT(vformat("Failed to rename the PDB file. The original PDB file for '%s' will be loaded.", path));
		}
	}

	typedef DLL_DIRECTORY_COOKIE(WINAPI * PAddDllDirectory)(PCWSTR);
	typedef BOOL(WINAPI * PRemoveDllDirectory)(DLL_DIRECTORY_COOKIE);

	PAddDllDirectory add_dll_directory = (PAddDllDirectory)GetProcAddress(GetModuleHandle("kernel32.dll"), "AddDllDirectory");
	PRemoveDllDirectory remove_dll_directory = (PRemoveDllDirectory)GetProcAddress(GetModuleHandle("kernel32.dll"), "RemoveDllDirectory");

	bool has_dll_directory_api = ((add_dll_directory != nullptr) && (remove_dll_directory != nullptr));
	DLL_DIRECTORY_COOKIE cookie = nullptr;

	if (p_data != nullptr && p_data->also_set_library_path && has_dll_directory_api) {
		cookie = add_dll_directory((LPCWSTR)(path.get_base_dir().utf16().get_data()));
	}

	p_library_handle = (void *)LoadLibraryExW((LPCWSTR)(path.utf16().get_data()), nullptr, (p_data != nullptr && p_data->also_set_library_path && has_dll_directory_api) ? LOAD_LIBRARY_SEARCH_DEFAULT_DIRS : 0);
	if (!p_library_handle) {
		if (p_data != nullptr && p_data->generate_temp_files) {
			DirAccess::remove_absolute(path);
		}

#ifdef DEBUG_ENABLED
		DWORD err_code = GetLastError();

		HashSet<String> checekd_libs;
		HashSet<String> missing_libs;
		debug_dynamic_library_check_dependencies(path, path, checekd_libs, missing_libs);
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
		remove_dll_directory(cookie);
	}

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = path;
	}

	if (p_data != nullptr && p_data->generate_temp_files) {
		temp_libraries[p_library_handle] = path;
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
	RtlGetVersionPtr version_ptr = (RtlGetVersionPtr)GetProcAddress(GetModuleHandle("ntdll.dll"), "RtlGetVersion");
	if (version_ptr != nullptr) {
		RTL_OSVERSIONINFOW fow;
		ZeroMemory(&fow, sizeof(fow));
		fow.dwOSVersionInfoSize = sizeof(fow);
		if (version_ptr(&fow) == 0x00000000) {
			return vformat("%d.%d.%d", (int64_t)fow.dwMajorVersion, (int64_t)fow.dwMinorVersion, (int64_t)fow.dwBuildNumber);
		}
	}
	return "";
}

Vector<String> OS_Windows::get_video_adapter_driver_info() const {
	if (RenderingServer::get_singleton() == nullptr) {
		return Vector<String>();
	}

	static Vector<String> info;
	if (!info.is_empty()) {
		return info;
	}

	REFCLSID clsid = CLSID_WbemLocator; // Unmarshaler CLSID
	REFIID uuid = IID_IWbemLocator; // Interface UUID
	IWbemLocator *wbemLocator = nullptr; // to get the services
	IWbemServices *wbemServices = nullptr; // to get the class
	IEnumWbemClassObject *iter = nullptr;
	IWbemClassObject *pnpSDriverObject[1]; // contains driver name, version, etc.
	String driver_name;
	String driver_version;

	const String device_name = RenderingServer::get_singleton()->get_video_adapter_name();
	if (device_name.is_empty()) {
		return Vector<String>();
	}

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

	const String gpu_device_class_query = vformat("SELECT * FROM Win32_PnPSignedDriver WHERE DeviceName = \"%s\"", device_name);
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
			if (hr == S_OK) {
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
				if (hr == S_OK) {
					driver_name = String(V_BSTR(&dn));
				}
			}

			VARIANT dv;
			VariantInit(&dv);
			object_name = SysAllocString(L"DriverVersion");
			hr = pnpSDriverObject[0]->Get(object_name, 0, &dv, nullptr, nullptr);
			SysFreeString(object_name);
			if (hr == S_OK) {
				driver_version = String(V_BSTR(&dv));
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
		(*r_pipe) += String(wchars.ptr(), total_wchars);
	}
	if (p_pipe_mutex) {
		p_pipe_mutex->unlock();
	}
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
	PGetCurrentThreadStackLimits GetCurrentThreadStackLimits = (PGetCurrentThreadStackLimits)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetCurrentThreadStackLimits");

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

Dictionary OS_Windows::execute_with_pipe(const String &p_path, const List<String> &p_arguments) {
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

	String path = p_path.replace("/", "\\");
	String command = _quote_command_line_argument(path);
	for (const String &E : p_arguments) {
		command += " " + _quote_command_line_argument(E);
	}

	// Create pipes.
	HANDLE pipe_in[2] = { 0, 0 };
	HANDLE pipe_out[2] = { 0, 0 };
	HANDLE pipe_err[2] = { 0, 0 };

	SECURITY_ATTRIBUTES sa;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.bInheritHandle = true;
	sa.lpSecurityDescriptor = nullptr;

	ERR_FAIL_COND_V(!CreatePipe(&pipe_in[0], &pipe_in[1], &sa, 0), ret);
	if (!SetHandleInformation(pipe_in[1], HANDLE_FLAG_INHERIT, 0)) {
		CLEAN_PIPES
		ERR_FAIL_V(ret);
	}
	if (!CreatePipe(&pipe_out[0], &pipe_out[1], &sa, 0)) {
		CLEAN_PIPES
		ERR_FAIL_V(ret);
	}
	if (!SetHandleInformation(pipe_out[0], HANDLE_FLAG_INHERIT, 0)) {
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
	pi.si.cb = sizeof(pi.si);
	ZeroMemory(&pi.pi, sizeof(pi.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&pi.si;

	pi.si.dwFlags |= STARTF_USESTDHANDLES;
	pi.si.hStdInput = pipe_in[0];
	pi.si.hStdOutput = pipe_out[1];
	pi.si.hStdError = pipe_err[1];

	DWORD creation_flags = NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW;

	if (!CreateProcessW(nullptr, (LPWSTR)(command.utf16().ptrw()), nullptr, nullptr, true, creation_flags, nullptr, nullptr, si_w, &pi.pi)) {
		CLEAN_PIPES
		ERR_FAIL_V_MSG(ret, "Could not create child process: " + command);
	}
	CloseHandle(pipe_in[0]);
	CloseHandle(pipe_out[1]);
	CloseHandle(pipe_err[1]);

	ProcessID pid = pi.pi.dwProcessId;
	process_map_mutex.lock();
	process_map->insert(pid, pi);
	process_map_mutex.unlock();

	Ref<FileAccessWindowsPipe> main_pipe;
	main_pipe.instantiate();
	main_pipe->open_existing(pipe_out[0], pipe_in[1]);

	Ref<FileAccessWindowsPipe> err_pipe;
	err_pipe.instantiate();
	err_pipe->open_existing(pipe_err[0], 0);

	ret["stdio"] = main_pipe;
	ret["stderr"] = err_pipe;
	ret["pid"] = pid;

#undef CLEAN_PIPES
	return ret;
}

Error OS_Windows::execute(const String &p_path, const List<String> &p_arguments, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
	String path = p_path.replace("/", "\\");
	String command = _quote_command_line_argument(path);
	for (const String &E : p_arguments) {
		command += " " + _quote_command_line_argument(E);
	}

	ProcessInfo pi;
	ZeroMemory(&pi.si, sizeof(pi.si));
	pi.si.cb = sizeof(pi.si);
	ZeroMemory(&pi.pi, sizeof(pi.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&pi.si;

	bool inherit_handles = false;
	HANDLE pipe[2] = { nullptr, nullptr };
	if (r_pipe) {
		// Create pipe for StdOut and StdErr.
		SECURITY_ATTRIBUTES sa;
		sa.nLength = sizeof(SECURITY_ATTRIBUTES);
		sa.bInheritHandle = true;
		sa.lpSecurityDescriptor = nullptr;

		ERR_FAIL_COND_V(!CreatePipe(&pipe[0], &pipe[1], &sa, 0), ERR_CANT_FORK);
		ERR_FAIL_COND_V(!SetHandleInformation(pipe[0], HANDLE_FLAG_INHERIT, 0), ERR_CANT_FORK); // Read handle is for host process only and should not be inherited.

		pi.si.dwFlags |= STARTF_USESTDHANDLES;
		pi.si.hStdOutput = pipe[1];
		if (read_stderr) {
			pi.si.hStdError = pipe[1];
		}
		inherit_handles = true;
	}
	DWORD creation_flags = NORMAL_PRIORITY_CLASS;
	if (p_open_console) {
		creation_flags |= CREATE_NEW_CONSOLE;
	} else {
		creation_flags |= CREATE_NO_WINDOW;
	}

	int ret = CreateProcessW(nullptr, (LPWSTR)(command.utf16().ptrw()), nullptr, nullptr, inherit_handles, creation_flags, nullptr, nullptr, si_w, &pi.pi);
	if (!ret && r_pipe) {
		CloseHandle(pipe[0]); // Cleanup pipe handles.
		CloseHandle(pipe[1]);
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

	return OK;
}

Error OS_Windows::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
	String path = p_path.replace("/", "\\");
	String command = _quote_command_line_argument(path);
	for (const String &E : p_arguments) {
		command += " " + _quote_command_line_argument(E);
	}

	ProcessInfo pi;
	ZeroMemory(&pi.si, sizeof(pi.si));
	pi.si.cb = sizeof(pi.si);
	ZeroMemory(&pi.pi, sizeof(pi.pi));
	LPSTARTUPINFOW si_w = (LPSTARTUPINFOW)&pi.si;

	DWORD creation_flags = NORMAL_PRIORITY_CLASS;
	if (p_open_console) {
		creation_flags |= CREATE_NEW_CONSOLE;
	} else {
		creation_flags |= CREATE_NO_WINDOW;
	}

	int ret = CreateProcessW(nullptr, (LPWSTR)(command.utf16().ptrw()), nullptr, nullptr, false, creation_flags, nullptr, nullptr, si_w, &pi.pi);
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

		name.resize(length + 1);
		hr = family_names->GetString(index, (WCHAR *)name.ptrw(), length + 1);
		ERR_CONTINUE(FAILED(hr));

		font_names.insert(String::utf16(name.ptr(), length));
	}

	for (const String &E : font_names) {
		ret.push_back(E);
	}
	return ret;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

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
	};

	virtual ~FallbackTextAnalysisSource() {}
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

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
	if (!dwrite2_init) {
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

		WCHAR file_path[MAX_PATH];
		hr = loader->GetFilePathFromKey(reference_key, reference_key_size, &file_path[0], MAX_PATH);
		if (FAILED(hr)) {
			continue;
		}
		String fpath = String::utf16((const char16_t *)&file_path[0]).replace("\\", "/");

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

		WCHAR file_path[MAX_PATH];
		hr = loader->GetFilePathFromKey(reference_key, reference_key_size, &file_path[0], MAX_PATH);
		if (FAILED(hr)) {
			continue;
		}
		String fpath = String::utf16((const char16_t *)&file_path[0]).replace("\\", "/");

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
	String s = String::utf16((const char16_t *)bufname).replace("\\", "/");
	return s;
}

bool OS_Windows::has_environment(const String &p_var) const {
#ifdef MINGW_ENABLED
	return _wgetenv((LPCWSTR)(p_var.utf16().get_data())) != nullptr;
#else
	WCHAR *env;
	size_t len;
	_wdupenv_s(&env, &len, (LPCWSTR)(p_var.utf16().get_data()));
	const bool has_env = env != nullptr;
	free(env);
	return has_env;
#endif
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
	ERR_FAIL_COND_MSG(p_var.is_empty() || p_var.contains("="), vformat("Invalid environment variable name '%s', cannot be empty or include '='.", p_var));
	Char16String var = p_var.utf16();
	Char16String value = p_value.utf16();
	ERR_FAIL_COND_MSG(var.length() + value.length() + 2 > 32767, vformat("Invalid definition for environment variable '%s', cannot exceed 32767 characters.", p_var));
	SetEnvironmentVariableW((LPCWSTR)(var.get_data()), (LPCWSTR)(value.get_data()));
}

void OS_Windows::unset_environment(const String &p_var) const {
	ERR_FAIL_COND_MSG(p_var.is_empty() || p_var.contains("="), vformat("Invalid environment variable name '%s', cannot be empty or include '='.", p_var));
	SetEnvironmentVariableW((LPCWSTR)(p_var.utf16().get_data()), nullptr); // Null to delete.
}

String OS_Windows::get_stdin_string() {
	WCHAR buff[1024];
	DWORD count = 0;
	if (ReadConsoleW(GetStdHandle(STD_INPUT_HANDLE), buff, 1024, &count, nullptr)) {
		return String::utf16((const char16_t *)buff, count);
	}

	return String();
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
	p_path = p_path.replace("/", "\\");

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
			return String(wl->locale).replace("-", "_");
		}

		wl++;
	}

	if (!neutral.is_empty()) {
		return String(neutral).replace("-", "_");
	}

	return "en";
}

// We need this because GetSystemInfo() is unreliable on WOW64
// see https://msdn.microsoft.com/en-us/library/windows/desktop/ms724381(v=vs.85).aspx
// Taken from MSDN
typedef BOOL(WINAPI *LPFN_ISWOW64PROCESS)(HANDLE, PBOOL);
LPFN_ISWOW64PROCESS fnIsWow64Process;

BOOL is_wow64() {
	BOOL wow64 = FALSE;

	fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

	if (fnIsWow64Process) {
		if (!fnIsWow64Process(GetCurrentProcess(), &wow64)) {
			wow64 = FALSE;
		}
	}

	return wow64;
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
		return get_environment("APPDATA").replace("\\", "/");
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
			cache_path_cache = get_environment("LOCALAPPDATA").replace("\\", "/");
		}
		if (cache_path_cache.is_empty() && has_environment("TEMP")) {
			cache_path_cache = get_environment("TEMP").replace("\\", "/");
		}
		if (cache_path_cache.is_empty()) {
			cache_path_cache = get_config_path();
		}
	}
	return cache_path_cache;
}

// Get properly capitalized engine name for system paths
String OS_Windows::get_godot_dir_name() const {
	return String(VERSION_SHORT_NAME).capitalize();
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
	String path = String::utf16((const char16_t *)szPath).replace("\\", "/");
	CoTaskMemFree(szPath);
	return path;
}

String OS_Windows::get_user_data_dir() const {
	String appname = get_safe_dir_name(GLOBAL_GET("application/config/name"));
	if (!appname.is_empty()) {
		bool use_custom_dir = GLOBAL_GET("application/config/use_custom_user_dir");
		if (use_custom_dir) {
			String custom_dir = get_safe_dir_name(GLOBAL_GET("application/config/custom_user_dir_name"), true);
			if (custom_dir.is_empty()) {
				custom_dir = appname;
			}
			return get_data_path().path_join(custom_dir).replace("\\", "/");
		} else {
			return get_data_path().path_join(get_godot_dir_name()).path_join("app_userdata").path_join(appname).replace("\\", "/");
		}
	}

	return get_data_path().path_join(get_godot_dir_name()).path_join("app_userdata").path_join("[unnamed project]");
}

String OS_Windows::get_unique_id() const {
	HW_PROFILE_INFOA HwProfInfo;
	ERR_FAIL_COND_V(!GetCurrentHwProfileA(&HwProfInfo), "");
	return String((HwProfInfo.szHwProfileGuid), HW_PROFILE_GUIDLEN);
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
		pba.resize(size);
		CryptBinaryToStringA(curr->pbCertEncoded, curr->cbCertEncoded, CRYPT_STRING_BASE64HEADER | CRYPT_STRING_NOCR, (char *)pba.ptrw(), &size);
		certs += String((char *)pba.ptr(), size);
		curr = CertEnumCertificatesInStore(cert_store, curr);
	}
	CertCloseStore(cert_store, 0);
	return certs;
}

OS_Windows::OS_Windows(HINSTANCE _hInstance) {
	hInstance = _hInstance;

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
	DWORD outMode = ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(stdoutHandle, outMode)) {
		// Windows 8.1 or below, or Windows 10 prior to Anniversary Update.
		print_verbose("Can't set the ENABLE_VIRTUAL_TERMINAL_PROCESSING Windows console mode. `print_rich()` will not work as expected.");
	}

	Vector<Logger *> loggers;
	loggers.push_back(memnew(WindowsTerminalLogger));
	_set_logger(memnew(CompositeLogger(loggers)));
}

OS_Windows::~OS_Windows() {
	CoUninitialize();
}
