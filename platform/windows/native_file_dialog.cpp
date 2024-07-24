/**************************************************************************/
/*  native_file_dialog.cpp                                                */
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

#include "native_file_dialog.h"

#include "display_server_windows.h"
#include "drivers/windows/file_access_windows_pipe.h"

#include <shellapi.h>
#include <shlwapi.h>
#include <shobjidl.h>

#include <propkey.h>
#include <propvarutil.h>

// Silence warning due to a COM API weirdness.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

class FileDialogEventHandler : public IFileDialogEvents, public IFileDialogControlEvents {
	LONG ref_count = 1;
	int ctl_id = 1;

	HashMap<int, String> ctls;
	Dictionary selected;
	String root;

public:
	// IUnknown methods
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppv) {
		static const QITAB qit[] = {
#ifdef __MINGW32__
			{ &__uuidof(IFileDialogEvents), static_cast<decltype(qit[0].dwOffset)>(OFFSETOFCLASS(IFileDialogEvents, FileDialogEventHandler)) },
			{ &__uuidof(IFileDialogControlEvents), static_cast<decltype(qit[0].dwOffset)>(OFFSETOFCLASS(IFileDialogControlEvents, FileDialogEventHandler)) },
#else
			QITABENT(FileDialogEventHandler, IFileDialogEvents),
			QITABENT(FileDialogEventHandler, IFileDialogControlEvents),
#endif
			{ nullptr, 0 },
		};
		return QISearch(this, qit, riid, ppv);
	}

	ULONG STDMETHODCALLTYPE AddRef() {
		return InterlockedIncrement(&ref_count);
	}

	ULONG STDMETHODCALLTYPE Release() {
		long ref = InterlockedDecrement(&ref_count);
		if (!ref) {
			delete this;
		}
		return ref;
	}

	// IFileDialogEvents methods
	HRESULT STDMETHODCALLTYPE OnFileOk(IFileDialog *) { return S_OK; };
	HRESULT STDMETHODCALLTYPE OnFolderChange(IFileDialog *) { return S_OK; };

	HRESULT STDMETHODCALLTYPE OnFolderChanging(IFileDialog *p_pfd, IShellItem *p_item) {
		if (root.is_empty()) {
			return S_OK;
		}

		LPWSTR lpw_path = nullptr;
		p_item->GetDisplayName(SIGDN_FILESYSPATH, &lpw_path);
		if (!lpw_path) {
			return S_FALSE;
		}
		String path = String::utf16((const char16_t *)lpw_path).simplify_path();
		if (!path.begins_with(root.simplify_path())) {
			return S_FALSE;
		}
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnHelp(IFileDialog *) { return S_OK; };
	HRESULT STDMETHODCALLTYPE OnSelectionChange(IFileDialog *) { return S_OK; };
	HRESULT STDMETHODCALLTYPE OnShareViolation(IFileDialog *, IShellItem *, FDE_SHAREVIOLATION_RESPONSE *) { return S_OK; };
	HRESULT STDMETHODCALLTYPE OnTypeChange(IFileDialog *pfd) { return S_OK; };
	HRESULT STDMETHODCALLTYPE OnOverwrite(IFileDialog *, IShellItem *, FDE_OVERWRITE_RESPONSE *) { return S_OK; };

	// IFileDialogControlEvents methods
	HRESULT STDMETHODCALLTYPE OnItemSelected(IFileDialogCustomize *p_pfdc, DWORD p_ctl_id, DWORD p_item_idx) {
		if (ctls.has(p_ctl_id)) {
			selected[ctls[p_ctl_id]] = (int)p_item_idx;
		}
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnButtonClicked(IFileDialogCustomize *, DWORD) { return S_OK; };
	HRESULT STDMETHODCALLTYPE OnCheckButtonToggled(IFileDialogCustomize *p_pfdc, DWORD p_ctl_id, BOOL p_checked) {
		if (ctls.has(p_ctl_id)) {
			selected[ctls[p_ctl_id]] = (bool)p_checked;
		}
		return S_OK;
	}
	HRESULT STDMETHODCALLTYPE OnControlActivating(IFileDialogCustomize *, DWORD) { return S_OK; };

	Dictionary get_selected() {
		return selected;
	}

	void set_root(const String &p_root) {
		root = p_root;
	}

	void add_option(IFileDialogCustomize *p_pfdc, const String &p_name, const Vector<String> &p_options, int p_default) {
		int gid = ctl_id++;
		int cid = ctl_id++;

		if (p_options.size() == 0) {
			// Add check box.
			p_pfdc->StartVisualGroup(gid, L"");
			p_pfdc->AddCheckButton(cid, (LPCWSTR)p_name.utf16().get_data(), p_default);
			p_pfdc->SetControlState(cid, CDCS_VISIBLE | CDCS_ENABLED);
			p_pfdc->EndVisualGroup();
			selected[p_name] = (bool)p_default;
		} else {
			// Add combo box.
			p_pfdc->StartVisualGroup(gid, (LPCWSTR)p_name.utf16().get_data());
			p_pfdc->AddComboBox(cid);
			p_pfdc->SetControlState(cid, CDCS_VISIBLE | CDCS_ENABLED);
			for (int i = 0; i < p_options.size(); i++) {
				p_pfdc->AddControlItem(cid, i, (LPCWSTR)p_options[i].utf16().get_data());
			}
			p_pfdc->SetSelectedControlItem(cid, p_default);
			p_pfdc->EndVisualGroup();
			selected[p_name] = p_default;
		}
		ctls[cid] = p_name;
	}

	virtual ~FileDialogEventHandler(){};
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

void show_native_file_dialog(const String &p_pipe_name) {
	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

	FileAccessWindowsPipe pipe;
	pipe.open_internal(p_pipe_name, 0);

	// Set DPI awarensess.
	HMODULE Shcore = LoadLibraryW(L"Shcore.dll");
	if (Shcore != nullptr) {
		typedef HRESULT(WINAPI * SetProcessDpiAwareness_t)(SHC_PROCESS_DPI_AWARENESS);

		SetProcessDpiAwareness_t SetProcessDpiAwareness = (SetProcessDpiAwareness_t)GetProcAddress(Shcore, "SetProcessDpiAwareness");

		if (SetProcessDpiAwareness) {
			SetProcessDpiAwareness(SHC_PROCESS_SYSTEM_DPI_AWARE);
		}
	}

	// Setup dark mode support.
	OSVERSIONINFOW os_ver;
	ZeroMemory(&os_ver, sizeof(OSVERSIONINFOW));
	os_ver.dwOSVersionInfoSize = sizeof(OSVERSIONINFOW);

	HMODULE nt_lib = LoadLibraryW(L"ntdll.dll");
	if (nt_lib) {
		WineGetVersionPtr wine_get_version = (WineGetVersionPtr)GetProcAddress(nt_lib, "wine_get_version"); // Do not read Windows build number under Wine, it can be set to arbitrary value.
		if (!wine_get_version) {
			RtlGetVersionPtr RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(nt_lib, "RtlGetVersion");
			if (RtlGetVersion) {
				RtlGetVersion(&os_ver);
			}
		}
		FreeLibrary(nt_lib);
	}

	// Load UXTheme, available on Windows 10+ only.
	if (os_ver.dwBuildNumber >= 10240) {
		HMODULE ux_theme_lib = LoadLibraryW(L"uxtheme.dll");
		if (ux_theme_lib) {
			ShouldAppsUseDarkModePtr ShouldAppsUseDarkMode = (ShouldAppsUseDarkModePtr)GetProcAddress(ux_theme_lib, MAKEINTRESOURCEA(132));
			if (os_ver.dwBuildNumber >= 17763) { // Windows 10 Redstone 5 (1809)+ only.
				AllowDarkModeForAppPtr AllowDarkModeForApp = nullptr;
				SetPreferredAppModePtr SetPreferredAppMode = nullptr;
				FlushMenuThemesPtr FlushMenuThemes = nullptr;
				if (os_ver.dwBuildNumber < 18362) { // Windows 10 Redstone 5 (1809) and 19H1 (1903) only.
					AllowDarkModeForApp = (AllowDarkModeForAppPtr)GetProcAddress(ux_theme_lib, MAKEINTRESOURCEA(135));
				} else { // Windows 10 19H2 (1909)+ only.
					SetPreferredAppMode = (SetPreferredAppModePtr)GetProcAddress(ux_theme_lib, MAKEINTRESOURCEA(135));
					FlushMenuThemes = (FlushMenuThemesPtr)GetProcAddress(ux_theme_lib, MAKEINTRESOURCEA(136));
				}
				RefreshImmersiveColorPolicyStatePtr RefreshImmersiveColorPolicyState = (RefreshImmersiveColorPolicyStatePtr)GetProcAddress(ux_theme_lib, MAKEINTRESOURCEA(104));
				if (ShouldAppsUseDarkMode) {
					bool dark_mode = ShouldAppsUseDarkMode();
					if (SetPreferredAppMode) {
						SetPreferredAppMode(dark_mode ? APPMODE_ALLOWDARK : APPMODE_DEFAULT);
					} else if (AllowDarkModeForApp) {
						AllowDarkModeForApp(dark_mode);
					}
					if (RefreshImmersiveColorPolicyState) {
						RefreshImmersiveColorPolicyState();
					}
					if (FlushMenuThemes) {
						FlushMenuThemes();
					}
				}
			}
		}
	}

	// Read parent app info.
	HWND root_hwnd = (HWND)pipe.get_64();
	int64_t x = pipe.get_64();
	int64_t y = pipe.get_64();
	int64_t w = pipe.get_64();
	int64_t h = pipe.get_64();
	String appid = pipe.get_pascal_string();

	WNDCLASSW wc = {};
	wc.lpfnWndProc = DefWindowProcW;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.lpszClassName = L"Engine File Dialog";
	RegisterClassW(&wc);

	HWND hwnd = CreateWindowExW(WS_EX_APPWINDOW, L"Engine File Dialog", L"", WS_OVERLAPPEDWINDOW, x, y, w, h, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
	if (hwnd) {
		HICON mainwindow_icon = (HICON)SendMessage(root_hwnd, WM_GETICON, ICON_SMALL, 0);
		if (mainwindow_icon) {
			SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)mainwindow_icon);
		}
		mainwindow_icon = (HICON)SendMessage(root_hwnd, WM_GETICON, ICON_BIG, 0);
		if (mainwindow_icon) {
			SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)mainwindow_icon);
		}
		IPropertyStore *prop_store;
		HRESULT hr = SHGetPropertyStoreForWindow(hwnd, IID_IPropertyStore, (void **)&prop_store);
		if (hr == S_OK) {
			PROPVARIANT val;
			InitPropVariantFromString((PCWSTR)appid.utf16().get_data(), &val);
			prop_store->SetValue(PKEY_AppUserModel_ID, val);
			prop_store->Release();
		}
	}

	SetCurrentProcessExplicitAppUserModelID((PCWSTR)appid.utf16().get_data());

	String title = pipe.get_pascal_string();
	String current_directory = pipe.get_pascal_string();
	String root = pipe.get_pascal_string();
	String filename = pipe.get_pascal_string();
	bool show_hidded = (bool)pipe.get_8();
	DisplayServer::FileDialogMode mode = (DisplayServer::FileDialogMode)pipe.get_8();

	int64_t filter_count = pipe.get_64();
	Vector<Char16String> filter_names;
	Vector<Char16String> filter_exts;
	for (int64_t i = 0; i < filter_count; i++) {
		String E = pipe.get_pascal_string();
		Vector<String> tokens = E.split(";");
		if (tokens.size() >= 1) {
			String flt = tokens[0].strip_edges();
			int filter_slice_count = flt.get_slice_count(",");
			Vector<String> exts;
			for (int j = 0; j < filter_slice_count; j++) {
				String str = (flt.get_slice(",", j).strip_edges());
				if (!str.is_empty()) {
					exts.push_back(str);
				}
			}
			if (!exts.is_empty()) {
				String str = String(";").join(exts);
				filter_exts.push_back(str.utf16());
				if (tokens.size() == 2) {
					filter_names.push_back(tokens[1].strip_edges().utf16());
				} else {
					filter_names.push_back(str.utf16());
				}
			}
		}
	}
	if (filter_names.is_empty()) {
		filter_exts.push_back(String("*.*").utf16());
		filter_names.push_back(RTR("All Files").utf16());
	}

	struct Option {
		String name;
		Vector<String> values;
		int default_idx = 0;
	};
	Vector<Option> options_info;
	int64_t option_count = pipe.get_64();
	for (int64_t i = 0; i < option_count; i++) {
		Option opt;
		opt.name = pipe.get_pascal_string();
		int64_t value_count = pipe.get_64();
		for (int64_t j = 0; j < value_count; j++) {
			opt.values.push_back(pipe.get_pascal_string());
		}
		opt.default_idx = pipe.get_64();
		options_info.push_back(opt);
	}

	if (mode < 0 && mode >= DisplayServer::FILE_DIALOG_MODE_SAVE_MAX) {
		pipe.store_8(false);
		pipe.close();

		if (hwnd) {
			IPropertyStore *prop_store;
			HRESULT hr = SHGetPropertyStoreForWindow(hwnd, IID_IPropertyStore, (void **)&prop_store);
			if (hr == S_OK) {
				PROPVARIANT val;
				PropVariantInit(&val);
				prop_store->SetValue(PKEY_AppUserModel_ID, val);
				prop_store->Release();
			}
			DestroyWindow(hwnd);
		}
		UnregisterClassW(L"Engine File Dialog", GetModuleHandle(nullptr));
		CoUninitialize();

		return;
	}

	Vector<COMDLG_FILTERSPEC> filters;
	for (int i = 0; i < filter_names.size(); i++) {
		filters.push_back({ (LPCWSTR)filter_names[i].ptr(), (LPCWSTR)filter_exts[i].ptr() });
	}

	HRESULT hr = S_OK;
	IFileDialog *pfd = nullptr;
	if (mode == DisplayServer::FILE_DIALOG_MODE_SAVE_FILE) {
		hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileSaveDialog, (void **)&pfd);
	} else {
		hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog, (void **)&pfd);
	}
	if (SUCCEEDED(hr)) {
		IFileDialogEvents *pfde = nullptr;
		FileDialogEventHandler *event_handler = new FileDialogEventHandler();
		hr = event_handler->QueryInterface(IID_PPV_ARGS(&pfde));

		DWORD cookie = 0;
		hr = pfd->Advise(pfde, &cookie);

		IFileDialogCustomize *pfdc = nullptr;
		hr = pfd->QueryInterface(IID_PPV_ARGS(&pfdc));

		for (int i = 0; i < options_info.size(); i++) {
			event_handler->add_option(pfdc, options_info[i].name, options_info[i].values, options_info[i].default_idx);
		}
		event_handler->set_root(root);

		pfdc->Release();

		DWORD flags;
		pfd->GetOptions(&flags);
		if (mode == DisplayServer::FILE_DIALOG_MODE_OPEN_FILES) {
			flags |= FOS_ALLOWMULTISELECT;
		}
		if (mode == DisplayServer::FILE_DIALOG_MODE_OPEN_DIR) {
			flags |= FOS_PICKFOLDERS;
		}
		if (show_hidded) {
			flags |= FOS_FORCESHOWHIDDEN;
		}
		pfd->SetOptions(flags | FOS_FORCEFILESYSTEM);
		pfd->SetTitle((LPCWSTR)title.utf16().ptr());

		String dir = current_directory.replace("/", "\\");

		IShellItem *shellitem = nullptr;
		hr = SHCreateItemFromParsingName((LPCWSTR)dir.utf16().ptr(), nullptr, IID_IShellItem, (void **)&shellitem);
		if (SUCCEEDED(hr)) {
			pfd->SetDefaultFolder(shellitem);
			pfd->SetFolder(shellitem);
		}

		pfd->SetFileName((LPCWSTR)filename.utf16().ptr());
		pfd->SetFileTypes(filters.size(), filters.ptr());
		pfd->SetFileTypeIndex(0);

		hr = pfd->Show(hwnd);
		pfd->Unadvise(cookie);

		Dictionary options = event_handler->get_selected();

		pfde->Release();
		event_handler->Release();

		UINT index = 0;
		pfd->GetFileTypeIndex(&index);
		if (index > 0) {
			index = index - 1;
		}

		if (SUCCEEDED(hr)) {
			Vector<String> file_names;

			if (mode == DisplayServer::FILE_DIALOG_MODE_OPEN_FILES) {
				IShellItemArray *results;
				hr = static_cast<IFileOpenDialog *>(pfd)->GetResults(&results);
				if (SUCCEEDED(hr)) {
					DWORD count = 0;
					results->GetCount(&count);
					for (DWORD i = 0; i < count; i++) {
						IShellItem *result;
						results->GetItemAt(i, &result);

						PWSTR file_path = nullptr;
						hr = result->GetDisplayName(SIGDN_FILESYSPATH, &file_path);
						if (SUCCEEDED(hr)) {
							file_names.push_back(String::utf16((const char16_t *)file_path));
							CoTaskMemFree(file_path);
						}
						result->Release();
					}
					results->Release();
				}
			} else {
				IShellItem *result;
				hr = pfd->GetResult(&result);
				if (SUCCEEDED(hr)) {
					PWSTR file_path = nullptr;
					hr = result->GetDisplayName(SIGDN_FILESYSPATH, &file_path);
					if (SUCCEEDED(hr)) {
						file_names.push_back(String::utf16((const char16_t *)file_path));
						CoTaskMemFree(file_path);
					}
					result->Release();
				}
			}
			pipe.store_8(true);
			pipe.store_64(file_names.size());
			for (int64_t i = 0; i < file_names.size(); i++) {
				pipe.store_pascal_string(file_names[i]);
			}
			pipe.store_64(index);
			pipe.store_64(options.size());
			for (int64_t i = 0; i < options.size(); i++) {
				pipe.store_pascal_string((String)options.get_key_at_index(i));
				pipe.store_64((int)options.get_value_at_index(i));
			}
		} else {
			pipe.store_8(false);
		}
		pfd->Release();
		pipe.close();
	} else {
		pipe.store_8(false);
		pipe.close();
	}

	if (hwnd) {
		IPropertyStore *prop_store;
		hr = SHGetPropertyStoreForWindow(hwnd, IID_IPropertyStore, (void **)&prop_store);
		if (hr == S_OK) {
			PROPVARIANT val;
			PropVariantInit(&val);
			prop_store->SetValue(PKEY_AppUserModel_ID, val);
			prop_store->Release();
		}
		DestroyWindow(hwnd);
	}
	UnregisterClassW(L"Engine File Dialog", GetModuleHandle(nullptr));
	CoUninitialize();
}
