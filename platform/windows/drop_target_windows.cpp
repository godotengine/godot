/**************************************************************************/
/*  drop_target_windows.cpp                                               */
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

#include "drop_target_windows.h"

#include "core/io/dir_access.h"
#include "core/math/random_pcg.h"
#include "core/os/time.h"

#include <fileapi.h>
#include <shellapi.h>

// Helpers

static String create_temp_dir() {
	Char16String buf;
	int bufsize = GetTempPathW(0, nullptr) + 1;
	buf.resize_uninitialized(bufsize);
	if (GetTempPathW(bufsize, (LPWSTR)buf.ptrw()) == 0) {
		return "";
	}

	String tmp_dir = String::utf16((const char16_t *)buf.ptr()).simplify_path();
	RandomPCG gen(Time::get_singleton()->get_ticks_usec());

	const int attempts = 4;

	for (int i = 0; i < attempts; ++i) {
		uint32_t rnd = gen.rand();
		String dirname = "godot_tmp_" + String::num_uint64(rnd);
		String res_dir = tmp_dir.path_join(dirname);
		Char16String res_dir16 = res_dir.utf16();

		if (CreateDirectoryW((LPCWSTR)res_dir16.ptr(), nullptr)) {
			return res_dir;
		}
	}

	return "";
}

static bool remove_dir_recursive(const String &p_dir) {
	Ref<DirAccess> dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (dir_access->change_dir(p_dir) != OK) {
		return false;
	}
	return dir_access->erase_contents_recursive() == OK;
}

static bool stream2file(IStream *p_stream, FILEDESCRIPTORW *p_desc, const String &p_path) {
	if (DirAccess::make_dir_recursive_absolute(p_path.get_base_dir()) != OK) {
		return false;
	}

	Char16String path16 = p_path.utf16();
	DWORD dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;

	if (p_desc->dwFlags & FD_ATTRIBUTES) {
		dwFlagsAndAttributes = p_desc->dwFileAttributes;
	}

	HANDLE file = CreateFileW(
			(LPCWSTR)path16.ptr(),
			GENERIC_WRITE,
			0,
			nullptr,
			CREATE_NEW,
			dwFlagsAndAttributes,
			nullptr);

	if (!file) {
		return false;
	}

	const int bufsize = 4096;
	char buf[bufsize];
	ULONG nread = 0;
	DWORD nwritten = 0;
	HRESULT read_result = S_OK;
	bool result = true;

	while (true) {
		read_result = p_stream->Read(buf, bufsize, &nread);
		if (read_result != S_OK && read_result != S_FALSE) {
			result = false;
			goto cleanup;
		}

		if (!nread) {
			break;
		}

		while (nread > 0) {
			if (!WriteFile(file, buf, nread, &nwritten, nullptr) || !nwritten) {
				result = false;
				goto cleanup;
			}
			nread -= nwritten;
		}
	}

cleanup:
	CloseHandle(file);
	return result;
}

// DropTargetWindows

bool DropTargetWindows::is_valid_filedescriptor() {
	return cf_filedescriptor != 0 && cf_filecontents != 0;
}

HRESULT DropTargetWindows::handle_hdrop_format(Vector<String> *p_files, IDataObject *pDataObj) {
	FORMATETC fmt = { CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
	STGMEDIUM stg;
	HRESULT res = S_OK;

	if (pDataObj->GetData(&fmt, &stg) != S_OK) {
		return E_UNEXPECTED;
	}

	HDROP hDropInfo = (HDROP)GlobalLock(stg.hGlobal);

	Char16String buf;

	if (hDropInfo == nullptr) {
		ReleaseStgMedium(&stg);
		return E_UNEXPECTED;
	}

	int fcount = DragQueryFileW(hDropInfo, 0xFFFFFFFF, nullptr, 0);

	for (int i = 0; i < fcount; i++) {
		int buffsize = DragQueryFileW(hDropInfo, i, nullptr, 0);
		buf.resize_uninitialized(buffsize + 1);
		if (DragQueryFileW(hDropInfo, i, (LPWSTR)buf.ptrw(), buffsize + 1) == 0) {
			res = E_UNEXPECTED;
			goto cleanup;
		}
		String file = String::utf16((const char16_t *)buf.ptr());
		p_files->push_back(file.simplify_path());
	}

cleanup:
	GlobalUnlock(stg.hGlobal);
	ReleaseStgMedium(&stg);

	return res;
}

HRESULT DropTargetWindows::handle_filedescriptor_format(Vector<String> *p_files, IDataObject *pDataObj) {
	FORMATETC fmt = { cf_filedescriptor, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
	STGMEDIUM stg;
	HRESULT res = S_OK;

	if (pDataObj->GetData(&fmt, &stg) != S_OK) {
		return E_UNEXPECTED;
	}

	FILEGROUPDESCRIPTORW *filegroup_desc = (FILEGROUPDESCRIPTORW *)GlobalLock(stg.hGlobal);

	if (!filegroup_desc) {
		ReleaseStgMedium(&stg);
		return E_UNEXPECTED;
	}

	tmp_path = create_temp_dir();
	Ref<DirAccess> dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	PackedStringArray copied;

	if (dir_access->change_dir(tmp_path) != OK) {
		res = E_UNEXPECTED;
		goto cleanup;
	}

	for (int i = 0; i < (int)filegroup_desc->cItems; ++i) {
		res = save_as_file(tmp_path, filegroup_desc->fgd + i, pDataObj, i);
		if (res != S_OK) {
			res = E_UNEXPECTED;
			goto cleanup;
		}
	}

	copied = dir_access->get_files();
	for (const String &file : copied) {
		p_files->push_back(tmp_path.path_join(file));
	}

	copied = dir_access->get_directories();
	for (const String &dir : copied) {
		p_files->push_back(tmp_path.path_join(dir));
	}

cleanup:
	GlobalUnlock(filegroup_desc);
	ReleaseStgMedium(&stg);
	if (res != S_OK) {
		remove_dir_recursive(tmp_path);
		tmp_path.clear();
	}
	return res;
}

HRESULT DropTargetWindows::save_as_file(const String &p_out_dir, FILEDESCRIPTORW *p_file_desc, IDataObject *pDataObj, int p_file_idx) {
	String relpath = String::utf16((const char16_t *)p_file_desc->cFileName);
	String fullpath = p_out_dir.path_join(relpath);

	if (p_file_desc->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
		if (DirAccess::make_dir_recursive_absolute(fullpath) != OK) {
			return E_UNEXPECTED;
		}
		return S_OK;
	}

	FORMATETC fmt = { cf_filecontents, nullptr, DVASPECT_CONTENT, p_file_idx, TYMED_ISTREAM };
	STGMEDIUM stg;
	HRESULT res = S_OK;

	if (pDataObj->GetData(&fmt, &stg) != S_OK) {
		return E_UNEXPECTED;
	}

	IStream *stream = stg.pstm;
	if (stream == nullptr) {
		res = E_UNEXPECTED;
		goto cleanup;
	}

	if (!stream2file(stream, p_file_desc, fullpath)) {
		res = E_UNEXPECTED;
		goto cleanup;
	}

cleanup:
	ReleaseStgMedium(&stg);
	return res;
}

DropTargetWindows::DropTargetWindows(DisplayServerWindows::WindowData *p_window_data) :
		ref_count(1), window_data(p_window_data) {
	cf_filedescriptor = RegisterClipboardFormat(CFSTR_FILEDESCRIPTORW);
	cf_filecontents = RegisterClipboardFormat(CFSTR_FILECONTENTS);
}

HRESULT STDMETHODCALLTYPE DropTargetWindows::QueryInterface(REFIID riid, void **ppvObject) {
	if (riid == IID_IUnknown || riid == IID_IDropTarget) {
		*ppvObject = static_cast<IDropTarget *>(this);
		AddRef();
		return S_OK;
	}
	*ppvObject = nullptr;
	return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE DropTargetWindows::AddRef() {
	return InterlockedIncrement(&ref_count);
}

ULONG STDMETHODCALLTYPE DropTargetWindows::Release() {
	ULONG count = InterlockedDecrement(&ref_count);
	if (count == 0) {
		memfree(this);
	}
	return count;
}

HRESULT STDMETHODCALLTYPE DropTargetWindows::DragEnter(IDataObject *pDataObj, DWORD grfKeyState, POINTL pt, DWORD *pdwEffect) {
	(void)grfKeyState;
	(void)pt;

	FORMATETC hdrop_fmt = { CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
	FORMATETC filedesc_fmt = { cf_filedescriptor, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };

	if (!window_data->drop_files_callback.is_valid()) {
		*pdwEffect = DROPEFFECT_NONE;
	} else if (pDataObj->QueryGetData(&hdrop_fmt) == S_OK) {
		*pdwEffect = DROPEFFECT_COPY;
	} else if (is_valid_filedescriptor() && pDataObj->QueryGetData(&filedesc_fmt) == S_OK) {
		*pdwEffect = DROPEFFECT_COPY;
	} else {
		*pdwEffect = DROPEFFECT_NONE;
	}

	return S_OK;
}

HRESULT STDMETHODCALLTYPE DropTargetWindows::DragOver(DWORD grfKeyState, POINTL pt, DWORD *pdwEffect) {
	(void)grfKeyState;
	(void)pt;

	*pdwEffect = DROPEFFECT_COPY;
	return S_OK;
}

HRESULT STDMETHODCALLTYPE DropTargetWindows::DragLeave() {
	return S_OK;
}

HRESULT STDMETHODCALLTYPE DropTargetWindows::Drop(IDataObject *pDataObj, DWORD grfKeyState, POINTL pt, DWORD *pdwEffect) {
	(void)grfKeyState;
	(void)pt;

	*pdwEffect = DROPEFFECT_NONE;

	if (!window_data->drop_files_callback.is_valid()) {
		return S_OK;
	}

	FORMATETC hdrop_fmt = { CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
	FORMATETC filedesc_fmt = { cf_filedescriptor, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
	Vector<String> files;

	if (pDataObj->QueryGetData(&hdrop_fmt) == S_OK) {
		HRESULT res = handle_hdrop_format(&files, pDataObj);
		if (res != S_OK) {
			return res;
		}
	} else if (pDataObj->QueryGetData(&filedesc_fmt) == S_OK && is_valid_filedescriptor()) {
		HRESULT res = handle_filedescriptor_format(&files, pDataObj);
		if (res != S_OK) {
			return res;
		}
	} else {
		return E_UNEXPECTED;
	}

	if (!files.size()) {
		return S_OK;
	}

	Variant v_files = files;
	const Variant *v_args[1] = { &v_files };
	Variant ret;
	Callable::CallError ce;
	window_data->drop_files_callback.callp((const Variant **)&v_args, 1, ret, ce);

	if (!tmp_path.is_empty()) {
		remove_dir_recursive(tmp_path);
		tmp_path.clear();
	}

	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_PRINT(vformat("Failed to execute drop files callback: %s.", Variant::get_callable_error_text(window_data->drop_files_callback, v_args, 1, ce)));
		return E_UNEXPECTED;
	}

	*pdwEffect = DROPEFFECT_COPY;
	return S_OK;
}
