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

DropTargetWindows::DropTargetWindows(DisplayServerWindows::WindowData *p_window_data) :
		ref_count(1), window_data(p_window_data) {}

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

	FORMATETC fmt = { CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };

	if (pDataObj->QueryGetData(&fmt) == S_OK) {
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

	FORMATETC fmt = { CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL };
	STGMEDIUM stg;
	const int buffsize = 4096;
	WCHAR buf[buffsize];

	if (pDataObj->GetData(&fmt, &stg) != S_OK) {
		return E_UNEXPECTED;
	}

	HDROP hDropInfo = (HDROP)GlobalLock(stg.hGlobal);

	Vector<String> files;

	if (hDropInfo != nullptr) {
		int fcount = DragQueryFileW(hDropInfo, 0xFFFFFFFF, nullptr, 0);

		for (int i = 0; i < fcount; i++) {
			DragQueryFileW(hDropInfo, i, buf, buffsize);
			String file = String::utf16((const char16_t *)buf);
			files.push_back(file);
		}

		GlobalUnlock(stg.hGlobal);
	}

	ReleaseStgMedium(&stg);

	if (!files.size() || !window_data->drop_files_callback.is_valid()) {
		return S_OK;
	}

	Variant v_files = files;
	const Variant *v_args[1] = { &v_files };
	Variant ret;
	Callable::CallError ce;
	window_data->drop_files_callback.callp((const Variant **)&v_args, 1, ret, ce);

	if (ce.error != Callable::CallError::CALL_OK) {
		ERR_PRINT(vformat("Failed to execute drop files callback: %s.", Variant::get_callable_error_text(window_data->drop_files_callback, v_args, 1, ce)));
		return E_UNEXPECTED;
	}

	*pdwEffect = DROPEFFECT_COPY;
	return S_OK;
}
