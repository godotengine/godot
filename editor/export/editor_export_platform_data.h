/**************************************************************************/
/*  editor_export_platform_data.h                                         */
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

#pragma once

#include "core/io/file_access.h"
#include "core/os/shared_object.h"
#include "editor/editor_node.h"

class EditorFileSystemDirectory;

class EditorExportPlatformData {
public:
	static constexpr int PCK_PADDING = 16;

	enum ExportMessageType {
		EXPORT_MESSAGE_NONE,
		EXPORT_MESSAGE_INFO,
		EXPORT_MESSAGE_WARNING,
		EXPORT_MESSAGE_ERROR,
	};

	struct ExportMessage {
		ExportMessageType msg_type;
		String category;
		String text;
	};

	struct SavedData {
		uint64_t ofs = 0;
		uint64_t size = 0;
		bool encrypted = false;
		bool removal = false;
		bool delta = false;
		Vector<uint8_t> md5;
		CharString path_utf8;

		bool operator<(const SavedData &p_data) const {
			return path_utf8 < p_data.path_utf8;
		}
	};

	struct PackData {
		String path;
		Ref<FileAccess> f;
		Vector<SavedData> file_ofs;
		EditorProgress *ep = nullptr;
		Vector<SharedObject> *so_files = nullptr;
		bool use_sparse_pck = false;
	};

	enum DebugFlags {
		DEBUG_FLAG_DUMB_CLIENT = 1,
		DEBUG_FLAG_REMOTE_DEBUG = 2,
		DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST = 4,
		DEBUG_FLAG_VIEW_COLLISIONS = 8,
		DEBUG_FLAG_VIEW_NAVIGATION = 16,
	};

	struct ZipData {
		void *zip = nullptr;
		EditorProgress *ep = nullptr;
		Vector<SharedObject> *so_files = nullptr;
		int file_count = 0;
	};

	typedef Error (*EditorExportSaveFunction)(const Ref<EditorExportPreset> &p_preset, void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key, uint64_t p_seed, bool p_delta);
	typedef Error (*EditorExportRemoveFunction)(const Ref<EditorExportPreset> &p_preset, void *p_userdata, const String &p_path);
	typedef Error (*EditorExportSaveSharedObject)(const Ref<EditorExportPreset> &p_preset, void *p_userdata, const SharedObject &p_so);

	class EditorExportSaveProxy {
		HashSet<String> saved_paths;
		EditorExportSaveFunction save_func;
		bool tracking_saves = false;

	public:
		bool has_saved(const String &p_path) const { return saved_paths.has(p_path); }

		Error save_file(const Ref<EditorExportPreset> &p_preset, void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key, uint64_t p_seed, bool p_delta);

		EditorExportSaveProxy(EditorExportSaveFunction p_save_func, bool p_track_saves) :
				save_func(p_save_func), tracking_saves(p_track_saves) {}
	};
};
