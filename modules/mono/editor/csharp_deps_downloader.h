/**************************************************************************/
/*  csharp_deps_downloader.h                                              */
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

#ifndef CSHARP_DEPS_DOWNLOADER_H
#define CSHARP_DEPS_DOWNLOADER_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/main/http_request.h"

class CSharpDepsDownloader : public EditorPlugin {
	GDCLASS(CSharpDepsDownloader, EditorPlugin);

	HTTPRequest *http_request = nullptr;
	bool is_downloading = false;
	float update_countdown = 0;
	EditorProgress *progress = nullptr;

	int current_progress = 0;
	String current_status;

	void _show_download_error_alert();

	void _start_download(const String &p_url);
	void _cancel_download();
	void _download_completed(int p_status, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_data);

	bool _humanize_http_status(HTTPRequest *p_request, String *r_status, int *r_downloaded_bytes, int *r_total_bytes);

	void _set_current_progress_status(const String &p_status, bool p_error = false);
	void _set_current_progress_value(float p_value, const String &p_status);

	bool _install_file(const String &p_file, bool p_skip_progress);
	void _install_completed();

protected:
	void _notification(int p_what);

public:
	void download_deps(const String &p_url);

	CSharpDepsDownloader();
};

#endif // CSHARP_DEPS_DOWNLOADER_H
