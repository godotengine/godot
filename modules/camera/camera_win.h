/**************************************************************************/
/*  camera_win.h                                                          */
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

#include "servers/camera/camera_feed.h"
#include "servers/camera_server.h"
#include <initguid.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <windows.h>

class CameraFeedWindows : public CameraFeed {
private:
	String device_id;
	IMFMediaSource *imf_media_source = NULL;

	IMFSourceReader *imf_source_reader = NULL;
	std::thread *worker;
	Vector<GUID> format_guids;
	Vector<uint32_t> format_mediatypes;

	Vector<uint32_t> warned_formats;

	// image_y is used as unique image when format is RGB
	Ref<Image> image_y;
	Ref<Image> image_uv;
	Vector<uint8_t> data_y;
	Vector<uint8_t> data_uv;

	static void capture(CameraFeedWindows *feed);

	void read();
	void fill_formats(IMFMediaTypeHandler *imf_media_type_handler);

protected:
public:
	static Ref<CameraFeedWindows> create(IMFActivate *pDevice);
	virtual ~CameraFeedWindows();

	virtual Array get_formats() const override;
	virtual bool set_format(int p_index, const Dictionary &p_parameters) override;

	virtual bool activate_feed() override;
	virtual void deactivate_feed() override;
};

class CameraWindows : public CameraServer {
private:
	void update_feeds();

public:
	CameraWindows();
	~CameraWindows();

	virtual void set_monitoring_feeds(bool p_monitoring_feeds) override;
};
