/**************************************************************************/
/*  spx_camera_mgr.h                                                      */
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

#ifndef SPX_CAMERA_MGR_H
#define SPX_CAMERA_MGR_H

#include "gdextension_spx_ext.h"
#include "spx_base_mgr.h"

class Camera2D;

class SpxCameraMgr : SpxBaseMgr {
	SPXCLASS(SpxCameraMgr, SpxBaseMgr)

public:
	virtual ~SpxCameraMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor

private:
	Camera2D *camera = nullptr;

public:
	void on_awake() override;

public:
	GdVec2 get_camera_position();
	void set_camera_position(GdVec2 position);
	GdVec2 get_camera_zoom();
	void set_camera_zoom(GdVec2 size);
	GdRect2 get_viewport_rect();
};

#endif // SPX_CAMERA_MGR_H
