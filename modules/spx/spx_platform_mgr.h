/**************************************************************************/
/*  spx_platform_mgr.h                                                       */
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

#ifndef SPX_OS_MGR_H
#define SPX_OS_MGR_H

#include "gdextension_spx_ext.h"
#include "spx_base_mgr.h"

class SpxPlatformMgr : SpxBaseMgr {
	SPXCLASS(SpxPlatformMgr, SpxBaseMgr)
	String persistant_data_dir = "res://";
public:
	virtual ~SpxPlatformMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor

	void on_awake() override;
	void _set_persistant_data_dir(String path);
	String _get_persistant_data_dir();
	
public:
	//Expose as few interfaces as possible to prevent misuse.
	void set_stretch_mode(GdBool enable);

	void set_window_position(GdVec2 pos);
	GdVec2 get_window_position();
	void set_window_size(GdInt width, GdInt height);
	GdVec2 get_window_size();
	void set_window_title(GdString title);
	GdString get_window_title();
	void set_window_fullscreen(GdBool enable);
	GdBool is_window_fullscreen();
	void set_debug_mode(GdBool enable);
	GdBool is_debug_mode();

	GdFloat get_time_scale();
	void set_time_scale(GdFloat time_scale);

	GdString get_persistant_data_dir();
	void set_persistant_data_dir(GdString path);
	GdBool is_in_persistant_data_dir(GdString path);

};

#endif // SPX_OS_MGR_H
