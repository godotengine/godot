/**************************************************************************/
/*  spx_ext_mgr.h                                                      */
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

#ifndef SPX_EXT_MGR_H
#define SPX_EXT_MGR_H

#include "gdextension_spx_ext.h"
#include "scene/2d/node_2d.h"
#include "spx_base_mgr.h"

class SpxPen;
class SpxExtMgr : SpxBaseMgr {
	SPXCLASS(SpxExtMgr, SpxBaseMgr)
public:
	virtual ~SpxExtMgr() = default;

private:
	RBMap<GdObj, SpxPen *> id_pens;
	Node *pen_root;

	static Mutex lock;
private:
	SpxPen *_get_pen(GdObj id);

public:
	void on_awake() override;
	void on_start() override;
	void on_destroy() override;
	void on_update(float delta) override;

public:
	// engine API
	void request_exit(GdInt exit_code);
	void on_runtime_panic(GdString msg);
	
	// pause API
	void pause();
	void resume();
	GdBool is_paused();
	void next_frame();

	// obj APIs
	void destroy_all_pens();
	GdObj create_pen();
	void destroy_pen(GdObj obj);
	void pen_stamp(GdObj obj);
	void move_pen_to(GdObj obj, GdVec2 position);
	void pen_down(GdObj obj, GdBool move_by_mouse);
	void pen_up(GdObj obj);
	void set_pen_color_to(GdObj obj, GdColor color);
	void change_pen_by(GdObj obj, GdInt property, GdFloat amount);
	void set_pen_to(GdObj obj, GdInt property, GdFloat value);
	void change_pen_size_by(GdObj obj, GdFloat amount);
	void set_pen_size_to(GdObj obj, GdFloat size);
	void set_pen_stamp_texture(GdObj obj, GdString texture_path);
};

#endif // SPX_EXT_MGR_H
