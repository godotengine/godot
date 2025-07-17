/**************************************************************************/
/*  spx_input_mgr.h                                                       */
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

#ifndef SPX_INPUT_MGR_H
#define SPX_INPUT_MGR_H

#include "gdextension_spx_ext.h"
#include "spx_base_mgr.h"
#include "spx_input_proxy.h"



class SpxInputMgr : SpxBaseMgr {
	SPXCLASS(SpxInputMgr, SpxBaseMgr)
public:
	virtual ~SpxInputMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor
	virtual void on_start() override;
protected:
	SpxInputProxy *input_proxy;
public:
	GdVec2 get_mouse_pos();
	GdBool get_key(GdInt key);
	GdBool get_mouse_state(GdInt mouse_id);
	GdInt get_key_state(GdInt key);
	GdFloat get_axis(GdString neg_action,GdString pos_action);
	GdBool is_action_pressed(GdString action);
	GdBool is_action_just_pressed(GdString action);
	GdBool is_action_just_released(GdString action);
};

#endif // SPX_INPUT_MGR_H
