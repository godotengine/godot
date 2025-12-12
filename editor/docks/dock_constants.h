/**************************************************************************/
/*  dock_constants.h                                                      */
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

namespace DockConstants {

enum DockSlot {
	DOCK_SLOT_NONE = -1,
	DOCK_SLOT_LEFT_UL,
	DOCK_SLOT_LEFT_BL,
	DOCK_SLOT_LEFT_UR,
	DOCK_SLOT_LEFT_BR,
	DOCK_SLOT_RIGHT_UL,
	DOCK_SLOT_RIGHT_BL,
	DOCK_SLOT_RIGHT_UR,
	DOCK_SLOT_RIGHT_BR,
	DOCK_SLOT_BOTTOM,
	DOCK_SLOT_MAIN_SCREEN,
	DOCK_SLOT_MAX
};

enum DockLayout {
	DOCK_LAYOUT_VERTICAL = 1,
	DOCK_LAYOUT_HORIZONTAL = 2,
	DOCK_LAYOUT_FLOATING = 4,
	DOCK_LAYOUT_MAIN_SCREEN = 8,
};

}; //namespace DockConstants
