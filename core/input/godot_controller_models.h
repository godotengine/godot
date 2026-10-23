/**************************************************************************/
/*  godot_controller_models.h                                             */
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

#include "core/input/input_enums.h"
#include "core/templates/hash_map.h"

#define CONTROLLER_VID_PID(VID, PID) (int32_t)((int32_t)VID << 16 | (int32_t)PID)

// Custom Godot controller models not defined by SDL or Godot's current SDL version.
static HashMap<int32_t, JoyModel> _godot_controller_models = {
	{ CONTROLLER_VID_PID(0x28de, 0x1205), JoyModel::STEAM }, // Steam Deck
	{ CONTROLLER_VID_PID(0x28de, 0x1302), JoyModel::STEAM }, // Steam Controller (2026)
	{ CONTROLLER_VID_PID(0x28de, 0x1303), JoyModel::STEAM }, // Steam Controller (2026)
	{ CONTROLLER_VID_PID(0x28de, 0x1304), JoyModel::STEAM }, // Steam Controller (2026) dongle
	{ CONTROLLER_VID_PID(0x28de, 0x1305), JoyModel::STEAM }, // Steam Controller (2026) dongle
};
