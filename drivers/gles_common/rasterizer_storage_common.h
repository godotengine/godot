/**************************************************************************/
/*  rasterizer_storage_common.h                                           */
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

#ifndef RASTERIZER_STORAGE_COMMON_H
#define RASTERIZER_STORAGE_COMMON_H

class RasterizerStorageCommon {
public:
	enum FVF {
		FVF_UNBATCHED,
		FVF_REGULAR,
		FVF_COLOR,
		FVF_LIGHT_ANGLE,
		FVF_MODULATED,
		FVF_LARGE,
	};

	// these flags are specifically for batching
	// some of the logic is thus in rasterizer_storage.cpp
	// we could alternatively set bitflags for each 'uses' and test on the fly
	enum BatchFlags : uint32_t {
		PREVENT_COLOR_BAKING = 1 << 0,
		PREVENT_VERTEX_BAKING = 1 << 1,

		// custom vertex shaders using BUILTINS that vary per item
		PREVENT_ITEM_JOINING = 1 << 2,

		USE_MODULATE_FVF = 1 << 3,
		USE_LARGE_FVF = 1 << 4,
	};

	enum BatchType : uint16_t {
		BT_DEFAULT = 0,
		BT_RECT = 1,
		BT_LINE = 2,
		BT_LINE_AA = 3,
		BT_POLY = 4,
		BT_DUMMY = 5, // dummy batch is just used to keep the batch creation loop simple
	};

	enum BatchTypeFlags : uint32_t {
		BTF_DEFAULT = 1 << BT_DEFAULT,
		BTF_RECT = 1 << BT_RECT,
		BTF_LINE = 1 << BT_LINE,
		BTF_LINE_AA = 1 << BT_LINE_AA,
		BTF_POLY = 1 << BT_POLY,
	};
};

#endif // RASTERIZER_STORAGE_COMMON_H
