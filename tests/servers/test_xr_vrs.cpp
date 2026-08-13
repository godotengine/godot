/**************************************************************************/
/*  test_xr_vrs.cpp                                                       */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_xr_vrs)

#include "servers/xr/xr_vrs.h"

namespace TestXRVRS {

TEST_SUITE("[XRVRS]") {
	TEST_CASE("[XRVRS] make_vrs_texture without a RenderingDevice") {
		// VRS textures are backed by a RenderingDevice. Rendering methods that do not
		// provide one (Compatibility, headless) must fail gracefully rather than
		// dereferencing a null singleton.
		XRVRS vrs;

		ERR_PRINT_OFF;
		const RID texture = vrs.make_vrs_texture(Size2(100, 100), { Vector2(0, 0) });
		ERR_PRINT_ON;

		CHECK_MESSAGE(texture.is_null(), "Should return an invalid RID instead of crashing.");
	}

	TEST_CASE("[XRVRS] make_vrs_texture with empty eye foci") {
		XRVRS vrs;

		ERR_PRINT_OFF;
		const RID texture = vrs.make_vrs_texture(Size2(100, 100), Vector<Vector2>());
		ERR_PRINT_ON;

		CHECK_MESSAGE(texture.is_null(), "Should return an invalid RID when no eye foci are given.");
	}
}

} // namespace TestXRVRS
