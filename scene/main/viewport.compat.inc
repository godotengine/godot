/**************************************************************************/
/*  viewport.compat.inc                                                   */
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

#ifndef DISABLE_DEPRECATED

Size2i SubViewport::_get_size_2d_override_bind_compat_104601() const {
	// This implementation and comment is from when size_2d_override was a Size2i:
	// Rounding will cause offset issues with the
	// exact positioning of subwindows, but changing the
	// type of size_2d_override would break compatibility.
	return Size2i((get_size_2d_override() + Size2(0.5, 0.5)).floor());
}

void SubViewport::_set_size_2d_override_bind_compat_104601(const Size2i &p_size) {
	set_size_2d_override(p_size);
}

void SubViewport::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("get_size_2d_override"), &SubViewport::_get_size_2d_override_bind_compat_104601);
	ClassDB::bind_compatibility_method(D_METHOD("set_size_2d_override", "p_size"), &SubViewport::_set_size_2d_override_bind_compat_104601);
}

#endif // DISABLE_DEPRECATED
