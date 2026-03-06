/**************************************************************************/
/*  gltf_attribute_map.h                                                  */
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

#include "core/io/resource.h"

class GLTFAttributeMap : public Resource {
	GDCLASS(GLTFAttributeMap, Resource);

protected:
	static void _bind_methods();

private:
	friend class GLTFDocument;

	String _color = "COLOR_0";
	String _uv = "TEXCOORD_0";
	String _uv2 = "TEXCOORD_1";

	String _custom[4]{
		"TEXCOORD_2",
		"TEXCOORD_4",
		"TEXCOORD_6",
		""
	};
	String _custom_mux[4]{
		"TEXCOORD_3",
		"TEXCOORD_5",
		"TEXCOORD_7",
		""
	};

public:
	String get_color() const;
	void set_color(const String &p_color);

	String get_uv() const;
	void set_uv(const String &p_uv);

	String get_uv2() const;
	void set_uv2(const String &p_uv2);

	String get_custom0() const;
	void set_custom0(const String &p_custom0);

	String get_custom0_mux() const;
	void set_custom0_mux(const String &p_custom0_mux);

	String get_custom1() const;
	void set_custom1(const String &p_custom1);

	String get_custom1_mux() const;
	void set_custom1_mux(const String &p_custom1_mux);

	String get_custom2() const;
	void set_custom2(const String &p_custom2);

	String get_custom2_mux() const;
	void set_custom2_mux(const String &p_custom2_mux);

	String get_custom3() const;
	void set_custom3(const String &p_custom3);

	String get_custom3_mux() const;
	void set_custom3_mux(const String &p_custom3_mux);
};
