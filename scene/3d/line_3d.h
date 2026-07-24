/**************************************************************************/
/*  line_3d.h                                                             */
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

#include "scene/3d/visual_instance_3d.h"

class Line3D : public GeometryInstance3D {
	GDCLASS(Line3D, GeometryInstance3D);

protected:
	static void _bind_methods();

public:
	enum MeshAlignment {
		MESH_ALIGNMENT_LOCAL,
		MESH_ALIGNMENT_BILLBOARD,
		MESH_ALIGNMENT_MAX,
	};

	enum TilingMode {
		TILING_MODE_UNIT,
		TILING_MODE_LENGTH,
		TILING_MAX,
	};

	enum MaterialMode {
		MATERIAL_MODE_MIX,
		MATERIAL_MODE_ADD,
		MATERIAL_MODE_CUSTOM,
		MATERIAL_MODE_MAX,
	};
};

VARIANT_ENUM_CAST(Line3D::TilingMode)
VARIANT_ENUM_CAST(Line3D::MeshAlignment)
VARIANT_ENUM_CAST(Line3D::MaterialMode)
