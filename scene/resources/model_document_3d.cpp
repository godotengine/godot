/**************************************************************************/
/*  model_document_3d.cpp                                                 */
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

#include "scene/resources/model_document_3d.h"

void ModelDocument3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("append_data_from_file", "path", "state", "flags", "base_path"),
			&ModelDocument3D::append_data_from_file, DEFVAL(0), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("append_data_from_buffer", "bytes", "base_path", "state", "flags"),
			&ModelDocument3D::append_data_from_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("append_data_from_scene", "node", "state", "flags"),
			&ModelDocument3D::append_data_from_scene, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("create_scene", "state", "bake_fps", "trimming", "remove_immutable_tracks"),
			&ModelDocument3D::create_scene, DEFVAL(30), DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("create_buffer", "state"),
			&ModelDocument3D::create_buffer);
	ClassDB::bind_method(D_METHOD("write_asset_to_filesystem", "state", "path"),
			&ModelDocument3D::write_asset_to_filesystem);
}
