/**************************************************************************/
/*  docker.h                                                              */
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

#include "core/string/ustring.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

struct Docker {
	using Array = Array;
	using PackedStringArray = PackedStringArray;
	using String = String;

	static bool ContainerStart(String container_name, String image_name, Array &output);
	static Array ContainerStop(String container_name);
	static bool ContainerExecute(String container_name, const PackedStringArray &args, Array &output, bool verbose = true);
	static int ContainerVersion(String container_name, const PackedStringArray &args);
	static String ContainerGetMountPath(String container_name);
	static bool ContainerPullLatest(String image_name, Array &output);
	static bool ContainerDelete(String container_name, Array &output);

	static String GetFolderName(const String &path) {
		String foldername = path.replace("res://", "");
		int idx = -1;
		do {
			idx = foldername.find("/");
			if (idx != -1)
				foldername = foldername.substr(idx + 1, foldername.length());
		} while (idx != -1);
		return foldername;
	}
};
