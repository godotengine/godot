/*************************************************************************/
/*  path_db.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef PATH_DB_H
#define PATH_DB_H

#include "string_db.h"
#include "ustring.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class NodePath {

	struct Data {

		SafeRefCount refcount;
		StringName property;
		Vector<StringName> path;
		Vector<StringName> subpath;
		bool absolute;
	};

	Data *data;
	void unref();

public:
	_FORCE_INLINE_ StringName get_sname() const {

		if (data && data->path.size() == 1 && data->subpath.empty() && !data->property) {
			return data->path[0];
		} else {
			return operator String();
		}
	}

	bool is_absolute() const;
	int get_name_count() const;
	StringName get_name(int p_idx) const;
	int get_subname_count() const;
	StringName get_subname(int p_idx) const;
	Vector<StringName> get_names() const;
	Vector<StringName> get_subnames() const;

	NodePath rel_path_to(const NodePath &p_np) const;

	void prepend_period();

	StringName get_property() const;

	NodePath get_parent() const;

	uint32_t hash() const;

	operator String() const;
	bool is_empty() const;

	bool operator==(const NodePath &p_path) const;
	bool operator!=(const NodePath &p_path) const;
	void operator=(const NodePath &p_path);

	void simplify();
	NodePath simplified() const;

	NodePath(const Vector<StringName> &p_path, bool p_absolute, const String &p_property = "");
	NodePath(const Vector<StringName> &p_path, const Vector<StringName> &p_subpath, bool p_absolute, const String &p_property = "");
	NodePath(const NodePath &p_path);
	NodePath(const String &p_path);
	NodePath();
	~NodePath();
};

#endif
