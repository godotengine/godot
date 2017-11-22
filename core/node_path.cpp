/*************************************************************************/
/*  node_path.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "node_path.h"

#include "print_string.h"

uint32_t NodePath::hash() const {

	if (!data)
		return 0;

	uint32_t h = data->absolute ? 1 : 0;
	int pc = data->path.size();
	const StringName *sn = data->path.ptr();
	for (int i = 0; i < pc; i++) {
		h = h ^ sn[i].hash();
	}
	int spc = data->subpath.size();
	const StringName *ssn = data->subpath.ptr();
	for (int i = 0; i < spc; i++) {
		h = h ^ ssn[i].hash();
	}

	return h;
}

void NodePath::prepend_period() {

	if (data->path.size() && data->path[0].operator String() != ".") {
		data->path.insert(0, ".");
	}
}

bool NodePath::is_absolute() const {

	if (!data)
		return false;

	return data->absolute;
}
int NodePath::get_name_count() const {

	if (!data)
		return 0;

	return data->path.size();
}
StringName NodePath::get_name(int p_idx) const {

	ERR_FAIL_COND_V(!data, StringName());
	ERR_FAIL_INDEX_V(p_idx, data->path.size(), StringName());
	return data->path[p_idx];
}

int NodePath::get_subname_count() const {

	if (!data)
		return 0;

	return data->subpath.size();
}
StringName NodePath::get_subname(int p_idx) const {

	ERR_FAIL_COND_V(!data, StringName());
	ERR_FAIL_INDEX_V(p_idx, data->subpath.size(), StringName());
	return data->subpath[p_idx];
}

void NodePath::unref() {

	if (data && data->refcount.unref()) {

		memdelete(data);
	}
	data = NULL;
}

bool NodePath::operator==(const NodePath &p_path) const {

	if (data == p_path.data)
		return true;

	if (!data || !p_path.data)
		return false;

	if (data->absolute != p_path.data->absolute)
		return false;

	if (data->path.size() != p_path.data->path.size())
		return false;

	if (data->subpath.size() != p_path.data->subpath.size())
		return false;

	for (int i = 0; i < data->path.size(); i++) {

		if (data->path[i] != p_path.data->path[i])
			return false;
	}

	for (int i = 0; i < data->subpath.size(); i++) {

		if (data->subpath[i] != p_path.data->subpath[i])
			return false;
	}

	return true;
}
bool NodePath::operator!=(const NodePath &p_path) const {

	return (!(*this == p_path));
}

void NodePath::operator=(const NodePath &p_path) {

	if (this == &p_path)
		return;

	unref();

	if (p_path.data && p_path.data->refcount.ref()) {

		data = p_path.data;
	}
}

NodePath::operator String() const {

	if (!data)
		return String();

	String ret;
	if (data->absolute)
		ret = "/";

	for (int i = 0; i < data->path.size(); i++) {

		if (i > 0)
			ret += "/";
		ret += data->path[i].operator String();
	}

	for (int i = 0; i < data->subpath.size(); i++) {

		ret += ":" + data->subpath[i].operator String();
	}

	return ret;
}

NodePath::NodePath(const NodePath &p_path) {

	data = NULL;

	if (p_path.data && p_path.data->refcount.ref()) {

		data = p_path.data;
	}
}

Vector<StringName> NodePath::get_names() const {

	if (data)
		return data->path;
	return Vector<StringName>();
}

Vector<StringName> NodePath::get_subnames() const {

	if (data)
		return data->subpath;
	return Vector<StringName>();
}

StringName NodePath::get_concatenated_subnames() const {
	ERR_FAIL_COND_V(!data, StringName());

	if (!data->concatenated_subpath) {
		int spc = data->subpath.size();
		String concatenated;
		const StringName *ssn = data->subpath.ptr();
		for (int i = 0; i < spc; i++) {
			concatenated += i == 0 ? ssn[i].operator String() : ":" + ssn[i];
		}
		data->concatenated_subpath = concatenated;
	}
	return data->concatenated_subpath;
}

NodePath NodePath::rel_path_to(const NodePath &p_np) const {

	ERR_FAIL_COND_V(!is_absolute(), NodePath());
	ERR_FAIL_COND_V(!p_np.is_absolute(), NodePath());

	Vector<StringName> src_dirs = get_names();
	Vector<StringName> dst_dirs = p_np.get_names();

	//find common parent
	int common_parent = 0;

	while (true) {
		if (src_dirs.size() == common_parent)
			break;
		if (dst_dirs.size() == common_parent)
			break;
		if (src_dirs[common_parent] != dst_dirs[common_parent])
			break;
		common_parent++;
	}

	common_parent--;

	Vector<StringName> relpath;

	for (int i = src_dirs.size() - 1; i > common_parent; i--) {

		relpath.push_back("..");
	}

	for (int i = common_parent + 1; i < dst_dirs.size(); i++) {

		relpath.push_back(dst_dirs[i]);
	}

	if (relpath.size() == 0)
		relpath.push_back(".");

	return NodePath(relpath, p_np.get_subnames(), false);
}

NodePath NodePath::get_as_property_path() const {

	if (!data->path.size()) {
		return *this;
	} else {
		Vector<StringName> new_path = data->subpath;

		String initial_subname = data->path[0];
		for (size_t i = 1; i < data->path.size(); i++) {
			initial_subname += i == 0 ? data->path[i].operator String() : "/" + data->path[i];
		}
		new_path.insert(0, initial_subname);

		return NodePath(Vector<StringName>(), new_path, false);
	}
}

NodePath::NodePath(const Vector<StringName> &p_path, bool p_absolute) {

	data = NULL;

	if (p_path.size() == 0)
		return;

	data = memnew(Data);
	data->refcount.init();
	data->absolute = p_absolute;
	data->path = p_path;
	data->has_slashes = true;
}

NodePath::NodePath(const Vector<StringName> &p_path, const Vector<StringName> &p_subpath, bool p_absolute) {

	data = NULL;

	if (p_path.size() == 0 && p_subpath.size() == 0)
		return;

	data = memnew(Data);
	data->refcount.init();
	data->absolute = p_absolute;
	data->path = p_path;
	data->subpath = p_subpath;
	data->has_slashes = true;
}

void NodePath::simplify() {

	if (!data)
		return;
	for (int i = 0; i < data->path.size(); i++) {
		if (data->path.size() == 1)
			break;
		if (data->path[i].operator String() == ".") {
			data->path.remove(i);
			i--;
		} else if (data->path[i].operator String() == ".." && i > 0 && data->path[i - 1].operator String() != "." && data->path[i - 1].operator String() != "..") {
			//remove both
			data->path.remove(i - 1);
			data->path.remove(i - 1);
			i -= 2;
			if (data->path.size() == 0) {
				data->path.push_back(".");
				break;
			}
		}
	}
}

NodePath NodePath::simplified() const {

	NodePath np = *this;
	np.simplify();
	return np;
}

NodePath::NodePath(const String &p_path) {

	data = NULL;

	if (p_path.length() == 0)
		return;

	String path = p_path;
	Vector<StringName> subpath;

	int absolute = (path[0] == '/') ? 1 : 0;
	bool last_is_slash = true;
	bool has_slashes = false;
	int slices = 0;
	int subpath_pos = path.find(":");

	if (subpath_pos != -1) {

		int from = subpath_pos + 1;

		for (int i = from; i <= path.length(); i++) {

			if (path[i] == ':' || path[i] == 0) {

				String str = path.substr(from, i - from);
				if (str == "") {
					if (path[i] == 0) continue; // Allow end-of-path :

					ERR_EXPLAIN("Invalid NodePath: " + p_path);
					ERR_FAIL();
				}
				subpath.push_back(str);

				from = i + 1;
			}
		}

		path = path.substr(0, subpath_pos);
	}

	for (int i = absolute; i < path.length(); i++) {

		if (path[i] == '/') {

			last_is_slash = true;
			has_slashes = true;
		} else {

			if (last_is_slash)
				slices++;

			last_is_slash = false;
		}
	}

	if (slices == 0 && !absolute && !subpath.size())
		return;

	data = memnew(Data);
	data->refcount.init();
	data->absolute = absolute ? true : false;
	data->has_slashes = has_slashes;
	data->subpath = subpath;

	if (slices == 0)
		return;
	data->path.resize(slices);
	last_is_slash = true;
	int from = absolute;
	int slice = 0;

	for (int i = absolute; i < path.length() + 1; i++) {

		if (path[i] == '/' || path[i] == 0) {

			if (!last_is_slash) {

				String name = path.substr(from, i - from);
				ERR_FAIL_INDEX(slice, data->path.size());
				data->path[slice++] = name;
			}
			from = i + 1;
			last_is_slash = true;
		} else {
			last_is_slash = false;
		}
	}
}

bool NodePath::is_empty() const {

	return !data;
}
NodePath::NodePath() {

	data = NULL;
}

NodePath::~NodePath() {

	unref();
}
