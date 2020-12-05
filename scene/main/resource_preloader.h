/*************************************************************************/
/*  resource_preloader.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RESOURCE_PRELOADER_H
#define RESOURCE_PRELOADER_H

#include "scene/main/node.h"

class ResourcePreloader : public Node {
	GDCLASS(ResourcePreloader, Node);

	Map<StringName, RES> resources;

	void _set_resources(const Array &p_data);
	Array _get_resources() const;
	Vector<String> _get_resource_list() const;

protected:
	static void _bind_methods();

public:
	void add_resource(const StringName &p_name, const RES &p_resource);
	void remove_resource(const StringName &p_name);
	void rename_resource(const StringName &p_from_name, const StringName &p_to_name);
	bool has_resource(const StringName &p_name) const;
	RES get_resource(const StringName &p_name) const;

	void get_resource_list(List<StringName> *p_list);

	ResourcePreloader();
};

#endif // RESOURCE_PRELOADER_H
