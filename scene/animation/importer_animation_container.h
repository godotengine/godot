/*************************************************************************/
/*  importer_animation_container.h                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef IMPORTERANIMATIONCONTAINER_H
#define IMPORTERANIMATIONCONTAINER_H

#include "scene/main/node.h"
#include "scene/resources/importer_animation.h"

class ImporterAnimationContainer : public Node {
	GDCLASS(ImporterAnimationContainer, Node)

	Map<StringName, Ref<ImporterAnimation>> animations;

	TypedArray<StringName> _get_animation_list();

	Dictionary _get_animations() const;
	void _set_animations(const Dictionary &p_animations);

protected:
	static void _bind_methods();

public:
	void add_animation(const StringName &p_name, Ref<ImporterAnimation> p_animation);
	void set_animation(const StringName &p_name, Ref<ImporterAnimation> p_animation);
	void rename_animation(const StringName &p_name, const StringName &p_to_name);
	Ref<ImporterAnimation> get_animation(const StringName &p_name) const;
	void remove_animation(const StringName &p_name);
	void clear();

	Vector<StringName> get_animation_list();

	ImporterAnimationContainer();
};

#endif // IMPORTERANIMATIONCONTAINER_H
