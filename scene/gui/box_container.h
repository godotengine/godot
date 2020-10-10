/*************************************************************************/
/*  box_container.h                                                      */
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

#ifndef BOX_CONTAINER_H
#define BOX_CONTAINER_H

#include "scene/gui/container.h"

class BoxContainer : public Container {
	GDCLASS(BoxContainer, Container);

public:
	enum AlignMode {
		ALIGN_BEGIN,
		ALIGN_CENTER,
		ALIGN_END
	};

private:
	bool vertical;
	AlignMode align;

	void _resort();

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void add_spacer(bool p_begin = false);

	void set_alignment(AlignMode p_align);
	AlignMode get_alignment() const;

	virtual Size2 get_minimum_size() const override;

	BoxContainer(bool p_vertical = false);
};

class HBoxContainer : public BoxContainer {
	GDCLASS(HBoxContainer, BoxContainer);

public:
	HBoxContainer() :
			BoxContainer(false) {}
};

class MarginContainer;
class VBoxContainer : public BoxContainer {
	GDCLASS(VBoxContainer, BoxContainer);

public:
	MarginContainer *add_margin_child(const String &p_label, Control *p_control, bool p_expand = false);

	VBoxContainer() :
			BoxContainer(true) {}
};

VARIANT_ENUM_CAST(BoxContainer::AlignMode);

#endif // BOX_CONTAINER_H
