/*************************************************************************/
/*  aspect_ratio_container.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef ASPECT_RATIO_CONTAINER_H
#define ASPECT_RATIO_CONTAINER_H

#include "scene/gui/container.h"

class AspectRatioContainer : public Container {

	GDCLASS(AspectRatioContainer, Container);

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual Size2 get_minimum_size() const;

public:
	enum StretchMode {
		WIDTH_CONTROLS_HEIGHT,
		HEIGHT_CONTROLS_WIDTH,
		FIT,
		COVER,
	};

private:
	float ratio;
	StretchMode stretch_mode;
	float alignment_x;
	float alignment_y;

public:
	void set_ratio(float p_ratio);
	float get_ratio() const;

	void set_stretch_mode(StretchMode p_mode);
	StretchMode get_stretch_mode() const;

	void set_alignment_x(float p_alignment_x);
	float get_alignment_x() const;

	void set_alignment_y(float p_alignment_y);
	float get_alignment_y() const;

	AspectRatioContainer();
};

VARIANT_ENUM_CAST(AspectRatioContainer::StretchMode);
#endif // ASPECT_RATIO_CONTAINER_H
