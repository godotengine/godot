/*************************************************************************/
/*  nav_base.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef NAV_BASE_H
#define NAV_BASE_H

#include "nav_rid.h"
#include "nav_utils.h"

class NavMap;

class NavBase : public NavRid {
protected:
	uint32_t navigation_layers = 1;
	float enter_cost = 0.0;
	float travel_cost = 1.0;

public:
	void set_navigation_layers(uint32_t p_navigation_layers) { navigation_layers = p_navigation_layers; }
	uint32_t get_navigation_layers() const { return navigation_layers; }

	void set_enter_cost(float p_enter_cost) { enter_cost = MAX(p_enter_cost, 0.0); }
	float get_enter_cost() const { return enter_cost; }

	void set_travel_cost(float p_travel_cost) { travel_cost = MAX(p_travel_cost, 0.0); }
	float get_travel_cost() const { return travel_cost; }
};

#endif // NAV_BASE_H
