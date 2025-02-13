/**************************************************************************/
/*  nav_base.h                                                            */
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

#ifndef NAV_BASE_H
#define NAV_BASE_H

#include "nav_rid.h"
#include "nav_utils.h"

#include "servers/navigation/navigation_utilities.h"

class NavMap;

class NavBase : public NavRid {
protected:
	uint32_t navigation_layers = 1;
	real_t enter_cost = 0.0;
	real_t travel_cost = 1.0;
	ObjectID owner_id;
	NavigationUtilities::PathSegmentType type;

public:
	NavigationUtilities::PathSegmentType get_type() const { return type; }

	virtual void set_use_edge_connections(bool p_enabled) {}
	virtual bool get_use_edge_connections() const { return false; }

	virtual void set_navigation_layers(uint32_t p_navigation_layers) {}
	uint32_t get_navigation_layers() const { return navigation_layers; }

	virtual void set_enter_cost(real_t p_enter_cost) {}
	real_t get_enter_cost() const { return enter_cost; }

	virtual void set_travel_cost(real_t p_travel_cost) {}
	real_t get_travel_cost() const { return travel_cost; }

	virtual void set_owner_id(ObjectID p_owner_id) {}
	ObjectID get_owner_id() const { return owner_id; }

	virtual ~NavBase() {}
};

#endif // NAV_BASE_H
