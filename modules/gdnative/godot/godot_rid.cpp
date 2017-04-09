/*************************************************************************/
/*  godot_rid.cpp                                                        */
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
#include "godot_rid.h"

#include "object.h"
#include "resource.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rid_api_anchor() {
}

void GDAPI godot_rid_new(godot_rid *p_rid, godot_object *p_from) {

	Resource *res_from = ((Object *)p_from)->cast_to<Resource>();

	RID *rid = (RID *)p_rid;
	memnew_placement(rid, RID);

	if (res_from) {
		*rid = RID(res_from->get_rid());
	}
}

uint32_t GDAPI godot_rid_get_rid(const godot_rid *p_rid) {
	RID *rid = (RID *)p_rid;
	return rid->get_id();
}

void GDAPI godot_rid_destroy(godot_rid *p_rid) {
	((RID *)p_rid)->~RID();
}

#ifdef __cplusplus
}
#endif
