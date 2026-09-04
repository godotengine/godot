/**************************************************************************/
/*  object_gdtype.cpp                                                     */
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

#include "core/object/object_gdtype.h"

#include "core/os/thread.h"

#ifdef TOOLS_ENABLED
void ObjectGDType::link_properties(const StringName &p_property, const StringName &p_linked_property) {
	ERR_FAIL_COND(Thread::get_caller_id() != owning_thread_id);
	ERR_FAIL_COND(init_state != InitState::MUTABLE);

	const Member *member = _self_members.getptr(p_property);
	ERR_FAIL_NULL(member);
	ERR_FAIL_COND(member->type != GDType::Member::Type::PROPERTY);

	const Member *linked_property = _self_members.getptr(p_linked_property);
	ERR_FAIL_NULL(linked_property);
	ERR_FAIL_COND(linked_property->type != GDType::Member::Type::PROPERTY);

	if (!self_linked_properties.has(p_property)) {
		self_linked_properties.insert(p_property, LocalVector<StringName>());
	}
	self_linked_properties[p_property].push_back(p_linked_property);
}
#endif
