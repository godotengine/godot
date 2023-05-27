/**************************************************************************/
/*  animation_tree.compat.inc                                             */
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

#ifndef DISABLE_DEPRECATED

#include "animation_tree.h"

double AnimationNode::blend_input_compat_75759(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync) {
	return blend_input(p_input, p_time, p_seek, p_is_external_seeking, p_blend, p_filter, p_sync, false);
}

double AnimationNode::blend_node_compat_75759(const StringName &p_sub_path, Ref<AnimationNode> p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync) {
	return blend_node(p_sub_path, p_node, p_time, p_seek, p_is_external_seeking, p_blend, p_filter, p_sync, false);
}

void AnimationNode::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("blend_node", "name", "node", "time", "seek", "is_external_seeking", "blend", "filter", "sync"), &AnimationNode::blend_node_compat_75759, DEFVAL(FILTER_IGNORE), DEFVAL(true));
	ClassDB::bind_compatibility_method(D_METHOD("blend_input", "input_index", "time", "seek", "is_external_seeking", "blend", "filter", "sync"), &AnimationNode::blend_input_compat_75759, DEFVAL(FILTER_IGNORE), DEFVAL(true));
}

#endif
