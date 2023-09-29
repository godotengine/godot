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

void AnimationTree::_set_process_callback_bind_compat_80813(AnimationTree::AnimationProcessCallback p_mode) {
	set_callback_mode_process(static_cast<AnimationMixer::AnimationCallbackModeProcess>(static_cast<int>(p_mode)));
}

AnimationTree::AnimationProcessCallback AnimationTree::_get_process_callback_bind_compat_80813() const {
	return static_cast<AnimationProcessCallback>(static_cast<int>(get_callback_mode_process()));
}

void AnimationTree::_set_tree_root_bind_compat_80813(const Ref<AnimationNode> &p_root) {
	const Ref<AnimationRootNode> rn = Ref<AnimationRootNode>(p_root.ptr());
	if (rn.is_null()) {
		return;
	}
	return (set_root_animation_node(rn));
}

Ref<AnimationNode> AnimationTree::_get_tree_root_bind_compat_80813() const {
	const Ref<AnimationRootNode> rn = Ref<AnimationNode>(get_root_animation_node().ptr());
	return rn;
}

void AnimationTree::_bind_compatibility_methods() {
	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &AnimationTree::_set_process_callback_bind_compat_80813);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &AnimationTree::_get_process_callback_bind_compat_80813);
	ClassDB::bind_compatibility_method(D_METHOD("set_tree_root", "root"), &AnimationTree::_set_tree_root_bind_compat_80813);
	ClassDB::bind_compatibility_method(D_METHOD("get_tree_root"), &AnimationTree::_get_tree_root_bind_compat_80813);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_IDLE);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_MANUAL);
}

#endif // DISABLE_DEPRECATED
