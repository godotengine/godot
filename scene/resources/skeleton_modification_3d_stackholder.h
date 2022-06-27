/*************************************************************************/
/*  skeleton_modification_3d_stackholder.h                               */
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

#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

#ifndef SKELETONMODIFICATION3DSTACKHOLDER_H
#define SKELETONMODIFICATION3DSTACKHOLDER_H

class SkeletonModification3DStackHolder : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DStackHolder, SkeletonModification3D);

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	Ref<SkeletonModificationStack3D> held_modification_stack;

	virtual void _execute(real_t p_delta) override;
	virtual void _setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_held_modification_stack(Ref<SkeletonModificationStack3D> p_held_stack);
	Ref<SkeletonModificationStack3D> get_held_modification_stack() const;

	SkeletonModification3DStackHolder();
	~SkeletonModification3DStackHolder();
};

#endif //SKELETONMODIFICATION3DSTACKHOLDER_H
