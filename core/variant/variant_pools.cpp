/**************************************************************************/
/*  variant_pools.cpp                                                     */
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

#include "variant_pools.h"
#include "core/templates/paged_allocator.h"

namespace VariantPools {
union BucketSmall {
	BucketSmall() {}
	~BucketSmall() {}
	Transform2D _transform2d;
	::AABB _aabb;
};
union BucketMedium {
	BucketMedium() {}
	~BucketMedium() {}
	Basis _basis;
	Transform3D _transform3d;
};
union BucketLarge {
	BucketLarge() {}
	~BucketLarge() {}
	Projection _projection;
};

PagedAllocator<BucketSmall, true> _bucket_small;
PagedAllocator<BucketMedium, true> _bucket_medium;
PagedAllocator<BucketLarge, true> _bucket_large;

Transform2D *alloc_transform_2d() {
	return (Transform2D *)_bucket_small.alloc();
}

void free_transform_2d(Transform2D *p_transform_2d) {
	_bucket_small.free((BucketSmall *)p_transform_2d);
}

AABB *alloc_aabb() {
	return (AABB *)_bucket_small.alloc();
}

void free_aabb(AABB *p_aabb) {
	_bucket_small.free((BucketSmall *)p_aabb);
}

Basis *alloc_basis() {
	return (Basis *)_bucket_medium.alloc();
}

void free_basis(Basis *p_basis) {
	_bucket_medium.free((BucketMedium *)p_basis);
}

Transform3D *alloc_transform_3d() {
	return (Transform3D *)_bucket_medium.alloc();
}

void free_transform_3d(Transform3D *p_transform_3d) {
	_bucket_medium.free((BucketMedium *)p_transform_3d);
}

Projection *alloc_projection() {
	return (Projection *)_bucket_large.alloc();
}

void free_projection(Projection *p_projection) {
	_bucket_large.free((BucketLarge *)p_projection);
}
} //namespace VariantPools
