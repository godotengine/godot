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

#include "core/variant/variant_pools.h"

#include "core/math/aabb.h"
#include "core/math/projection.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/templates/paged_allocator.h"

namespace VariantPools {
union BucketSmall {
	BucketSmall() {}
	~BucketSmall() {}
	Transform2D _transform2d;
	::AABB _aabb;
};
static_assert(sizeof(BucketSmall) == VariantPools::BUCKET_SMALL);
static_assert(alignof(BucketSmall) == alignof(real_t));

union BucketMedium {
	BucketMedium() {}
	~BucketMedium() {}
	Basis _basis;
	Transform3D _transform3d;
};
static_assert(sizeof(BucketMedium) == VariantPools::BUCKET_MEDIUM);
static_assert(alignof(BucketMedium) == alignof(real_t));

union BucketLarge {
	BucketLarge() {}
	~BucketLarge() {}
	Projection _projection;
};
static_assert(sizeof(BucketLarge) == VariantPools::BUCKET_LARGE);
static_assert(alignof(BucketLarge) == alignof(real_t));
} //namespace VariantPools

static PagedAllocator<VariantPools::BucketSmall, true> _bucket_small;
static PagedAllocator<VariantPools::BucketMedium, true> _bucket_medium;
static PagedAllocator<VariantPools::BucketLarge, true> _bucket_large;

void *VariantPools::alloc_small() {
	return _bucket_small.alloc();
}

void *VariantPools::alloc_medium() {
	return _bucket_medium.alloc();
}

void *VariantPools::alloc_large() {
	return _bucket_large.alloc();
}

void VariantPools::free_small(void *p_ptr) {
	_bucket_small.free(static_cast<BucketSmall *>(p_ptr));
}

void VariantPools::free_medium(void *p_ptr) {
	_bucket_medium.free(static_cast<BucketMedium *>(p_ptr));
}

void VariantPools::free_large(void *p_ptr) {
	_bucket_large.free(static_cast<BucketLarge *>(p_ptr));
}
