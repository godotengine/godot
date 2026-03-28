/**************************************************************************/
/*  gaussian_splat_storage.cpp                                            */
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

#include "gaussian_splat_storage.h"

using namespace RendererRD;

GaussianSplatStorage *GaussianSplatStorage::singleton = nullptr;

GaussianSplatStorage *GaussianSplatStorage::get_singleton() {
        return singleton;
}

GaussianSplatStorage::GaussianSplatStorage() {
        ERR_FAIL_COND(singleton != nullptr);
        singleton = this;
}

GaussianSplatStorage::~GaussianSplatStorage() {
        singleton = nullptr;
}

RID GaussianSplatStorage::gaussian_allocate() {
        return gaussian_owner.allocate_rid();
}

void GaussianSplatStorage::gaussian_initialize(RID p_rid) {
        gaussian_owner.initialize_rid(p_rid, GaussianSplat());
}

void GaussianSplatStorage::gaussian_free(RID p_rid) {
        gaussian_owner.free(p_rid);
}

bool GaussianSplatStorage::owns_gaussian(RID p_rid) const {
        return gaussian_owner.owns(p_rid);
}

#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
void GaussianSplatStorage::gaussian_set_renderer(RID p_rid, const Ref<GaussianSplatRenderer> &p_renderer) {
        GaussianSplat *splat = gaussian_owner.get_or_null(p_rid);
        ERR_FAIL_NULL(splat);
        splat->renderer = p_renderer;
        if (p_renderer.is_valid()) {
                splat->aabb = p_renderer->get_aabb();
        }
}

Ref<GaussianSplatRenderer> GaussianSplatStorage::gaussian_get_renderer(RID p_rid) const {
        const GaussianSplat *splat = gaussian_owner.get_or_null(p_rid);
        ERR_FAIL_NULL_V(splat, Ref<GaussianSplatRenderer>());
        return splat->renderer;
}
#endif

void GaussianSplatStorage::gaussian_set_aabb(RID p_rid, const AABB &p_aabb) {
        GaussianSplat *splat = gaussian_owner.get_or_null(p_rid);
        ERR_FAIL_NULL(splat);
        splat->aabb = p_aabb;
}

AABB GaussianSplatStorage::gaussian_get_aabb(RID p_rid) const {
	const GaussianSplat *splat = gaussian_owner.get_or_null(p_rid);
	ERR_FAIL_NULL_V(splat, AABB());
	return splat->aabb;
}

void GaussianSplatStorage::gaussian_set_casts_shadow(RID p_rid, bool p_casts_shadow) {
	GaussianSplat *splat = gaussian_owner.get_or_null(p_rid);
	ERR_FAIL_NULL(splat);
	splat->casts_shadow = p_casts_shadow;
}

bool GaussianSplatStorage::gaussian_get_casts_shadow(RID p_rid) const {
	const GaussianSplat *splat = gaussian_owner.get_or_null(p_rid);
	ERR_FAIL_NULL_V(splat, false);
	return splat->casts_shadow;
}
