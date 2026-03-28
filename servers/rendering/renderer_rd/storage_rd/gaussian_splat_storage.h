/**************************************************************************/
/*  gaussian_splat_storage.h                                              */
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

#pragma once

#include "modules/modules_enabled.gen.h"
#include "core/templates/rid_owner.h"
#include "core/templates/vector.h"
#include "servers/rendering/storage/utilities.h"

#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
#include "modules/gaussian_splatting/renderer/gaussian_splat_renderer.h"
#endif

namespace RendererRD {

class GaussianSplatStorage {
	struct GaussianSplat {
#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
		Ref<GaussianSplatRenderer> renderer;
#endif
		AABB aabb;
		bool casts_shadow = false;
	};

        static GaussianSplatStorage *singleton;
        mutable RID_Owner<GaussianSplat> gaussian_owner;

public:
        static GaussianSplatStorage *get_singleton();

        GaussianSplatStorage();
        ~GaussianSplatStorage();

        RID gaussian_allocate();
        void gaussian_initialize(RID p_rid);
        void gaussian_free(RID p_rid);

        bool owns_gaussian(RID p_rid) const;

#ifdef MODULE_GAUSSIAN_SPLATTING_ENABLED
	void gaussian_set_renderer(RID p_rid, const Ref<GaussianSplatRenderer> &p_renderer);
	Ref<GaussianSplatRenderer> gaussian_get_renderer(RID p_rid) const;
#endif
	void gaussian_set_aabb(RID p_rid, const AABB &p_aabb);
	AABB gaussian_get_aabb(RID p_rid) const;
	void gaussian_set_casts_shadow(RID p_rid, bool p_casts_shadow);
	bool gaussian_get_casts_shadow(RID p_rid) const;
};

} // namespace RendererRD
