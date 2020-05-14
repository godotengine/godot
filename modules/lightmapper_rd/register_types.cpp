/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "register_types.h"

#include "core/project_settings.h"
#include "lightmapper_rd.h"
#include "scene/3d/lightmapper.h"

#ifndef _3D_DISABLED
static Lightmapper *create_lightmapper_rd() {
	return memnew(LightmapperRD);
}
#endif

void register_lightmapper_rd_types() {
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/low_quality_ray_count", 16);
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/medium_quality_ray_count", 64);
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/high_quality_ray_count", 256);
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/ultra_quality_ray_count", 1024);
	GLOBAL_DEF("rendering/gpu_lightmapper/performance/max_rays_per_pass", 32);
	GLOBAL_DEF("rendering/gpu_lightmapper/performance/region_size", 512);

	GLOBAL_DEF("rendering/gpu_lightmapper/quality/low_quality_probe_ray_count", 64);
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/medium_quality_probe_ray_count", 256);
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/high_quality_probe_ray_count", 512);
	GLOBAL_DEF("rendering/gpu_lightmapper/quality/ultra_quality_probe_ray_count", 2048);
	GLOBAL_DEF("rendering/gpu_lightmapper/performance/max_rays_per_probe_pass", 64);
#ifndef _3D_DISABLED
	ClassDB::register_class<LightmapperRD>();
	Lightmapper::create_gpu = create_lightmapper_rd;
#endif
}

void unregister_lightmapper_rd_types() {
}
