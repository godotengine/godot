/**************************************************************************/
/*  hddagi_screen_probe_svgf.h                                            */
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

#include "core/math/projection.h"
#include "core/math/transform_3d.h"
#include "servers/rendering/rendering_device.h"

class HDDAGIScreenProbeSVGF {
public:
	enum Quality {
		QUALITY_LOW,
		QUALITY_MEDIUM,
		QUALITY_HIGH,
		QUALITY_MAX,
	};

	struct FrameSettings {
		Projection projection;
		Projection previous_projection;
		Transform3D camera_transform;
		Transform3D previous_camera_transform;
		Vector2 taa_jitter;
		Vector2 previous_taa_jitter;
		Size2i size;
		float denoising_range = 500000.0f;
		Quality quality = QUALITY_HIGH;
		bool history_valid = false;
		bool specular = false;
		bool specular_full_resolution = false;
	};

	struct Resources {
		RID motion_vectors;
		RID normal_roughness;
		RID view_z;
		RID diffuse_radiance_hit_distance;
		RID output_diffuse_radiance_hit_distance;

		bool is_valid() const {
			return motion_vectors.is_valid() && normal_roughness.is_valid() && view_z.is_valid() &&
					diffuse_radiance_hit_distance.is_valid() && output_diffuse_radiance_hit_distance.is_valid() &&
					diffuse_radiance_hit_distance != output_diffuse_radiance_hit_distance &&
					motion_vectors != output_diffuse_radiance_hit_distance;
		}
	};

	static bool is_supported();
	static constexpr uint32_t get_atrous_iteration_count(Quality p_quality) {
		return p_quality == QUALITY_LOW ? 2u : (p_quality == QUALITY_MEDIUM ? 3u : 4u);
	}

	Error denoise(uint32_t p_view_id, const FrameSettings &p_frame, const Resources &p_resources);
	void clear();

	HDDAGIScreenProbeSVGF();
	~HDDAGIScreenProbeSVGF();

	HDDAGIScreenProbeSVGF(const HDDAGIScreenProbeSVGF &) = delete;
	HDDAGIScreenProbeSVGF &operator=(const HDDAGIScreenProbeSVGF &) = delete;

private:
	struct Implementation;
	Implementation *implementation = nullptr;
};
