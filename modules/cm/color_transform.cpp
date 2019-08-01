/*************************************************************************/
/*  color_transform.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "color_transform.h"
#include "core/image.h"
#include "servers/visual_server.h"

bool ColorTransform::is_valid() {
	return src_profile.is_valid() && src_profile->is_valid() && dst_profile.is_valid() && dst_profile->is_valid();
}

void ColorTransform::set_src_profile(Ref<ColorProfile> p_profile) {

	if (p_profile.is_null()) {
		ERR_PRINT("Source profile is null.");
		return;
	}
	src_profile = p_profile;
}

void ColorTransform::set_dst_profile(Ref<ColorProfile> p_profile) {

	if (p_profile.is_null()) {
		ERR_PRINT("Destination profile is null.");
		return;
	}
	dst_profile = p_profile;
}

void ColorTransform::set_intent(Intent p_intent) {

	intent = (cmsUInt32Number)p_intent;
}

void ColorTransform::set_black_point_compensation(bool p_black_point_compensation) {

	use_bpc = p_black_point_compensation;
}

bool ColorTransform::apply_screen_lut() {

	if (!is_valid()) {
		ERR_PRINT("Transform is not valid.");
		return false;
	}

	// hard-coded 256x256x256 size for now
	PoolVector<uint8_t> data;
	data.resize(4096 * 4096 * 3);

	{
		PoolVector<uint8_t>::Write write = data.write();

		// Fill image with identity data in source space
		int t = 0;
		for (int y = 0; y < 4096; y++) {
			for (int x = 0; x < 4096; x++) {
				write[t++] = x % 256; // R
				write[t++] = y % 256; // G
				write[t++] = y / 256 * 16 + x / 256; // B
			}
		}

		// Use LCMS to transform data
		cmsHPROFILE src = src_profile->get_profile_handle().profile; // handles owned by ColorProfile, don't free
		cmsHPROFILE dst = dst_profile->get_profile_handle().profile;
		if (src == NULL || dst == NULL) {
			ERR_PRINT("Transform has invalid profiles. This should have been checked earlier.");
			return false;
		}
		cmsUInt32Number flags = use_bpc ? cmsFLAGS_BLACKPOINTCOMPENSATION : 0;
		cmsHTRANSFORM transform = cmsCreateTransform(src, TYPE_RGB_8, dst, TYPE_RGB_8, intent, flags);
		if (transform == NULL) {
			ERR_PRINT("Failed to create lcms transform.");
			return false;
		}
		void *data_ptr = write.ptr();
		cmsDoTransform(transform, data_ptr, data_ptr, 4096 * 4096); // cmsDoTransform wants number of pixels
		cmsDeleteTransform(transform); // we don't need it after this one use
	}

	Ref<Image> lut = memnew(Image(4096, 4096, false, Image::FORMAT_RGB8, data));
	if (lut.is_null()) {
		ERR_PRINT("Failed to create LUT texture.");
		return false;
	}

	lut->set_name("LUT Texture");

	VS::get_singleton()->set_screen_lut(lut, 16, 16);
	return true;
}

ColorTransform::ColorTransform() {

	intent = CM_INTENT_PERCEPTUAL;
	use_bpc = true;
}

ColorTransform::ColorTransform(Ref<ColorProfile> p_src, Ref<ColorProfile> p_dst, Intent p_intent, bool p_black_point_compensation) {

	if (p_src.is_null() || !p_src->is_valid()) {
		ERR_PRINT("p_src is null or invalid.");
		return;
	}
	if (p_dst.is_null() || !p_dst->is_valid()) {
		ERR_PRINT("p_dst is null or invalid.");
		return;
	}

	src_profile = p_src;
	dst_profile = p_dst;
	intent = p_intent;
	use_bpc = p_black_point_compensation;
}

ColorTransform::~ColorTransform() {
}