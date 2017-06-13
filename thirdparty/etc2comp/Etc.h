/*
 * Copyright 2015 The Etc2Comp Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "EtcConfig.h"
#include "EtcImage.h"
#include "EtcColor.h"
#include "EtcErrorMetric.h"
#include <memory>

#define ETCCOMP_MIN_EFFORT_LEVEL (0.0f)
#define ETCCOMP_DEFAULT_EFFORT_LEVEL (40.0f)
#define ETCCOMP_MAX_EFFORT_LEVEL (100.0f)

namespace Etc
{
	class Block4x4EncodingBits;

	struct RawImage
	{
		int uiExtendedWidth;
		int uiExtendedHeight;
		unsigned int uiEncodingBitsBytes;
		std::shared_ptr<unsigned char> paucEncodingBits;
	};



	// C-style inteface to the encoder
	void Encode(float *a_pafSourceRGBA,
				unsigned int a_uiSourceWidth,
				unsigned int a_uiSourceHeight,
				Image::Format a_format,
				ErrorMetric a_eErrMetric,
				float a_fEffort,
				unsigned int a_uiJobs,
				unsigned int a_uimaxJobs,
				unsigned char **a_ppaucEncodingBits,
				unsigned int *a_puiEncodingBitsBytes,
				unsigned int *a_puiExtendedWidth,
				unsigned int *a_puiExtendedHeight,
				int *a_piEncodingTime_ms, bool a_bVerboseOutput = false);

	void EncodeMipmaps(float *a_pafSourceRGBA,
		unsigned int a_uiSourceWidth,
		unsigned int a_uiSourceHeight,
		Image::Format a_format,
		ErrorMetric a_eErrMetric,
		float a_fEffort,
		unsigned int a_uiJobs,
		unsigned int a_uiMaxJobs,
		unsigned int a_uiMaxMipmaps,
		unsigned int a_uiMipFilterFlags,
		RawImage* a_pMipmaps,
		int *a_piEncodingTime_ms, bool a_bVerboseOutput = false);

}
