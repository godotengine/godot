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

#include "EtcColor.h"
#include "EtcColorFloatRGBA.h"
#include "EtcErrorMetric.h"
#include "EtcImage.h"
#include "EtcBlock4x4Encoding.h"

namespace Etc
{
	class Block4x4EncodingBits;

	class Block4x4
	{
	public:

		static const unsigned int ROWS = 4;
		static const unsigned int COLUMNS = 4;
		static const unsigned int PIXELS = ROWS * COLUMNS;

		// the alpha mix for a 4x4 block of pixels
		enum class SourceAlphaMix
		{
			UNKNOWN,
			//
			OPAQUE,			// all 1.0
			TRANSPARENT,	// all 0.0 or NAN
			TRANSLUCENT		// not all opaque or transparent
		};

		typedef void (Block4x4::*EncoderFunctionPtr)(void);

		Block4x4(void);
		~Block4x4();
		void InitFromSource(Image *a_pimageSource,
							unsigned int a_uiSourceH,
							unsigned int a_uiSourceV,
							unsigned char *a_paucEncodingBits,
							ErrorMetric a_errormetric);

		void InitFromEtcEncodingBits(Image::Format a_imageformat,
										unsigned int a_uiSourceH,
										unsigned int a_uiSourceV,
										unsigned char *a_paucEncodingBits,
										Image *a_pimageSource,
										ErrorMetric a_errormetric);

		// return true if final iteration was performed
		inline void PerformEncodingIteration(float a_fEffort)
		{
			m_pencoding->PerformIteration(a_fEffort);
		}

		inline void SetEncodingBitsFromEncoding(void)
		{
			m_pencoding->SetEncodingBits();
		}

		inline unsigned int GetSourceH(void)
		{
			return m_uiSourceH;
		}

		inline unsigned int GetSourceV(void)
		{
			return m_uiSourceV;
		}

		inline float GetError(void)
		{
			return m_pencoding->GetError();
		}

		static const unsigned int s_auiPixelOrderHScan[PIXELS];

		inline ColorFloatRGBA * GetDecodedColors(void)
		{
			return m_pencoding->GetDecodedColors();
		}

		inline float * GetDecodedAlphas(void)
		{
			return m_pencoding->GetDecodedAlphas();
		}

		inline Block4x4Encoding::Mode GetEncodingMode(void)
		{
			return m_pencoding->GetMode();
		}

		inline bool GetFlip(void)
		{
			return m_pencoding->GetFlip();
		}

		inline bool IsDifferential(void)
		{
			return m_pencoding->IsDifferential();
		}

		inline ColorFloatRGBA * GetSource()
		{
			return m_afrgbaSource;
		}

		inline ErrorMetric GetErrorMetric()
		{
			return m_errormetric;
		}

		const char * GetEncodingModeName(void);

		inline Block4x4Encoding * GetEncoding(void)
		{
			return m_pencoding;
		}

		inline SourceAlphaMix GetSourceAlphaMix(void)
		{
			return m_sourcealphamix;
		}

		inline Image * GetImageSource(void)
		{
			return m_pimageSource;
		}

		inline bool HasBorderPixels(void)
		{
			return m_boolBorderPixels;
		}

		inline bool HasPunchThroughPixels(void)
		{
			return m_boolPunchThroughPixels;
		}

	private:

		void SetSourcePixels(void);

		Image				*m_pimageSource;
		unsigned int		m_uiSourceH;
		unsigned int		m_uiSourceV;
		ErrorMetric			m_errormetric;
		ColorFloatRGBA		m_afrgbaSource[PIXELS];		// vertical scan

		SourceAlphaMix		m_sourcealphamix;
		bool				m_boolBorderPixels;			// marked as rgba(NAN, NAN, NAN, NAN)
		bool				m_boolPunchThroughPixels;	// RGB8A1 or SRGB8A1 with any pixels with alpha < 0.5

		Block4x4Encoding	*m_pencoding;

	};

} // namespace Etc
