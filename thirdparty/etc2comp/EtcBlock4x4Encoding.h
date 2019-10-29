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

#include "EtcColorFloatRGBA.h"

#include "EtcErrorMetric.h"

#include <assert.h>
#include <float.h>

namespace Etc
{
	class Block4x4;

	// abstract base class for specific encodings
	class Block4x4Encoding
	{
	public:

		static const unsigned int ROWS = 4;
		static const unsigned int COLUMNS = 4;
		static const unsigned int PIXELS = ROWS * COLUMNS;
		static const float LUMA_WEIGHT;
		static const float CHROMA_BLUE_WEIGHT;

		typedef enum
		{
			MODE_UNKNOWN,
			//
			MODE_ETC1,
			MODE_T,
			MODE_H,
			MODE_PLANAR,
			MODE_R11,
			MODE_RG11,
			//
			MODES
		} Mode;

		Block4x4Encoding(void);
		//virtual ~Block4x4Encoding(void) =0;
		virtual ~Block4x4Encoding(void) {}
		virtual void InitFromSource(Block4x4 *a_pblockParent,
									ColorFloatRGBA *a_pafrgbaSource,

									unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric) = 0;

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
											unsigned char *a_paucEncodingBits,
											ColorFloatRGBA *a_pafrgbaSource,

											ErrorMetric a_errormetric) = 0;

		// perform an iteration of the encoding
		// the first iteration must generate a complete, valid (if poor) encoding
		virtual void PerformIteration(float a_fEffort) = 0;

		void CalcBlockError(void);

		inline float GetError(void)
		{
			assert(m_fError >= 0.0f);

			return m_fError;
		}

		inline ColorFloatRGBA * GetDecodedColors(void)
		{
			return m_afrgbaDecodedColors;
		}

		inline float * GetDecodedAlphas(void)
		{
			return m_afDecodedAlphas;
		}

		virtual void SetEncodingBits(void) = 0;

		virtual bool GetFlip(void) = 0;

		virtual bool IsDifferential(void) = 0;

		virtual bool HasSeverelyBentDifferentialColors(void) const = 0;

		inline Mode GetMode(void)
		{
			return m_mode;
		}

		inline bool IsDone(void)
		{
			return m_boolDone;
		}

		inline void SetDoneIfPerfect()
		{
			if (GetError() == 0.0f)
			{
				m_boolDone = true;
			}
		}

		float CalcPixelError(ColorFloatRGBA a_frgbaDecodedColor, float a_fDecodedAlpha,
								ColorFloatRGBA a_frgbaSourcePixel);

	protected:

		void Init(Block4x4 *a_pblockParent,
					ColorFloatRGBA *a_pafrgbaSource,

					ErrorMetric a_errormetric);

		Block4x4		*m_pblockParent;
		ColorFloatRGBA	*m_pafrgbaSource;

		bool			m_boolBorderPixels;				// if block has any border pixels

		ColorFloatRGBA	m_afrgbaDecodedColors[PIXELS];	// decoded RGB components, ignore Alpha
		float			m_afDecodedAlphas[PIXELS];		// decoded alpha component
		float			m_fError;						// error for RGBA relative to m_pafrgbaSource

		// intermediate encoding
		Mode			m_mode;

		unsigned int	m_uiEncodingIterations;
		bool			m_boolDone;						// all iterations have been done
		ErrorMetric		m_errormetric;

	private:

	};

} // namespace Etc
