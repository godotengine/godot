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

//#include "Etc.h"
#include "EtcColorFloatRGBA.h"
#include "EtcBlock4x4EncodingBits.h"
#include "EtcErrorMetric.h"


namespace Etc
{
	class Block4x4;
	class EncoderSpec;
	class SortedBlockList;

    class Image
    {
    public:

		//the differnt warning and errors that can come up during encoding
		enum  EncodingStatus
		{
			SUCCESS = 0,
			//
			WARNING_THRESHOLD = 1 << 0,
			//
			WARNING_EFFORT_OUT_OF_RANGE = 1 << 1,
			WARNING_JOBS_OUT_OF_RANGE = 1 << 2,
			WARNING_SOME_NON_OPAQUE_PIXELS = 1 << 3,//just for opaque formats, etc1, rgb8, r11, rg11
			WARNING_ALL_OPAQUE_PIXELS = 1 << 4,
			WARNING_ALL_TRANSPARENT_PIXELS = 1 << 5,
			WARNING_SOME_TRANSLUCENT_PIXELS = 1 << 6,//just for rgb8A1
			WARNING_SOME_RGBA_NOT_0_TO_1 = 1 << 7,
			WARNING_SOME_BLUE_VALUES_ARE_NOT_ZERO = 1 << 8,
			WARNING_SOME_GREEN_VALUES_ARE_NOT_ZERO = 1 << 9,
			//
			ERROR_THRESHOLD = 1 << 16,
			//
			ERROR_UNKNOWN_FORMAT = 1 << 17,
			ERROR_UNKNOWN_ERROR_METRIC = 1 << 18,
			ERROR_ZERO_WIDTH_OR_HEIGHT = 1 << 19,
			//
		};
		
		enum class Format
		{
			UNKNOWN,
			//
			ETC1,
			//
			// ETC2 formats
			RGB8,
			SRGB8,
			RGBA8,
			SRGBA8,
			R11,
			SIGNED_R11,
			RG11,
			SIGNED_RG11,
			RGB8A1,
			SRGB8A1,
			//
			FORMATS,
			//
			DEFAULT = SRGB8
		};

		// constructor using source image
		Image(float *a_pafSourceRGBA, unsigned int a_uiSourceWidth,
				unsigned int a_uiSourceHeight,
				ErrorMetric a_errormetric);

		// constructor using encoding bits
		Image(Format a_format, 
				unsigned int a_uiSourceWidth, unsigned int a_uiSourceHeight,
				unsigned char *a_paucEncidingBits, unsigned int a_uiEncodingBitsBytes,
				Image *a_pimageSource,
				ErrorMetric a_errormetric);

		~Image(void);

		EncodingStatus Encode(Format a_format, ErrorMetric a_errormetric, float a_fEffort, 
			unsigned int a_uiJobs, unsigned int a_uiMaxJobs);

		inline void AddToEncodingStatus(EncodingStatus a_encStatus)
		{
			m_encodingStatus = (EncodingStatus)((unsigned int)m_encodingStatus | (unsigned int)a_encStatus);
		}
		
		inline unsigned int GetSourceWidth(void)
		{
			return m_uiSourceWidth;
		}

		inline unsigned int GetSourceHeight(void)
		{
			return m_uiSourceHeight;
		}

		inline unsigned int GetExtendedWidth(void)
		{
			return m_uiExtendedWidth;
		}

		inline unsigned int GetExtendedHeight(void)
		{
			return m_uiExtendedHeight;
		}

		inline unsigned int GetNumberOfBlocks()
		{
			return m_uiBlockColumns * m_uiBlockRows;
		}

		inline Block4x4 * GetBlocks()
		{
			return m_pablock;
		}

		inline unsigned char * GetEncodingBits(void)
		{
			return m_paucEncodingBits;
		}

		inline unsigned int GetEncodingBitsBytes(void)
		{
			return m_uiEncodingBitsBytes;
		}

		inline int GetEncodingTimeMs(void)
		{
			return m_iEncodeTime_ms;
		}

		float GetError(void);

		inline ColorFloatRGBA * GetSourcePixel(unsigned int a_uiH, unsigned int a_uiV)
		{
			if (a_uiH >= m_uiSourceWidth || a_uiV >= m_uiSourceHeight)
			{
				return nullptr;
			}

			return &m_pafrgbaSource[a_uiV*m_uiSourceWidth + a_uiH];
		}

		inline Format GetFormat(void)
		{
			return m_format;
		}

		static Block4x4EncodingBits::Format DetermineEncodingBitsFormat(Format a_format);

		inline static unsigned short CalcExtendedDimension(unsigned short a_ushOriginalDimension)
		{
			return (unsigned short)((a_ushOriginalDimension + 3) & ~3);
		}

		inline ErrorMetric GetErrorMetric(void)
		{
			return m_errormetric;
		}

		static const char * EncodingFormatToString(Image::Format a_format);
		const char * EncodingFormatToString(void);
		//used to get basic information about the image data
		int m_iNumOpaquePixels;
		int m_iNumTranslucentPixels;
		int m_iNumTransparentPixels;

		ColorFloatRGBA m_numColorValues;
		ColorFloatRGBA m_numOutOfRangeValues;

		bool m_bVerboseOutput;
	private:
		//add a warning or error to check for while encoding
		inline void TrackEncodingWarning(EncodingStatus a_encStatus)
		{
			m_warningsToCapture = (EncodingStatus)((unsigned int)m_warningsToCapture | (unsigned int)a_encStatus);
		}

		//report the warning if it is something we care about for this encoding
		inline void AddToEncodingStatusIfSignfigant(EncodingStatus a_encStatus)
		{
			if ((EncodingStatus)((unsigned int)m_warningsToCapture & (unsigned int)a_encStatus) == a_encStatus)
			{
				AddToEncodingStatus(a_encStatus);
			}
		}

		Image(void);
		void FindEncodingWarningTypesForCurFormat();
		void FindAndSetEncodingWarnings();

		void InitBlocksAndBlockSorter(void);

		void RunFirstPass(unsigned int a_uiMultithreadingOffset, 
							unsigned int a_uiMultithreadingStride);

		void SetEncodingBits(unsigned int a_uiMultithreadingOffset,
								unsigned int a_uiMultithreadingStride);

		unsigned int IterateThroughWorstBlocks(unsigned int a_uiMaxBlocks,
												unsigned int a_uiMultithreadingOffset,
												unsigned int a_uiMultithreadingStride);

		// inputs
		ColorFloatRGBA *m_pafrgbaSource;
		unsigned int m_uiSourceWidth;
		unsigned int m_uiSourceHeight;
		unsigned int m_uiExtendedWidth;
		unsigned int m_uiExtendedHeight;
		unsigned int m_uiBlockColumns;
		unsigned int m_uiBlockRows;
		// intermediate data
		Block4x4 *m_pablock;
		// encoding
		Format m_format;
		Block4x4EncodingBits::Format m_encodingbitsformat;
		unsigned int m_uiEncodingBitsBytes;		// for entire image
		unsigned char *m_paucEncodingBits;
		ErrorMetric m_errormetric;
		float m_fEffort;
		// stats
		int m_iEncodeTime_ms;
		
		SortedBlockList *m_psortedblocklist;
		//this will hold any warning or errors that happen during encoding
		EncodingStatus m_encodingStatus;
		//these will be the warnings we are tracking
		EncodingStatus m_warningsToCapture;
	};

} // namespace Etc
