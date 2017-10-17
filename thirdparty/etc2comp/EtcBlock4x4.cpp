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

/* 
EtcBlock4x4.cpp

Implements the state associated with each 4x4 block of pixels in an image

Source images that are not a multiple of 4x4 are extended to fill the Block4x4 using pixels with an 
alpha of NAN

*/

#include "EtcConfig.h"
#include "EtcBlock4x4.h"

#include "EtcBlock4x4EncodingBits.h"
#include "EtcColor.h"
#include "EtcImage.h"
#include "EtcColorFloatRGBA.h"
#include "EtcBlock4x4Encoding_RGB8.h"
#include "EtcBlock4x4Encoding_RGBA8.h"
#include "EtcBlock4x4Encoding_RGB8A1.h"
#include "EtcBlock4x4Encoding_R11.h"
#include "EtcBlock4x4Encoding_RG11.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

namespace Etc
{
	// ETC pixels are scanned vertically.  
	// this mapping is for when someone wants to scan the ETC pixels horizontally
	const unsigned int Block4x4::s_auiPixelOrderHScan[PIXELS] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4::Block4x4(void)
	{
		m_pimageSource = nullptr;
		m_uiSourceH = 0;
		m_uiSourceV = 0;

		m_sourcealphamix = SourceAlphaMix::UNKNOWN;
		m_boolBorderPixels = false;
		m_boolPunchThroughPixels = false;

		m_pencoding = nullptr;

		m_errormetric = ErrorMetric::NUMERIC;

	}
	Block4x4::~Block4x4()
	{
		m_pimageSource = nullptr;
		if (m_pencoding)
		{
			delete m_pencoding;
			m_pencoding = nullptr;
		}
	}
	// ----------------------------------------------------------------------------------------------------
	// initialization prior to encoding from a source image
	// [a_uiSourceH,a_uiSourceV] is the location of the block in a_pimageSource
	// a_paucEncodingBits is the place to store the final encoding
	// a_errormetric is used for finding the best encoding
	//
	void Block4x4::InitFromSource(Image *a_pimageSource, 
									unsigned int a_uiSourceH, unsigned int a_uiSourceV,
									unsigned char *a_paucEncodingBits,
									ErrorMetric a_errormetric)
	{

		Block4x4();

		m_pimageSource = a_pimageSource;
		m_uiSourceH = a_uiSourceH;
		m_uiSourceV = a_uiSourceV;
		m_errormetric = a_errormetric;

		SetSourcePixels();

		// set block encoder function
		switch (m_pimageSource->GetFormat())
		{
		case Image::Format::ETC1:
			m_pencoding = new Block4x4Encoding_ETC1;
			break;

		case Image::Format::RGB8:
		case Image::Format::SRGB8:
			m_pencoding = new Block4x4Encoding_RGB8;
			break;

		case Image::Format::RGBA8:
		case Image::Format::SRGBA8:
			if (a_errormetric == RGBX)
			{
				m_pencoding = new Block4x4Encoding_RGBA8;
			}
			else
			{
				switch (m_sourcealphamix)
				{
				case SourceAlphaMix::OPAQUE:
					m_pencoding = new Block4x4Encoding_RGBA8_Opaque;
					break;

				case SourceAlphaMix::TRANSPARENT:
					m_pencoding = new Block4x4Encoding_RGBA8_Transparent;
					break;

				case SourceAlphaMix::TRANSLUCENT:
					m_pencoding = new Block4x4Encoding_RGBA8;
					break;

				default:
					assert(0);
					break;
				}
				break;
			}
			break;

		case Image::Format::RGB8A1:
		case Image::Format::SRGB8A1:
			switch (m_sourcealphamix)
			{
			case SourceAlphaMix::OPAQUE:
				m_pencoding = new Block4x4Encoding_RGB8A1_Opaque;
				break;

			case SourceAlphaMix::TRANSPARENT:
				m_pencoding = new Block4x4Encoding_RGB8A1_Transparent;
				break;

			case SourceAlphaMix::TRANSLUCENT:
				if (m_boolPunchThroughPixels)
				{
					m_pencoding = new Block4x4Encoding_RGB8A1;
				}
				else
				{
					m_pencoding = new Block4x4Encoding_RGB8A1_Opaque;
				}
				break;

			default:
				assert(0);
				break;
			}
			break;

		case Image::Format::R11:
		case Image::Format::SIGNED_R11:
			m_pencoding = new Block4x4Encoding_R11;
			break;
		case Image::Format::RG11:
		case Image::Format::SIGNED_RG11:
			m_pencoding = new Block4x4Encoding_RG11;
			break;
		default:
			assert(0);
			break;
		}

		m_pencoding->InitFromSource(this, m_afrgbaSource,
									a_paucEncodingBits, a_errormetric);

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization of encoding state from a prior encoding using encoding bits
	// [a_uiSourceH,a_uiSourceV] is the location of the block in a_pimageSource
	// a_paucEncodingBits is the place to read the prior encoding
	// a_imageformat is used to determine how to interpret a_paucEncodingBits
	// a_errormetric was used for the prior encoding
	//
	void Block4x4::InitFromEtcEncodingBits(Image::Format a_imageformat,
											unsigned int a_uiSourceH, unsigned int a_uiSourceV,
											unsigned char *a_paucEncodingBits,
											Image *a_pimageSource,
											ErrorMetric a_errormetric)
	{
		Block4x4();

		m_pimageSource = a_pimageSource;
		m_uiSourceH = a_uiSourceH;
		m_uiSourceV = a_uiSourceV;
		m_errormetric = a_errormetric;

		SetSourcePixels();

		// set block encoder function
		switch (a_imageformat)
		{
		case Image::Format::ETC1:
			m_pencoding = new Block4x4Encoding_ETC1;
			break;

		case Image::Format::RGB8:
		case Image::Format::SRGB8:
			m_pencoding = new Block4x4Encoding_RGB8;
			break;

		case Image::Format::RGBA8:
		case Image::Format::SRGBA8:
			m_pencoding = new Block4x4Encoding_RGBA8;
			break;

		case Image::Format::RGB8A1:
		case Image::Format::SRGB8A1:
			m_pencoding = new Block4x4Encoding_RGB8A1;
			break;

		case Image::Format::R11:
		case Image::Format::SIGNED_R11:
			m_pencoding = new Block4x4Encoding_R11;
			break;
		case Image::Format::RG11:
		case Image::Format::SIGNED_RG11:
			m_pencoding = new Block4x4Encoding_RG11;
			break;
		default:
			assert(0);
			break;
		}

		m_pencoding->InitFromEncodingBits(this, a_paucEncodingBits, m_afrgbaSource,
										m_pimageSource->GetErrorMetric());

	}
	
	// ----------------------------------------------------------------------------------------------------
	// set source pixels from m_pimageSource
	// set m_alphamix
	//
	void Block4x4::SetSourcePixels(void)
	{

		Image::Format imageformat = m_pimageSource->GetFormat();

		// alpha census
		unsigned int uiTransparentSourcePixels = 0;
		unsigned int uiOpaqueSourcePixels = 0;

		// copy source to consecutive memory locations
		// convert from image horizontal scan to block vertical scan
		unsigned int uiPixel = 0;
		for (unsigned int uiBlockPixelH = 0; uiBlockPixelH < Block4x4::COLUMNS; uiBlockPixelH++)
		{
			unsigned int uiSourcePixelH = m_uiSourceH + uiBlockPixelH;

			for (unsigned int uiBlockPixelV = 0; uiBlockPixelV < Block4x4::ROWS; uiBlockPixelV++)
			{
				unsigned int uiSourcePixelV = m_uiSourceV + uiBlockPixelV;

				ColorFloatRGBA *pfrgbaSource = m_pimageSource->GetSourcePixel(uiSourcePixelH, uiSourcePixelV);

				// if pixel extends beyond source image because of block padding
				if (pfrgbaSource == nullptr)
				{
					m_afrgbaSource[uiPixel] = ColorFloatRGBA(0.0f, 0.0f, 0.0f, NAN);	// denotes border pixel
					m_boolBorderPixels = true;
					uiTransparentSourcePixels++;
				}
				else
				{
					//get teh current pixel data, and store some of the attributes
					//before capping values to fit the encoder type
					
					m_afrgbaSource[uiPixel] = (*pfrgbaSource).ClampRGBA();

					if (m_afrgbaSource[uiPixel].fA == 1.0f || m_errormetric == RGBX)
					{
						m_pimageSource->m_iNumOpaquePixels++;
					}
					else if (m_afrgbaSource[uiPixel].fA == 0.0f)
					{
						m_pimageSource->m_iNumTransparentPixels++;
					}
					else if(m_afrgbaSource[uiPixel].fA > 0.0f && m_afrgbaSource[uiPixel].fA < 1.0f)
					{
						m_pimageSource->m_iNumTranslucentPixels++;
					}
					else
					{
						m_pimageSource->m_numOutOfRangeValues.fA++;
					}

					if (m_afrgbaSource[uiPixel].fR != 0.0f)
					{
						m_pimageSource->m_numColorValues.fR++;
						//make sure we are getting a float between 0-1
						if (m_afrgbaSource[uiPixel].fR - 1.0f > 0.0f)
						{
							m_pimageSource->m_numOutOfRangeValues.fR++;
						}
					}

					if (m_afrgbaSource[uiPixel].fG != 0.0f)
					{
						m_pimageSource->m_numColorValues.fG++;
						if (m_afrgbaSource[uiPixel].fG - 1.0f > 0.0f)
						{
							m_pimageSource->m_numOutOfRangeValues.fG++;
						}
					}
					if (m_afrgbaSource[uiPixel].fB != 0.0f)
					{
						m_pimageSource->m_numColorValues.fB++;
						if (m_afrgbaSource[uiPixel].fB - 1.0f > 0.0f)
						{
							m_pimageSource->m_numOutOfRangeValues.fB++;
						}
					}
					// for formats with no alpha, set source alpha to 1
					if (imageformat == Image::Format::ETC1 ||
						imageformat == Image::Format::RGB8 ||
						imageformat == Image::Format::SRGB8)
					{
						m_afrgbaSource[uiPixel].fA = 1.0f;
					}

					if (imageformat == Image::Format::R11 ||
						imageformat == Image::Format::SIGNED_R11)
					{
						m_afrgbaSource[uiPixel].fA = 1.0f;
						m_afrgbaSource[uiPixel].fG = 0.0f;
						m_afrgbaSource[uiPixel].fB = 0.0f;
					}

					if (imageformat == Image::Format::RG11 ||
						imageformat == Image::Format::SIGNED_RG11)
					{
						m_afrgbaSource[uiPixel].fA = 1.0f;
						m_afrgbaSource[uiPixel].fB = 0.0f;
					}

				
					// for RGB8A1, set source alpha to 0.0 or 1.0
					// set punch through flag
					if (imageformat == Image::Format::RGB8A1 ||
						imageformat == Image::Format::SRGB8A1)
					{
						if (m_afrgbaSource[uiPixel].fA >= 0.5f)
						{
							m_afrgbaSource[uiPixel].fA = 1.0f;
						}
						else
						{
							m_afrgbaSource[uiPixel].fA = 0.0f;
							m_boolPunchThroughPixels = true;
						}
					}

					if (m_afrgbaSource[uiPixel].fA == 1.0f || m_errormetric == RGBX)
					{
						uiOpaqueSourcePixels++;
					}
					else if (m_afrgbaSource[uiPixel].fA == 0.0f)
					{
						uiTransparentSourcePixels++;
					}

				}

				uiPixel += 1;
			}
		}

		if (uiOpaqueSourcePixels == PIXELS)
		{
			m_sourcealphamix = SourceAlphaMix::OPAQUE;
		}
		else if (uiTransparentSourcePixels == PIXELS)
		{
			m_sourcealphamix = SourceAlphaMix::TRANSPARENT;
		}
		else
		{
			m_sourcealphamix = SourceAlphaMix::TRANSLUCENT;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// return a name for the encoding mode
	//
	const char * Block4x4::GetEncodingModeName(void)
	{

		switch (m_pencoding->GetMode())
		{
		case Block4x4Encoding::MODE_ETC1:
			return "ETC1";
		case Block4x4Encoding::MODE_T:
			return "T";
		case Block4x4Encoding::MODE_H:
			return "H";
		case Block4x4Encoding::MODE_PLANAR:
			return "PLANAR";
		default:
			return "???";
		}
	}

	// ----------------------------------------------------------------------------------------------------
	//

}
