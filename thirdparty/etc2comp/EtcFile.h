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
#include "EtcImage.h"
#include "Etc.h"

namespace Etc
{
	class FileHeader;
	class SourceImage;

	class File
	{
	public:

		enum class Format
		{
			INFER_FROM_FILE_EXTENSION,
			PKM,
			KTX,
		};

		File(const char *a_pstrFilename, Format a_fileformat, Image::Format a_imageformat,
				unsigned char *a_paucEncodingBits, unsigned int a_uiEncodingBitsBytes,
				unsigned int a_uiSourceWidth, unsigned int a_uiSourceHeight,
				unsigned int a_uiExtendedWidth, unsigned int a_uiExtendedHeight);

		File(const char *a_pstrFilename, Format a_fileformat, Image::Format a_imageformat,
			unsigned int a_uiNumMipmaps, RawImage *pMipmapImages,
			unsigned int a_uiSourceWidth, unsigned int a_uiSourceHeight );

		File(const char *a_pstrFilename, Format a_fileformat);
		~File();
		const char *GetFilename(void) { return m_pstrFilename; }

		void Read(const char *a_pstrFilename);
		void Write(void);

		inline unsigned int GetSourceWidth(void)
		{
			return m_uiSourceWidth;
		}

		inline unsigned int GetSourceHeight(void)
		{
			return m_uiSourceHeight;
		}

		inline unsigned int GetExtendedWidth(unsigned int mipmapIndex = 0)
		{
			if (mipmapIndex < m_uiNumMipmaps)
			{
				return m_pMipmapImages[mipmapIndex].uiExtendedWidth;
			}
			else
			{
				return 0;
			}
		}

		inline unsigned int GetExtendedHeight(unsigned int mipmapIndex = 0)
		{
			if (mipmapIndex < m_uiNumMipmaps)
			{
				return m_pMipmapImages[mipmapIndex].uiExtendedHeight;
			}
			else
			{
				return 0;
			}
		}

		inline Image::Format GetImageFormat()
		{
			return m_imageformat;
		}

		inline unsigned int GetEncodingBitsBytes(unsigned int mipmapIndex = 0)
		{
			if (mipmapIndex < m_uiNumMipmaps)
			{
				return m_pMipmapImages[mipmapIndex].uiEncodingBitsBytes;
			}
			else
			{
				return 0;
			}
		}

		inline unsigned char*  GetEncodingBits(unsigned int mipmapIndex = 0)
		{
			if( mipmapIndex < m_uiNumMipmaps)
			{
				return m_pMipmapImages[mipmapIndex].paucEncodingBits.get();
			}
			else
			{
				return nullptr;
			}
		}

		inline unsigned int GetNumMipmaps() 
		{
			return m_uiNumMipmaps; 
		}

		void UseSingleBlock(int a_iPixelX = -1, int a_iPixelY = -1);
	private:

		char *m_pstrFilename;               // includes directory path and file extension
		Format m_fileformat;
		Image::Format m_imageformat;
		FileHeader *m_pheader;
		unsigned int m_uiNumMipmaps;
		RawImage*	 m_pMipmapImages;
		unsigned int m_uiSourceWidth;
		unsigned int m_uiSourceHeight;
	};

}
