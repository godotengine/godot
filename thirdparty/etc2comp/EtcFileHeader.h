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

#include "EtcFile.h"
#include <stdio.h>
#include <inttypes.h>

namespace Etc
{

	class Image;

	class FileHeader
	{
	public:

		virtual void Write(FILE *a_pfile) = 0;
		File GetFile();
		virtual ~FileHeader(void) {}
	protected:

		File *m_pfile;
	};

	// ----------------------------------------------------------------------------------------------------
	//
    class FileHeader_Pkm : public FileHeader
    {
    public:

		FileHeader_Pkm(File *a_pfile);

		virtual void Write(FILE *a_pfile);
		virtual ~FileHeader_Pkm(void) {}
	private:

		typedef struct
		{
			char m_acMagicNumber[4];
			char m_acVersion[2];
			unsigned char m_ucDataType_msb;             // e.g. ETC1_RGB_NO_MIPMAPS
			unsigned char m_ucDataType_lsb;
			unsigned char m_ucExtendedWidth_msb;     //  padded to 4x4 blocks
			unsigned char m_ucExtendedWidth_lsb;
			unsigned char m_ucExtendedHeight_msb;    //  padded to 4x4 blocks
			unsigned char m_ucExtendedHeight_lsb;
			unsigned char m_ucOriginalWidth_msb;
			unsigned char m_ucOriginalWidth_lsb;
			unsigned char m_ucOriginalHeight_msb;
			unsigned char m_ucOriginalHeight_lsb;
		} Data;

		Data m_data;
	};

	// ----------------------------------------------------------------------------------------------------
	//
    class FileHeader_Ktx : public FileHeader
    {
    public:

		typedef struct
		{
			uint32_t	u32KeyAndValueByteSize;
		} KeyValuePair;

		typedef struct
		{
			uint8_t m_au8Identifier[12];
			uint32_t m_u32Endianness;
			uint32_t m_u32GlType;
			uint32_t m_u32GlTypeSize;
			uint32_t m_u32GlFormat;
			uint32_t m_u32GlInternalFormat;
			uint32_t m_u32GlBaseInternalFormat;
			uint32_t m_u32PixelWidth;
			uint32_t m_u32PixelHeight;
			uint32_t m_u32PixelDepth;
			uint32_t m_u32NumberOfArrayElements;
			uint32_t m_u32NumberOfFaces;
			uint32_t m_u32NumberOfMipmapLevels;
			uint32_t m_u32BytesOfKeyValueData;
		} Data;

		enum class InternalFormat
		{
			ETC1_RGB8 = 0x8D64,
			ETC1_ALPHA8 = ETC1_RGB8,
			//
			ETC2_R11 = 0x9270,
			ETC2_SIGNED_R11 = 0x9271,
			ETC2_RG11 = 0x9272,
			ETC2_SIGNED_RG11 = 0x9273,
			ETC2_RGB8 = 0x9274,
			ETC2_SRGB8 = 0x9275,
			ETC2_RGB8A1 = 0x9276,
			ETC2_SRGB8_PUNCHTHROUGH_ALPHA1 = 0x9277,
			ETC2_RGBA8 = 0x9278
		};

		enum class BaseInternalFormat
		{
			ETC2_R11 = 0x1903,
			ETC2_RG11 = 0x8227,
			ETC1_RGB8 = 0x1907,
			ETC1_ALPHA8 = ETC1_RGB8,
			//
			ETC2_RGB8 = 0x1907,
			ETC2_RGB8A1 = 0x1908,
			ETC2_RGBA8 = 0x1908,
		};

		FileHeader_Ktx(File *a_pfile);

		virtual void Write(FILE *a_pfile);
		virtual ~FileHeader_Ktx(void) {}

		void AddKeyAndValue(KeyValuePair *a_pkeyvaluepair);

		Data* GetData();

	private:

		Data m_data;
		KeyValuePair *m_pkeyvaluepair;
		
		uint32_t m_u32Images;
		uint32_t m_u32KeyValuePairs;
	};

} // namespace Etc
