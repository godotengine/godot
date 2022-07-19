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

#include <assert.h>

namespace Etc
{

	// ################################################################################
	// Block4x4EncodingBits
	// Base class for Block4x4EncodingBits_XXXX
	// ################################################################################

	class Block4x4EncodingBits
	{
	public:

		enum class Format
		{
			UNKNOWN,
			//
			RGB8,
			RGBA8,
			R11,
			RG11,
			RGB8A1,
			//
			FORMATS
		};

		static unsigned int GetBytesPerBlock(Format a_format)
		{
			switch (a_format)
			{
			case Format::RGB8:
			case Format::R11:
			case Format::RGB8A1:
				return 8;
				break;

			case Format::RGBA8:
			case Format::RG11:
				return 16;
				break;

			default:
				return 0;
				break;
			}

		}

	};

	// ################################################################################
	// Block4x4EncodingBits_RGB8
	// Encoding bits for the RGB portion of ETC1, RGB8, RGB8A1 and RGBA8
	// ################################################################################

	class Block4x4EncodingBits_RGB8
	{
	public:

		static const unsigned int BYTES_PER_BLOCK = 8;

		inline Block4x4EncodingBits_RGB8(void)
		{
			assert(sizeof(Block4x4EncodingBits_RGB8) == BYTES_PER_BLOCK);

			for (unsigned int uiByte = 0; uiByte < BYTES_PER_BLOCK; uiByte++)
			{
				auc[uiByte] = 0;
			}

		}

		typedef struct
		{
			unsigned red2 : 4;
			unsigned red1 : 4;
			//
			unsigned green2 : 4;
			unsigned green1 : 4;
			//
			unsigned blue2 : 4;
			unsigned blue1 : 4;
			//
			unsigned flip : 1;
			unsigned diff : 1;
			unsigned cw2 : 3;
			unsigned cw1 : 3;
			//
			unsigned int selectors;
		} Individual;

		typedef struct
		{
			signed dred2 : 3;
			unsigned red1 : 5;
			//
			signed dgreen2 : 3;
			unsigned green1 : 5;
			//
			signed dblue2 : 3;
			unsigned blue1 : 5;
			//
			unsigned flip : 1;
			unsigned diff : 1;
			unsigned cw2 : 3;
			unsigned cw1 : 3;
			//
			unsigned int selectors;
		} Differential;

		typedef struct
		{
			unsigned red1b : 2;
			unsigned detect2 : 1;
			unsigned red1a : 2;
			unsigned detect1 : 3;
			//
			unsigned blue1 : 4;
			unsigned green1 : 4;
			//
			unsigned green2 : 4;
			unsigned red2 : 4;
			//
			unsigned db : 1;
			unsigned diff : 1;
			unsigned da : 2;
			unsigned blue2 : 4;
			//
			unsigned int selectors;
		} T;

		typedef struct
		{
			unsigned green1a : 3;
			unsigned red1 : 4;
			unsigned detect1 : 1;
			//
			unsigned blue1b : 2;
			unsigned detect3 : 1;
			unsigned blue1a : 1;
			unsigned green1b : 1;
			unsigned detect2 : 3;
			//
			unsigned green2a : 3;
			unsigned red2 : 4;
			unsigned blue1c : 1;
			//
			unsigned db : 1;
			unsigned diff : 1;
			unsigned da : 1;
			unsigned blue2 : 4;
			unsigned green2b : 1;
			//
			unsigned int selectors;
		} H;

		typedef struct
		{
			unsigned originGreen1 : 1;
			unsigned originRed : 6;
			unsigned detect1 : 1;
			//
			unsigned originBlue1 : 1;
			unsigned originGreen2 : 6;
			unsigned detect2 : 1;
			//
			unsigned originBlue3 : 2;
			unsigned detect4 : 1;
			unsigned originBlue2 : 2;
			unsigned detect3 : 3;
			//
			unsigned horizRed2 : 1;
			unsigned diff : 1;
			unsigned horizRed1 : 5;
			unsigned originBlue4 : 1;
			//
			unsigned horizBlue1: 1;
			unsigned horizGreen : 7;
			//
			unsigned vertRed1 : 3;
			unsigned horizBlue2 : 5;
			//
			unsigned vertGreen1 : 5;
			unsigned vertRed2 : 3;
			//
			unsigned vertBlue : 6;
			unsigned vertGreen2 : 2;
		} Planar;

		union
		{
			unsigned char auc[BYTES_PER_BLOCK];
			unsigned long int ul;
			Individual individual;
			Differential differential;
			T t;
			H h;
			Planar planar;
		};

	};

	// ################################################################################
	// Block4x4EncodingBits_A8
	// Encoding bits for the A portion of RGBA8
	// ################################################################################

	class Block4x4EncodingBits_A8
	{
	public:

		static const unsigned int BYTES_PER_BLOCK = 8;
		static const unsigned int SELECTOR_BYTES = 6;

		typedef struct
		{
			unsigned base : 8;
			unsigned table : 4;
			unsigned multiplier : 4;
			unsigned selectors0 : 8;
			unsigned selectors1 : 8;
			unsigned selectors2 : 8;
			unsigned selectors3 : 8;
			unsigned selectors4 : 8;
			unsigned selectors5 : 8;
		} Data;

		Data data;

	};

	// ################################################################################
	// Block4x4EncodingBits_R11
	// Encoding bits for the R portion of R11
	// ################################################################################

	class Block4x4EncodingBits_R11
	{
	public:

		static const unsigned int BYTES_PER_BLOCK = 8;
		static const unsigned int SELECTOR_BYTES = 6;

		typedef struct
		{
			unsigned base : 8;
			unsigned table : 4;
			unsigned multiplier : 4;
			unsigned selectors0 : 8;
			unsigned selectors1 : 8;
			unsigned selectors2 : 8;
			unsigned selectors3 : 8;
			unsigned selectors4 : 8;
			unsigned selectors5 : 8;
		} Data;

		Data data;

	};

	class Block4x4EncodingBits_RG11
	{
	public:

		static const unsigned int BYTES_PER_BLOCK = 16;
		static const unsigned int SELECTOR_BYTES = 12;

		typedef struct
		{
			//Red portion
			unsigned baseR : 8;
			unsigned tableIndexR : 4;
			unsigned multiplierR : 4;
			unsigned selectorsR0 : 8;
			unsigned selectorsR1 : 8;
			unsigned selectorsR2 : 8;
			unsigned selectorsR3 : 8;
			unsigned selectorsR4 : 8;
			unsigned selectorsR5 : 8;
			//Green portion
			unsigned baseG : 8;
			unsigned tableIndexG : 4;
			unsigned multiplierG : 4;
			unsigned selectorsG0 : 8;
			unsigned selectorsG1 : 8;
			unsigned selectorsG2 : 8;
			unsigned selectorsG3 : 8;
			unsigned selectorsG4 : 8;
			unsigned selectorsG5 : 8;
		} Data;

		Data data;

	};

}
