/*
 * Musepack audio compression
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#pragma once

// mpcenc.h
#define CENTER            448                   // offset for centering current data in Main-array
#define BLOCK            1152                   // blocksize
#define ANABUFFER    (BLOCK + CENTER)           // size of PCM-data array for analysis


typedef struct {
	float  L [36];
	float  R [36];
} SubbandFloatTyp;

typedef struct {
	float  L [ANABUFFER];
	float  R [ANABUFFER];
	float  M [ANABUFFER];
	float  S [ANABUFFER];
} PCMDataTyp;

