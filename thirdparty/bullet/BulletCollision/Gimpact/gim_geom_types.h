#ifndef GIM_GEOM_TYPES_H_INCLUDED
#define GIM_GEOM_TYPES_H_INCLUDED

/*! \file gim_geom_types.h
\author Francisco Leon Najera
*/
/*
-----------------------------------------------------------------------------
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2006 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of EITHER:
   (1) The GNU Lesser General Public License as published by the Free
       Software Foundation; either version 2.1 of the License, or (at
       your option) any later version. The text of the GNU Lesser
       General Public License is included with this library in the
       file GIMPACT-LICENSE-LGPL.TXT.
   (2) The BSD-style license that is included with this library in
       the file GIMPACT-LICENSE-BSD.TXT.
   (3) The zlib/libpng license that is included with this library in
       the file GIMPACT-LICENSE-ZLIB.TXT.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 GIMPACT-LICENSE-LGPL.TXT, GIMPACT-LICENSE-ZLIB.TXT and GIMPACT-LICENSE-BSD.TXT for more details.

-----------------------------------------------------------------------------
*/

#include "gim_math.h"

//! Short Integer vector 2D
typedef GSHORT vec2s[2];
//! Integer vector 3D
typedef GSHORT vec3s[3];
//! Integer vector 4D
typedef GSHORT vec4s[4];

//! Short Integer vector 2D
typedef GUSHORT vec2us[2];
//! Integer vector 3D
typedef GUSHORT vec3us[3];
//! Integer vector 4D
typedef GUSHORT vec4us[4];

//! Integer vector 2D
typedef GINT vec2i[2];
//! Integer vector 3D
typedef GINT vec3i[3];
//! Integer vector 4D
typedef GINT vec4i[4];

//! Unsigned Integer vector 2D
typedef GUINT vec2ui[2];
//! Unsigned Integer vector 3D
typedef GUINT vec3ui[3];
//! Unsigned Integer vector 4D
typedef GUINT vec4ui[4];

//! Float vector 2D
typedef GREAL vec2f[2];
//! Float vector 3D
typedef GREAL vec3f[3];
//! Float vector 4D
typedef GREAL vec4f[4];

//! Double vector 2D
typedef GREAL2 vec2d[2];
//! Float vector 3D
typedef GREAL2 vec3d[3];
//! Float vector 4D
typedef GREAL2 vec4d[4];

//! Matrix 2D, row ordered
typedef GREAL mat2f[2][2];
//! Matrix 3D, row ordered
typedef GREAL mat3f[3][3];
//! Matrix 4D, row ordered
typedef GREAL mat4f[4][4];

//! Quaternion
typedef GREAL quatf[4];

//typedef struct _aabb3f aabb3f;

#endif  // GIM_GEOM_TYPES_H_INCLUDED
