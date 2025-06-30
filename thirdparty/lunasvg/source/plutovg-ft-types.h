/****************************************************************************
 *
 * fttypes.h
 *
 *   FreeType simple types definitions (specification only).
 *
 * Copyright (C) 1996-2020 by
 * David Turner, Robert Wilhelm, and Werner Lemberg.
 *
 * This file is part of the FreeType project, and may only be used,
 * modified, and distributed under the terms of the FreeType project
 * license, FTL.TXT.  By continuing to use, modify, or distribute
 * this file you indicate that you have read the license and
 * understand and accept it fully.
 *
 */

#ifndef PLUTOVG_FT_TYPES_H
#define PLUTOVG_FT_TYPES_H

/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Fixed                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    This type is used to store 16.16 fixed-point values, like scaling  */
/*    values or matrix coefficients.                                     */
/*                                                                       */
typedef signed long  PVG_FT_Fixed;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Int                                                             */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for the int type.                                        */
/*                                                                       */
typedef signed int  PVG_FT_Int;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_UInt                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for the unsigned int type.                               */
/*                                                                       */
typedef unsigned int  PVG_FT_UInt;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Long                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for signed long.                                         */
/*                                                                       */
typedef signed long  PVG_FT_Long;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_ULong                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for unsigned long.                                       */
/*                                                                       */
typedef unsigned long PVG_FT_ULong;

/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Short                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for signed short.                                        */
/*                                                                       */
typedef signed short  PVG_FT_Short;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Byte                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A simple typedef for the _unsigned_ char type.                     */
/*                                                                       */
typedef unsigned char  PVG_FT_Byte;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Bool                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef of unsigned char, used for simple booleans.  As usual,   */
/*    values 1 and~0 represent true and false, respectively.             */
/*                                                                       */
typedef unsigned char  PVG_FT_Bool;



/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Error                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    The FreeType error code type.  A value of~0 is always interpreted  */
/*    as a successful operation.                                         */
/*                                                                       */
typedef int  PVG_FT_Error;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    PVG_FT_Pos                                                             */
/*                                                                       */
/* <Description>                                                         */
/*    The type PVG_FT_Pos is used to store vectorial coordinates.  Depending */
/*    on the context, these can represent distances in integer font      */
/*    units, or 16.16, or 26.6 fixed-point pixel coordinates.            */
/*                                                                       */
typedef signed long  PVG_FT_Pos;


/*************************************************************************/
/*                                                                       */
/* <Struct>                                                              */
/*    PVG_FT_Vector                                                          */
/*                                                                       */
/* <Description>                                                         */
/*    A simple structure used to store a 2D vector; coordinates are of   */
/*    the PVG_FT_Pos type.                                                   */
/*                                                                       */
/* <Fields>                                                              */
/*    x :: The horizontal coordinate.                                    */
/*    y :: The vertical coordinate.                                      */
/*                                                                       */
typedef struct  PVG_FT_Vector_
{
    PVG_FT_Pos  x;
    PVG_FT_Pos  y;

} PVG_FT_Vector;


typedef long long int           PVG_FT_Int64;
typedef unsigned long long int  PVG_FT_UInt64;

typedef signed int              PVG_FT_Int32;
typedef unsigned int            PVG_FT_UInt32;

#define PVG_FT_BOOL( x )  ( (PVG_FT_Bool)( x ) )

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE  0
#endif

#endif // PLUTOVG_FT_TYPES_H
