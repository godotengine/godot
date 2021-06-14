#ifndef SW_FT_TYPES_H
#define SW_FT_TYPES_H

/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Fixed                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    This type is used to store 16.16 fixed-point values, like scaling  */
/*    values or matrix coefficients.                                     */
/*                                                                       */
typedef signed long  SW_FT_Fixed;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Int                                                             */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for the int type.                                        */
/*                                                                       */
typedef signed int  SW_FT_Int;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_UInt                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for the unsigned int type.                               */
/*                                                                       */
typedef unsigned int  SW_FT_UInt;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Long                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for signed long.                                         */
/*                                                                       */
typedef signed long  SW_FT_Long;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_ULong                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for unsigned long.                                       */
/*                                                                       */
typedef unsigned long SW_FT_ULong;

/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Short                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef for signed short.                                        */
/*                                                                       */
typedef signed short  SW_FT_Short;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Byte                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A simple typedef for the _unsigned_ char type.                     */
/*                                                                       */
typedef unsigned char  SW_FT_Byte;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Bool                                                            */
/*                                                                       */
/* <Description>                                                         */
/*    A typedef of unsigned char, used for simple booleans.  As usual,   */
/*    values 1 and~0 represent true and false, respectively.             */
/*                                                                       */
typedef unsigned char  SW_FT_Bool;



/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Error                                                           */
/*                                                                       */
/* <Description>                                                         */
/*    The FreeType error code type.  A value of~0 is always interpreted  */
/*    as a successful operation.                                         */
/*                                                                       */
typedef int  SW_FT_Error;


/*************************************************************************/
/*                                                                       */
/* <Type>                                                                */
/*    SW_FT_Pos                                                             */
/*                                                                       */
/* <Description>                                                         */
/*    The type SW_FT_Pos is used to store vectorial coordinates.  Depending */
/*    on the context, these can represent distances in integer font      */
/*    units, or 16.16, or 26.6 fixed-point pixel coordinates.            */
/*                                                                       */
typedef signed long  SW_FT_Pos;


/*************************************************************************/
/*                                                                       */
/* <Struct>                                                              */
/*    SW_FT_Vector                                                          */
/*                                                                       */
/* <Description>                                                         */
/*    A simple structure used to store a 2D vector; coordinates are of   */
/*    the SW_FT_Pos type.                                                   */
/*                                                                       */
/* <Fields>                                                              */
/*    x :: The horizontal coordinate.                                    */
/*    y :: The vertical coordinate.                                      */
/*                                                                       */
typedef struct  SW_FT_Vector_
{
  SW_FT_Pos  x;
  SW_FT_Pos  y;

} SW_FT_Vector;


typedef long long int           SW_FT_Int64;
typedef unsigned long long int  SW_FT_UInt64;

typedef signed int              SW_FT_Int32;
typedef unsigned int            SW_FT_UInt32;


#define SW_FT_BOOL( x )  ( (SW_FT_Bool)( x ) )

#define SW_FT_SIZEOF_LONG 4

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE  0
#endif


#endif // SW_FT_TYPES_H
