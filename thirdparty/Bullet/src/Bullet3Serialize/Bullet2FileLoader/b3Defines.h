/* Copyright (C) 2006-2009 Charlie C & Erwin Coumans http://gamekit.googlecode.com
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
*
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
*
* 1. The origin of this software must not be misrepresented; you must not
*    claim that you wrote the original software. If you use this software
*    in a product, an acknowledgment in the product documentation would be
*    appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
*    misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
#ifndef __B_DEFINES_H__
#define __B_DEFINES_H__


// MISC defines, see BKE_global.h, BKE_utildefines.h
#define B3_SIZEOFBLENDERHEADER 12


// ------------------------------------------------------------
#if defined(__sgi) || defined (__sparc) || defined (__sparc__) || defined (__PPC__) || defined (__ppc__) || defined (__BIG_ENDIAN__)
#	define B3_MAKE_ID(a,b,c,d) ( (int)(a)<<24 | (int)(b)<<16 | (c)<<8 | (d) )
#else
#	define B3_MAKE_ID(a,b,c,d) ( (int)(d)<<24 | (int)(c)<<16 | (b)<<8 | (a) )
#endif


// ------------------------------------------------------------
#if defined(__sgi) || defined(__sparc) || defined(__sparc__) || defined (__PPC__) || defined (__ppc__) || defined (__BIG_ENDIAN__)
#	define B3_MAKE_ID2(c, d) ( (c)<<8 | (d) )
#else
#	define B3_MAKE_ID2(c, d) ( (d)<<8 | (c) )
#endif

// ------------------------------------------------------------
#define B3_ID_SCE		B3_MAKE_ID2('S', 'C')
#define B3_ID_LI		B3_MAKE_ID2('L', 'I')
#define B3_ID_OB		B3_MAKE_ID2('O', 'B')
#define B3_ID_ME		B3_MAKE_ID2('M', 'E')
#define B3_ID_CU		B3_MAKE_ID2('C', 'U')
#define B3_ID_MB		B3_MAKE_ID2('M', 'B')
#define B3_ID_MA		B3_MAKE_ID2('M', 'A')
#define B3_ID_TE		B3_MAKE_ID2('T', 'E')
#define B3_ID_IM		B3_MAKE_ID2('I', 'M')
#define B3_ID_IK		B3_MAKE_ID2('I', 'K')
#define B3_ID_WV		B3_MAKE_ID2('W', 'V')
#define B3_ID_LT		B3_MAKE_ID2('L', 'T')
#define B3_ID_SE		B3_MAKE_ID2('S', 'E')
#define B3_ID_LF		B3_MAKE_ID2('L', 'F')
#define B3_ID_LA		B3_MAKE_ID2('L', 'A')
#define B3_ID_CA		B3_MAKE_ID2('C', 'A')
#define B3_ID_IP		B3_MAKE_ID2('I', 'P')
#define B3_ID_KE		B3_MAKE_ID2('K', 'E')
#define B3_ID_WO		B3_MAKE_ID2('W', 'O')
#define B3_ID_SCR		B3_MAKE_ID2('S', 'R')
#define B3_ID_VF		B3_MAKE_ID2('V', 'F')
#define B3_ID_TXT		B3_MAKE_ID2('T', 'X')
#define B3_ID_SO		B3_MAKE_ID2('S', 'O')
#define B3_ID_SAMPLE	B3_MAKE_ID2('S', 'A')
#define B3_ID_GR		B3_MAKE_ID2('G', 'R')
#define B3_ID_ID		B3_MAKE_ID2('I', 'D')
#define B3_ID_AR		B3_MAKE_ID2('A', 'R')
#define B3_ID_AC		B3_MAKE_ID2('A', 'C')
#define B3_ID_SCRIPT	B3_MAKE_ID2('P', 'Y')
#define B3_ID_FLUIDSIM	B3_MAKE_ID2('F', 'S')
#define B3_ID_NT		B3_MAKE_ID2('N', 'T')
#define B3_ID_BR		B3_MAKE_ID2('B', 'R')


#define B3_ID_SEQ		B3_MAKE_ID2('S', 'Q')
#define B3_ID_CO		B3_MAKE_ID2('C', 'O')
#define B3_ID_PO		B3_MAKE_ID2('A', 'C')
#define B3_ID_NLA		B3_MAKE_ID2('N', 'L')

#define B3_ID_VS		B3_MAKE_ID2('V', 'S')
#define B3_ID_VN		B3_MAKE_ID2('V', 'N')


// ------------------------------------------------------------
#define B3_FORM B3_MAKE_ID('F','O','R','M')
#define B3_DDG1 B3_MAKE_ID('3','D','G','1')
#define B3_DDG2 B3_MAKE_ID('3','D','G','2')
#define B3_DDG3 B3_MAKE_ID('3','D','G','3')
#define B3_DDG4 B3_MAKE_ID('3','D','G','4')
#define B3_GOUR B3_MAKE_ID('G','O','U','R')
#define B3_BLEN B3_MAKE_ID('B','L','E','N')
#define B3_DER_ B3_MAKE_ID('D','E','R','_')
#define B3_V100 B3_MAKE_ID('V','1','0','0')
#define B3_DATA B3_MAKE_ID('D','A','T','A')
#define B3_GLOB B3_MAKE_ID('G','L','O','B')
#define B3_IMAG B3_MAKE_ID('I','M','A','G')
#define B3_TEST B3_MAKE_ID('T','E','S','T')
#define B3_USER B3_MAKE_ID('U','S','E','R')


// ------------------------------------------------------------
#define B3_DNA1 B3_MAKE_ID('D','N','A','1')
#define B3_REND B3_MAKE_ID('R','E','N','D')
#define B3_ENDB B3_MAKE_ID('E','N','D','B')
#define B3_NAME B3_MAKE_ID('N','A','M','E')
#define B3_SDNA B3_MAKE_ID('S','D','N','A')
#define B3_TYPE B3_MAKE_ID('T','Y','P','E')
#define B3_TLEN B3_MAKE_ID('T','L','E','N')
#define B3_STRC B3_MAKE_ID('S','T','R','C')


// ------------------------------------------------------------
#define B3_SWITCH_INT(a) { \
    char s_i, *p_i; \
    p_i= (char *)&(a); \
    s_i=p_i[0]; p_i[0]=p_i[3]; p_i[3]=s_i; \
    s_i=p_i[1]; p_i[1]=p_i[2]; p_i[2]=s_i; }

// ------------------------------------------------------------
#define B3_SWITCH_SHORT(a)	{ \
    char s_i, *p_i; \
	p_i= (char *)&(a); \
	s_i=p_i[0]; p_i[0]=p_i[1]; p_i[1]=s_i; }

// ------------------------------------------------------------
#define B3_SWITCH_LONGINT(a) { \
    char s_i, *p_i; \
    p_i= (char *)&(a);  \
    s_i=p_i[0]; p_i[0]=p_i[7]; p_i[7]=s_i; \
    s_i=p_i[1]; p_i[1]=p_i[6]; p_i[6]=s_i; \
    s_i=p_i[2]; p_i[2]=p_i[5]; p_i[5]=s_i; \
    s_i=p_i[3]; p_i[3]=p_i[4]; p_i[4]=s_i; }

#endif//__B_DEFINES_H__
