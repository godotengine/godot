/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

// This file contains SDL replacements for functions in the C library

#if !defined(HAVE_LIBC) && !defined(SDL_STATIC_LIB)

// These are some C runtime intrinsics that need to be defined

#ifdef _MSC_VER

#ifndef __FLTUSED__
#define __FLTUSED__
__declspec(selectany) int _fltused = 1;
#endif

#ifdef _M_IX86

// Float to long
void __declspec(naked) _ftol()
{
    /* *INDENT-OFF* */
    __asm {
        push        ebp
        mov         ebp,esp
        sub         esp,20h
        and         esp,0FFFFFFF0h
        fld         st(0)
        fst         dword ptr [esp+18h]
        fistp       qword ptr [esp+10h]
        fild        qword ptr [esp+10h]
        mov         edx,dword ptr [esp+18h]
        mov         eax,dword ptr [esp+10h]
        test        eax,eax
        je          integer_QnaN_or_zero
arg_is_not_integer_QnaN:
        fsubp       st(1),st
        test        edx,edx
        jns         positive
        fstp        dword ptr [esp]
        mov         ecx,dword ptr [esp]
        xor         ecx,80000000h
        add         ecx,7FFFFFFFh
        adc         eax,0
        mov         edx,dword ptr [esp+14h]
        adc         edx,0
        jmp         localexit
positive:
        fstp        dword ptr [esp]
        mov         ecx,dword ptr [esp]
        add         ecx,7FFFFFFFh
        sbb         eax,0
        mov         edx,dword ptr [esp+14h]
        sbb         edx,0
        jmp         localexit
integer_QnaN_or_zero:
        mov         edx,dword ptr [esp+14h]
        test        edx,7FFFFFFFh
        jne         arg_is_not_integer_QnaN
        fstp        dword ptr [esp+18h]
        fstp        dword ptr [esp+18h]
localexit:
        leave
        ret
    }
    /* *INDENT-ON* */
}

void _ftol2_sse()
{
    _ftol();
}

void _ftol2()
{
    _ftol();
}

// 64-bit math operators for 32-bit systems
void __declspec(naked) _allmul()
{
    /* *INDENT-OFF* */
    __asm {
        mov         eax, dword ptr[esp+8]
        mov         ecx, dword ptr[esp+10h]
        or          ecx, eax
        mov         ecx, dword ptr[esp+0Ch]
        jne         hard
        mov         eax, dword ptr[esp+4]
        mul         ecx
        ret         10h
hard:
        push        ebx
        mul         ecx
        mov         ebx, eax
        mov         eax, dword ptr[esp+8]
        mul         dword ptr[esp+14h]
        add         ebx, eax
        mov         eax, dword ptr[esp+8]
        mul         ecx
        add         edx, ebx
        pop         ebx
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _alldiv()
{
    /* *INDENT-OFF* */
    __asm {
        push        edi
        push        esi
        push        ebx
        xor         edi,edi
        mov         eax,dword ptr [esp+14h]
        or          eax,eax
        jge         L1
        inc         edi
        mov         edx,dword ptr [esp+10h]
        neg         eax
        neg         edx
        sbb         eax,0
        mov         dword ptr [esp+14h],eax
        mov         dword ptr [esp+10h],edx
L1:
        mov         eax,dword ptr [esp+1Ch]
        or          eax,eax
        jge         L2
        inc         edi
        mov         edx,dword ptr [esp+18h]
        neg         eax
        neg         edx
        sbb         eax,0
        mov         dword ptr [esp+1Ch],eax
        mov         dword ptr [esp+18h],edx
L2:
        or          eax,eax
        jne         L3
        mov         ecx,dword ptr [esp+18h]
        mov         eax,dword ptr [esp+14h]
        xor         edx,edx
        div         ecx
        mov         ebx,eax
        mov         eax,dword ptr [esp+10h]
        div         ecx
        mov         edx,ebx
        jmp         L4
L3:
        mov         ebx,eax
        mov         ecx,dword ptr [esp+18h]
        mov         edx,dword ptr [esp+14h]
        mov         eax,dword ptr [esp+10h]
L5:
        shr         ebx,1
        rcr         ecx,1
        shr         edx,1
        rcr         eax,1
        or          ebx,ebx
        jne         L5
        div         ecx
        mov         esi,eax
        mul         dword ptr [esp+1Ch]
        mov         ecx,eax
        mov         eax,dword ptr [esp+18h]
        mul         esi
        add         edx,ecx
        jb          L6
        cmp         edx,dword ptr [esp+14h]
        ja          L6
        jb          L7
        cmp         eax,dword ptr [esp+10h]
        jbe         L7
L6:
        dec         esi
L7:
        xor         edx,edx
        mov         eax,esi
L4:
        dec         edi
        jne         L8
        neg         edx
        neg         eax
        sbb         edx,0
L8:
        pop         ebx
        pop         esi
        pop         edi
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _aulldiv()
{
    /* *INDENT-OFF* */
    __asm {
        push        ebx
        push        esi
        mov         eax,dword ptr [esp+18h]
        or          eax,eax
        jne         L1
        mov         ecx,dword ptr [esp+14h]
        mov         eax,dword ptr [esp+10h]
        xor         edx,edx
        div         ecx
        mov         ebx,eax
        mov         eax,dword ptr [esp+0Ch]
        div         ecx
        mov         edx,ebx
        jmp         L2
L1:
        mov         ecx,eax
        mov         ebx,dword ptr [esp+14h]
        mov         edx,dword ptr [esp+10h]
        mov         eax,dword ptr [esp+0Ch]
L3:
        shr         ecx,1
        rcr         ebx,1
        shr         edx,1
        rcr         eax,1
        or          ecx,ecx
        jne         L3
        div         ebx
        mov         esi,eax
        mul         dword ptr [esp+18h]
        mov         ecx,eax
        mov         eax,dword ptr [esp+14h]
        mul         esi
        add         edx,ecx
        jb          L4
        cmp         edx,dword ptr [esp+10h]
        ja          L4
        jb          L5
        cmp         eax,dword ptr [esp+0Ch]
        jbe         L5
L4:
        dec         esi
L5:
        xor         edx,edx
        mov         eax,esi
L2:
        pop         esi
        pop         ebx
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _allrem()
{
    /* *INDENT-OFF* */
    __asm {
        push        ebx
        push        edi
        xor         edi,edi
        mov         eax,dword ptr [esp+10h]
        or          eax,eax
        jge         L1
        inc         edi
        mov         edx,dword ptr [esp+0Ch]
        neg         eax
        neg         edx
        sbb         eax,0
        mov         dword ptr [esp+10h],eax
        mov         dword ptr [esp+0Ch],edx
L1:
        mov         eax,dword ptr [esp+18h]
        or          eax,eax
        jge         L2
        mov         edx,dword ptr [esp+14h]
        neg         eax
        neg         edx
        sbb         eax,0
        mov         dword ptr [esp+18h],eax
        mov         dword ptr [esp+14h],edx
L2:
        or          eax,eax
        jne         L3
        mov         ecx,dword ptr [esp+14h]
        mov         eax,dword ptr [esp+10h]
        xor         edx,edx
        div         ecx
        mov         eax,dword ptr [esp+0Ch]
        div         ecx
        mov         eax,edx
        xor         edx,edx
        dec         edi
        jns         L4
        jmp         L8
L3:
        mov         ebx,eax
        mov         ecx,dword ptr [esp+14h]
        mov         edx,dword ptr [esp+10h]
        mov         eax,dword ptr [esp+0Ch]
L5:
        shr         ebx,1
        rcr         ecx,1
        shr         edx,1
        rcr         eax,1
        or          ebx,ebx
        jne         L5
        div         ecx
        mov         ecx,eax
        mul         dword ptr [esp+18h]
        xchg        eax,ecx
        mul         dword ptr [esp+14h]
        add         edx,ecx
        jb          L6
        cmp         edx,dword ptr [esp+10h]
        ja          L6
        jb          L7
        cmp         eax,dword ptr [esp+0Ch]
        jbe         L7
L6:
        sub         eax,dword ptr [esp+14h]
        sbb         edx,dword ptr [esp+18h]
L7:
        sub         eax,dword ptr [esp+0Ch]
        sbb         edx,dword ptr [esp+10h]
        dec         edi
        jns         L8
L4:
        neg         edx
        neg         eax
        sbb         edx,0
L8:
        pop         edi
        pop         ebx
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _aullrem()
{
    /* *INDENT-OFF* */
    __asm {
        push        ebx
        mov         eax,dword ptr [esp+14h]
        or          eax,eax
        jne         L1
        mov         ecx,dword ptr [esp+10h]
        mov         eax,dword ptr [esp+0Ch]
        xor         edx,edx
        div         ecx
        mov         eax,dword ptr [esp+8]
        div         ecx
        mov         eax,edx
        xor         edx,edx
        jmp         L2
L1:
        mov         ecx,eax
        mov         ebx,dword ptr [esp+10h]
        mov         edx,dword ptr [esp+0Ch]
        mov         eax,dword ptr [esp+8]
L3:
        shr         ecx,1
        rcr         ebx,1
        shr         edx,1
        rcr         eax,1
        or          ecx,ecx
        jne         L3
        div         ebx
        mov         ecx,eax
        mul         dword ptr [esp+14h]
        xchg        eax,ecx
        mul         dword ptr [esp+10h]
        add         edx,ecx
        jb          L4
        cmp         edx,dword ptr [esp+0Ch]
        ja          L4
        jb          L5
        cmp         eax,dword ptr [esp+8]
        jbe         L5
L4:
        sub         eax,dword ptr [esp+10h]
        sbb         edx,dword ptr [esp+14h]
L5:
        sub         eax,dword ptr [esp+8]
        sbb         edx,dword ptr [esp+0Ch]
        neg         edx
        neg         eax
        sbb         edx,0
L2:
        pop         ebx
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _alldvrm()
{
    /* *INDENT-OFF* */
    __asm {
        push        edi
        push        esi
        push        ebp
        xor         edi,edi
        xor         ebp,ebp
        mov         eax,dword ptr [esp+14h]
        or          eax,eax
        jge         L1
        inc         edi
        inc         ebp
        mov         edx,dword ptr [esp+10h]
        neg         eax
        neg         edx
        sbb         eax,0
        mov         dword ptr [esp+14h],eax
        mov         dword ptr [esp+10h],edx
L1:
        mov         eax,dword ptr [esp+1Ch]
        or          eax,eax
        jge         L2
        inc         edi
        mov         edx,dword ptr [esp+18h]
        neg         eax
        neg         edx
        sbb         eax,0
        mov         dword ptr [esp+1Ch],eax
        mov         dword ptr [esp+18h],edx
L2:
        or          eax,eax
        jne         L3
        mov         ecx,dword ptr [esp+18h]
        mov         eax,dword ptr [esp+14h]
        xor         edx,edx
        div         ecx
        mov         ebx,eax
        mov         eax,dword ptr [esp+10h]
        div         ecx
        mov         esi,eax
        mov         eax,ebx
        mul         dword ptr [esp+18h]
        mov         ecx,eax
        mov         eax,esi
        mul         dword ptr [esp+18h]
        add         edx,ecx
        jmp         L4
L3:
        mov         ebx,eax
        mov         ecx,dword ptr [esp+18h]
        mov         edx,dword ptr [esp+14h]
        mov         eax,dword ptr [esp+10h]
L5:
        shr         ebx,1
        rcr         ecx,1
        shr         edx,1
        rcr         eax,1
        or          ebx,ebx
        jne         L5
        div         ecx
        mov         esi,eax
        mul         dword ptr [esp+1Ch]
        mov         ecx,eax
        mov         eax,dword ptr [esp+18h]
        mul         esi
        add         edx,ecx
        jb          L6
        cmp         edx,dword ptr [esp+14h]
        ja          L6
        jb          L7
        cmp         eax,dword ptr [esp+10h]
        jbe         L7
L6:
        dec         esi
        sub         eax,dword ptr [esp+18h]
        sbb         edx,dword ptr [esp+1Ch]
L7:
        xor         ebx,ebx
L4:
        sub         eax,dword ptr [esp+10h]
        sbb         edx,dword ptr [esp+14h]
        dec         ebp
        jns         L9
        neg         edx
        neg         eax
        sbb         edx,0
L9:
        mov         ecx,edx
        mov         edx,ebx
        mov         ebx,ecx
        mov         ecx,eax
        mov         eax,esi
        dec         edi
        jne         L8
        neg         edx
        neg         eax
        sbb         edx,0
L8:
        pop         ebp
        pop         esi
        pop         edi
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _aulldvrm()
{
    /* *INDENT-OFF* */
    __asm {
        push        esi
        mov         eax,dword ptr [esp+14h]
        or          eax,eax
        jne         L1
        mov         ecx,dword ptr [esp+10h]
        mov         eax,dword ptr [esp+0Ch]
        xor         edx,edx
        div         ecx
        mov         ebx,eax
        mov         eax,dword ptr [esp+8]
        div         ecx
        mov         esi,eax
        mov         eax,ebx
        mul         dword ptr [esp+10h]
        mov         ecx,eax
        mov         eax,esi
        mul         dword ptr [esp+10h]
        add         edx,ecx
        jmp         L2
L1:
        mov         ecx,eax
        mov         ebx,dword ptr [esp+10h]
        mov         edx,dword ptr [esp+0Ch]
        mov         eax,dword ptr [esp+8]
L3:
        shr         ecx,1
        rcr         ebx,1
        shr         edx,1
        rcr         eax,1
        or          ecx,ecx
        jne         L3
        div         ebx
        mov         esi,eax
        mul         dword ptr [esp+14h]
        mov         ecx,eax
        mov         eax,dword ptr [esp+10h]
        mul         esi
        add         edx,ecx
        jb          L4
        cmp         edx,dword ptr [esp+0Ch]
        ja          L4
        jb          L5
        cmp         eax,dword ptr [esp+8]
        jbe         L5
L4:
        dec         esi
        sub         eax,dword ptr [esp+10h]
        sbb         edx,dword ptr [esp+14h]
L5:
        xor         ebx,ebx
L2:
        sub         eax,dword ptr [esp+8]
        sbb         edx,dword ptr [esp+0Ch]
        neg         edx
        neg         eax
        sbb         edx,0
        mov         ecx,edx
        mov         edx,ebx
        mov         ebx,ecx
        mov         ecx,eax
        mov         eax,esi
        pop         esi
        ret         10h
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _allshl()
{
    /* *INDENT-OFF* */
    __asm {
        cmp         cl,40h
        jae         RETZERO
        cmp         cl,20h
        jae         MORE32
        shld        edx,eax,cl
        shl         eax,cl
        ret
MORE32:
        mov         edx,eax
        xor         eax,eax
        and         cl,1Fh
        shl         edx,cl
        ret
RETZERO:
        xor         eax,eax
        xor         edx,edx
        ret
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _allshr()
{
    /* *INDENT-OFF* */
    __asm {
        cmp         cl,3Fh
        jae         RETSIGN
        cmp         cl,20h
        jae         MORE32
        shrd        eax,edx,cl
        sar         edx,cl
        ret
MORE32:
        mov         eax,edx
        sar         edx,1Fh
        and         cl,1Fh
        sar         eax,cl
        ret
RETSIGN:
        sar         edx,1Fh
        mov         eax,edx
        ret
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _aullshr()
{
    /* *INDENT-OFF* */
    __asm {
        cmp         cl,40h
        jae         RETZERO
        cmp         cl,20h
        jae         MORE32
        shrd        eax,edx,cl
        shr         edx,cl
        ret
MORE32:
        mov         eax,edx
        xor         edx,edx
        and         cl,1Fh
        shr         eax,cl
        ret
RETZERO:
        xor         eax,eax
        xor         edx,edx
        ret
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _chkstk(void)
{
    __asm {
        push        ecx
        mov         ecx,esp     ; lea         ecx,dword ptr [esp]+4
        add         ecx,4
        sub         ecx,eax
        sbb         eax,eax
        not         eax
        and         ecx,eax
        mov         eax,esp
        and         eax,0xfffff000
L1:
        cmp         ecx,eax
        jb          short L2
        mov         eax,ecx
        pop         ecx
        xchg        esp,eax
        mov         eax,dword ptr [eax]
        mov         dword ptr [esp],eax
        ret
L2:
        sub         eax,0x1000
        test        dword ptr [eax],eax
        jmp         short L1
    }
}

void __declspec(naked) _alloca_probe_8(void)
{
    /* *INDENT-OFF* */
    __asm {
        push        ecx
        mov         ecx,esp     ; lea         ecx,dword ptr [esp]+8
        add         ecx,8
        sub         ecx,eax
        and         ecx,0x7
        add         eax,ecx
        sbb         ecx,ecx
        or          eax,ecx
        pop         ecx
        jmp         _chkstk
    }
    /* *INDENT-ON* */
}

void __declspec(naked) _alloca_probe_16(void)
{
    /* *INDENT-OFF* */
    __asm {
        push        ecx
        mov         ecx,esp     ; lea         ecx,dword ptr [esp]+8
        add         ecx,8
        sub         ecx,eax
        and         ecx,0xf
        add         eax,ecx
        sbb         ecx,ecx
        or          eax,ecx
        pop         ecx
        jmp         _chkstk
    }
    /* *INDENT-ON* */
}

#endif // _M_IX86

#endif // MSC_VER

#ifdef __ICL
/* The classic Intel compiler generates calls to _intel_fast_memcpy
 * and _intel_fast_memset when building an optimized SDL library */
void *_intel_fast_memcpy(void *dst, const void *src, size_t len)
{
    return SDL_memcpy(dst, src, len);
}
void *_intel_fast_memset(void *dst, int c, size_t len)
{
    return SDL_memset(dst, c, len);
}
#endif

#endif // !HAVE_LIBC && !SDL_STATIC_LIB
