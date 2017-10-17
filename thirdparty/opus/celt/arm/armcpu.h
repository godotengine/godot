/* Copyright (c) 2010 Xiph.Org Foundation
 * Copyright (c) 2013 Parrot */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if !defined(ARMCPU_H)
# define ARMCPU_H

# if defined(OPUS_ARM_MAY_HAVE_EDSP)
#  define MAY_HAVE_EDSP(name) name ## _edsp
# else
#  define MAY_HAVE_EDSP(name) name ## _c
# endif

# if defined(OPUS_ARM_MAY_HAVE_MEDIA)
#  define MAY_HAVE_MEDIA(name) name ## _media
# else
#  define MAY_HAVE_MEDIA(name) MAY_HAVE_EDSP(name)
# endif

# if defined(OPUS_ARM_MAY_HAVE_NEON)
#  define MAY_HAVE_NEON(name) name ## _neon
# else
#  define MAY_HAVE_NEON(name) MAY_HAVE_MEDIA(name)
# endif

# if defined(OPUS_ARM_PRESUME_EDSP)
#  define PRESUME_EDSP(name) name ## _edsp
# else
#  define PRESUME_EDSP(name) name ## _c
# endif

# if defined(OPUS_ARM_PRESUME_MEDIA)
#  define PRESUME_MEDIA(name) name ## _media
# else
#  define PRESUME_MEDIA(name) PRESUME_EDSP(name)
# endif

# if defined(OPUS_ARM_PRESUME_NEON)
#  define PRESUME_NEON(name) name ## _neon
# else
#  define PRESUME_NEON(name) PRESUME_MEDIA(name)
# endif

# if defined(OPUS_HAVE_RTCD)
int opus_select_arch(void);

#define OPUS_ARCH_ARM_V4    (0)
#define OPUS_ARCH_ARM_EDSP  (1)
#define OPUS_ARCH_ARM_MEDIA (2)
#define OPUS_ARCH_ARM_NEON  (3)

# endif

#endif
