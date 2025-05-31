/* backtrace-supported.h.in -- Whether stack backtrace is supported.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   Written by Ian Lance Taylor, Google.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    (1) Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

    (3) The name of the author may not be used to
    endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.  */

/* The file backtrace-supported.h.in is used by configure to generate
   the file backtrace-supported.h.  The file backtrace-supported.h may
   be #include'd to see whether the backtrace library will be able to
   get a backtrace and produce symbolic information.  */


/* BACKTRACE_SUPPORTED will be #define'd as 1 if the backtrace library
   should work, 0 if it will not.  Libraries may #include this to make
   other arrangements.  */

#define BACKTRACE_SUPPORTED 1

/* BACKTRACE_USES_MALLOC will be #define'd as 1 if the backtrace
   library will call malloc as it works, 0 if it will call mmap
   instead.  This may be used to determine whether it is safe to call
   the backtrace functions from a signal handler.  In general this
   only applies to calls like backtrace and backtrace_pcinfo.  It does
   not apply to backtrace_simple, which never calls malloc.  It does
   not apply to backtrace_print, which always calls fprintf and
   therefore malloc.  */

#define BACKTRACE_USES_MALLOC 1

/* BACKTRACE_SUPPORTS_THREADS will be #define'd as 1 if the backtrace
   library is configured with threading support, 0 if not.  If this is
   0, the threaded parameter to backtrace_create_state must be passed
   as 0.  */

#define BACKTRACE_SUPPORTS_THREADS 1

/* BACKTRACE_SUPPORTS_DATA will be #defined'd as 1 if the backtrace_syminfo
   will work for variables.  It will always work for functions.  */

#define BACKTRACE_SUPPORTS_DATA 0
