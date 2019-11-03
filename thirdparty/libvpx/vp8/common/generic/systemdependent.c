/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "vpx_config.h"
#include "vp8_rtcd.h"
#if ARCH_ARM
#include "vpx_ports/arm.h"
#elif ARCH_X86 || ARCH_X86_64
#include "vpx_ports/x86.h"
#endif
#include "vp8/common/onyxc_int.h"
#include "vp8/common/systemdependent.h"

#if CONFIG_MULTITHREAD
#if HAVE_UNISTD_H && !defined(__OS2__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
typedef void (WINAPI *PGNSI)(LPSYSTEM_INFO);
#elif defined(__OS2__)
#define INCL_DOS
#define INCL_DOSSPINLOCK
#include <os2.h>
#endif
#endif

#if CONFIG_MULTITHREAD
static int get_cpu_count()
{
    int core_count = 16;

#if HAVE_UNISTD_H && !defined(__OS2__)
#if defined(_SC_NPROCESSORS_ONLN)
    core_count = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_SC_NPROC_ONLN)
    core_count = sysconf(_SC_NPROC_ONLN);
#endif
#elif defined(_WIN32)
    {
#if _WIN32_WINNT >= 0x0501
        SYSTEM_INFO sysinfo;
        GetNativeSystemInfo(&sysinfo);
#else
        PGNSI pGNSI;
        SYSTEM_INFO sysinfo;

        /* Call GetNativeSystemInfo if supported or
         * GetSystemInfo otherwise. */

        pGNSI = (PGNSI) GetProcAddress(
                GetModuleHandle(TEXT("kernel32.dll")), "GetNativeSystemInfo");
        if (pGNSI != NULL)
            pGNSI(&sysinfo);
        else
            GetSystemInfo(&sysinfo);
#endif

        core_count = sysinfo.dwNumberOfProcessors;
    }
#elif defined(__OS2__)
    {
        ULONG proc_id;
        ULONG status;

        core_count = 0;
        for (proc_id = 1; ; proc_id++)
        {
            if (DosGetProcessorStatus(proc_id, &status))
                break;

            if (status == PROC_ONLINE)
                core_count++;
        }
    }
#else
    /* other platforms */
#endif

    return core_count > 0 ? core_count : 1;
}
#endif

void vp8_clear_system_state_c() {};

void vp8_machine_specific_config(VP8_COMMON *ctx)
{
#if CONFIG_MULTITHREAD
    ctx->processor_core_count = get_cpu_count();
#else
    (void)ctx;
#endif /* CONFIG_MULTITHREAD */

#if ARCH_ARM
    ctx->cpu_caps = arm_cpu_caps();
#elif ARCH_X86 || ARCH_X86_64
    ctx->cpu_caps = x86_simd_caps();
#endif
}
