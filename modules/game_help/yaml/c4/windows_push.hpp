#ifndef _C4_WINDOWS_PUSH_HPP_
#define _C4_WINDOWS_PUSH_HPP_

/** @file windows_push.hpp sets up macros to include windows header files
 * without pulling in all of <windows.h>
 *
 * @see #include windows_pop.hpp to undefine these macros
 *
 * @see https://aras-p.info/blog/2018/01/12/Minimizing-windows.h/ */


#if defined(_WIN64) || defined(_WIN32)

#if defined(_M_AMD64)
#   ifndef _AMD64_
#       define _c4_AMD64_
#       define _AMD64_
#   endif
#elif defined(_M_IX86)
#   ifndef _X86_
#       define _c4_X86_
#       define _X86_
#   endif
#elif defined(_M_ARM64)
#   ifndef _ARM64_
#       define _c4_ARM64_
#       define _ARM64_
#   endif
#elif defined(_M_ARM)
#   ifndef _ARM_
#       define _c4_ARM_
#       define _ARM_
#   endif
#endif

#ifndef NOMINMAX
#    define _c4_NOMINMAX
#    define NOMINMAX
#endif

#ifndef NOGDI
#    define _c4_NOGDI
#    define NOGDI
#endif

#ifndef VC_EXTRALEAN
#    define _c4_VC_EXTRALEAN
#    define VC_EXTRALEAN
#endif

#ifndef WIN32_LEAN_AND_MEAN
#    define _c4_WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif

/*  If defined, the following flags inhibit definition
 *     of the indicated items.
 *
 *  NOGDICAPMASKS     - CC_*, LC_*, PC_*, CP_*, TC_*, RC_
 *  NOVIRTUALKEYCODES - VK_*
 *  NOWINMESSAGES     - WM_*, EM_*, LB_*, CB_*
 *  NOWINSTYLES       - WS_*, CS_*, ES_*, LBS_*, SBS_*, CBS_*
 *  NOSYSMETRICS      - SM_*
 *  NOMENUS           - MF_*
 *  NOICONS           - IDI_*
 *  NOKEYSTATES       - MK_*
 *  NOSYSCOMMANDS     - SC_*
 *  NORASTEROPS       - Binary and Tertiary raster ops
 *  NOSHOWWINDOW      - SW_*
 *  OEMRESOURCE       - OEM Resource values
 *  NOATOM            - Atom Manager routines
 *  NOCLIPBOARD       - Clipboard routines
 *  NOCOLOR           - Screen colors
 *  NOCTLMGR          - Control and Dialog routines
 *  NODRAWTEXT        - DrawText() and DT_*
 *  NOGDI             - All GDI defines and routines
 *  NOKERNEL          - All KERNEL defines and routines
 *  NOUSER            - All USER defines and routines
 *  NONLS             - All NLS defines and routines
 *  NOMB              - MB_* and MessageBox()
 *  NOMEMMGR          - GMEM_*, LMEM_*, GHND, LHND, associated routines
 *  NOMETAFILE        - typedef METAFILEPICT
 *  NOMINMAX          - Macros min(a,b) and max(a,b)
 *  NOMSG             - typedef MSG and associated routines
 *  NOOPENFILE        - OpenFile(), OemToAnsi, AnsiToOem, and OF_*
 *  NOSCROLL          - SB_* and scrolling routines
 *  NOSERVICE         - All Service Controller routines, SERVICE_ equates, etc.
 *  NOSOUND           - Sound driver routines
 *  NOTEXTMETRIC      - typedef TEXTMETRIC and associated routines
 *  NOWH              - SetWindowsHook and WH_*
 *  NOWINOFFSETS      - GWL_*, GCL_*, associated routines
 *  NOCOMM            - COMM driver routines
 *  NOKANJI           - Kanji support stuff.
 *  NOHELP            - Help engine interface.
 *  NOPROFILER        - Profiler interface.
 *  NODEFERWINDOWPOS  - DeferWindowPos routines
 *  NOMCX             - Modem Configuration Extensions
 */

#endif /* defined(_WIN64) || defined(_WIN32) */

#endif /* _C4_WINDOWS_PUSH_HPP_ */
