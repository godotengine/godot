#ifndef _C4_WINDOWS_POP_HPP_
#define _C4_WINDOWS_POP_HPP_

#if defined(_WIN64) || defined(_WIN32)

#ifdef _c4_AMD64_
#    undef _c4_AMD64_
#    undef _AMD64_
#endif
#ifdef _c4_X86_
#    undef _c4_X86_
#    undef _X86_
#endif
#ifdef _c4_ARM_
#    undef _c4_ARM_
#    undef _ARM_
#endif

#ifdef _c4_NOMINMAX
#    undef _c4_NOMINMAX
#    undef NOMINMAX
#endif

#ifdef NOGDI
#    undef _c4_NOGDI
#    undef NOGDI
#endif

#ifdef VC_EXTRALEAN
#    undef _c4_VC_EXTRALEAN
#    undef VC_EXTRALEAN
#endif

#ifdef WIN32_LEAN_AND_MEAN
#    undef _c4_WIN32_LEAN_AND_MEAN
#    undef WIN32_LEAN_AND_MEAN
#endif

#endif /* defined(_WIN64) || defined(_WIN32) */

#endif /* _C4_WINDOWS_POP_HPP_ */
