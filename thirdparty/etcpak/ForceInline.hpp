#ifndef __FORCEINLINE_HPP__
#define __FORCEINLINE_HPP__

#if defined(__GNUC__)
#  define etcpak_force_inline __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#  define etcpak_force_inline __forceinline
#else
#  define etcpak_force_inline inline
#endif

#if defined(__GNUC__)
#  define etcpak_no_inline __attribute__((noinline))
#elif defined(_MSC_VER)
#  define etcpak_no_inline __declspec(noinline)
#else
#  define etcpak_no_inline
#endif

#endif
