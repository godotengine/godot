/* fficonfig.h.  Generated from fficonfig.h.in by configure.  */
/* fficonfig.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Define to the flags needed for the .section .eh_frame directive. */
/* #define EH_FRAME_FLAGS "a" */

/* Define this if you want extra debugging. */
/* #undef FFI_DEBUG */

/* Define this if you want statically defined trampolines */
/* #undef FFI_EXEC_STATIC_TRAMP */

/* Cannot use PROT_EXEC on this target, so, we revert to alternative means */
/* #define FFI_EXEC_TRAMPOLINE_TABLE 1 */

/* Define this if you want to enable pax emulated trampolines (experimental) */
/* #undef FFI_MMAP_EXEC_EMUTRAMP_PAX */

/* Cannot use malloc on this target, so, we revert to alternative means */
/* #undef FFI_MMAP_EXEC_WRIT */

/* Define this if you do not want support for the raw API. */
/* #undef FFI_NO_RAW_API */

/* Define this if you do not want support for aggregate types. */
/* #undef FFI_NO_STRUCTS */

/* Define to 1 if you have the <alloca.h> header file. */
/* #define HAVE_ALLOCA_H 1 */

/* Define if your assembler supports .cfi_* directives. */
/* #define HAVE_AS_CFI_PSEUDO_OP 1 */

/* Define if your assembler supports .register. */
/* #undef HAVE_AS_REGISTER_PSEUDO_OP */

/* Define if the compiler uses zarch features. */
/* #undef HAVE_AS_S390_ZARCH */

/* Define if your assembler and linker support unaligned PC relative relocs. */
/* #undef HAVE_AS_SPARC_UA_PCREL */

/* Define if your assembler supports unwind section type. */
/* #undef HAVE_AS_X86_64_UNWIND_SECTION_TYPE */

/* Define if your assembler supports PC relative relocs. */
/* #undef HAVE_AS_X86_PCREL */

/* Define to 1 if you have the <dlfcn.h> header file. */
/* #define HAVE_DLFCN_H 1 */

/* Define if __attribute__((visibility("hidden"))) is supported. */
/* #define HAVE_HIDDEN_VISIBILITY_ATTRIBUTE 1 */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define if you have the long double type and it is bigger than a double */
/* #undef HAVE_LONG_DOUBLE */

/* Define if you support more than one size of the long double type */
/* #undef HAVE_LONG_DOUBLE_VARIANT */

/* Define to 1 if you have the `memcpy' function. */
#define HAVE_MEMCPY 1

/* Define to 1 if you have the `memfd_create' function. */
/* #undef HAVE_MEMFD_CREATE */

/* Define if your compiler supports pointer authentication. */
/* #undef HAVE_PTRAUTH */

/* Define if .eh_frame sections should be read-only. */
/* #define HAVE_RO_EH_FRAME 1 */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/memfd.h> header file. */
/* #undef HAVE_SYS_MEMFD_H */

/* Define to 1 if you have the <sys/stat.h> header file. */
/* #define HAVE_SYS_STAT_H 1 */

/* Define to 1 if you have the <sys/types.h> header file. */
/* #define HAVE_SYS_TYPES_H 1 */

/* Define to 1 if you have the <unistd.h> header file. */
/* #define HAVE_UNISTD_H 1 */

/* Define to 1 if GNU symbol versioning is used for libatomic. */
/* #undef LIBFFI_GNU_SYMBOL_VERSIONING */

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "libffi"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "http://github.com/libffi/libffi/issues"

/* Define to the full name of this package. */
#define PACKAGE_NAME "libffi"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "libffi 3.4.4"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "libffi"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.4.4"

/* The size of `double', as computed by sizeof. */
#define SIZEOF_DOUBLE sizeof(double)

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T sizeof(size_t)

/* Define to 1 if all of the C90 standard headers exist (not just the ones
   required in a freestanding environment). This macro is provided for
   backward compatibility; new code need not use it. */
#define STDC_HEADERS 1

/* Define if symbols are underscored. */
#define SYMBOL_UNDERSCORE 1

/* Define this if you are using Purify and want to suppress spurious messages.
   */
/* #undef USING_PURIFY */

/* Version number of package */
#define VERSION "3.4.4"


#ifdef HAVE_HIDDEN_VISIBILITY_ATTRIBUTE
#ifdef LIBFFI_ASM
#ifdef __APPLE__
#define FFI_HIDDEN(name) .private_extern name
#else
#define FFI_HIDDEN(name) .hidden name
#endif
#else
#define FFI_HIDDEN __attribute__ ((visibility ("hidden")))
#endif
#else
#ifdef LIBFFI_ASM
#define FFI_HIDDEN(name)
#else
#define FFI_HIDDEN
#endif
#endif

