
#ifndef JXL_CMS_EXPORT_H
#define JXL_CMS_EXPORT_H

#ifdef JXL_CMS_STATIC_DEFINE
#  define JXL_CMS_EXPORT
#  define JXL_CMS_NO_EXPORT
#else
#  ifndef JXL_CMS_EXPORT
#    ifdef JXL_INTERNAL_LIBRARY_BUILD
        /* We are building this library */
#      define JXL_CMS_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define JXL_CMS_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef JXL_CMS_NO_EXPORT
#    define JXL_CMS_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef JXL_CMS_DEPRECATED
#  define JXL_CMS_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef JXL_CMS_DEPRECATED_EXPORT
#  define JXL_CMS_DEPRECATED_EXPORT JXL_CMS_EXPORT JXL_CMS_DEPRECATED
#endif

#ifndef JXL_CMS_DEPRECATED_NO_EXPORT
#  define JXL_CMS_DEPRECATED_NO_EXPORT JXL_CMS_NO_EXPORT JXL_CMS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef JXL_CMS_NO_DEPRECATED
#    define JXL_CMS_NO_DEPRECATED
#  endif
#endif

#endif /* JXL_CMS_EXPORT_H */
