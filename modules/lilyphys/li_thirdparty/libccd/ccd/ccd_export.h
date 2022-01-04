#ifndef CCD_EXPORT_H
#define CCD_EXPORT_H

#ifdef CCD_STATIC_DEFINE
#  define CCD_EXPORT
#else
#  ifdef _MSC_VER
#    ifdef ccd_EXPORTS
#      define CCD_EXPORT __declspec(dllexport)
#   else /* ccd_EXPORTS */
#     define CCD_EXPORT  __declspec(dllimport)
#   endif /* ccd_EXPORTS */
#  else
#    ifndef CCD_EXPORT
#      ifdef ccd_EXPORTS
          /* We are building this library */
#        define CCD_EXPORT __attribute__((visibility("default")))
#      else
          /* We are using this library */
#        define CCD_EXPORT __attribute__((visibility("default")))
#      endif
#    endif
#  endif
#endif

#endif
