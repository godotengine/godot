#ifndef SPEEX_BIND_H
#define SPEEX_BIND_H

#ifdef __cplusplus
extern "C" {
#endif

/*
#define OVERRIDE_SPEEX_ALLOC
#define OVERRIDE_SPEEX_ALLOC_SCRATCH
#define OVERRIDE_SPEEX_REALLOC
#define OVERRIDE_SPEEX_FREE
#define OVERRIDE_SPEEX_FREE_SCRATCH
#define OVERRIDE_SPEEX_FATAL
#define OVERRIDE_SPEEX_WARNING
#define OVERRIDE_SPEEX_WARNING_INT
#define OVERRIDE_SPEEX_NOTIFY
#define OVERRIDE_SPEEX_PUTC

void *speex_alloc (int size);
void *speex_alloc_scratch (int size);
void *speex_realloc (void *ptr, int size);
void speex_free (void *ptr);
void speex_free_scratch (void *ptr);
void _speex_fatal(const char *str, const char *file, int line);
void speex_warning(const char *str);
void speex_warning_int(const char *str, int val);
void speex_notify(const char *str);
void _speex_putc(int ch, void *file);


*/
#define RELEASE
#define SPEEX_PI 3.14159265358979323846

#ifdef _MSC_VER
#define SPEEX_INLINE __inline
#else
#define SPEEX_INLINE inline
#endif



#ifdef __cplusplus
}
#endif

#endif // SPEEX_BIND_H
