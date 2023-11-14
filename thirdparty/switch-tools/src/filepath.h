#pragma once

#include "types.h"
#include <dirent.h>
#ifdef _WIN32
#include <direct.h>
#include <wchar.h>
#endif

#define MAX_SWITCHPATH 0x300

typedef enum {
    VALIDITY_UNCHECKED = 0,
    VALIDITY_INVALID,
    VALIDITY_VALID
} validity_t;

#ifdef _MSC_VER
inline int fseeko64(FILE *__stream, long long __off, int __whence)
{
    return _fseeki64(__stream, __off, __whence);
}
#else
    /* off_t is 64-bit with large file support */
    #define fseeko64 fseek
#endif

#ifdef _WIN32
typedef wchar_t oschar_t; /* utf-16 */
typedef _WDIR osdir_t;
typedef struct _wdirent osdirent_t;
typedef struct _stati64 os_stat64_t;

#define os_fopen _wfopen
#define os_opendir _wopendir
#define os_closedir _wclosedir
#define os_readdir _wreaddir
#define os_stat _wstati64
#define os_fclose fclose

#define OS_MODE_READ L"rb"
#define OS_MODE_WRITE L"wb"
#define OS_MODE_EDIT L"rb+"
#else
typedef char oschar_t; /* utf-8 */
typedef DIR osdir_t;
typedef struct dirent osdirent_t;
typedef struct stat os_stat64_t;

#define os_fopen fopen
#define os_opendir opendir
#define os_closedir closedir
#define os_readdir readdir
#define os_stat stat
#define os_fclose fclose

#define OS_MODE_READ "rb"
#define OS_MODE_WRITE "wb"
#define OS_MODE_EDIT "rb+"
#endif

#define OS_PATH_SEPARATOR "/"

typedef struct filepath {
    char char_path[MAX_SWITCHPATH];
    oschar_t os_path[MAX_SWITCHPATH];
    validity_t valid;
} filepath_t;

void os_strncpy(oschar_t *dst, const char *src, size_t size);
void os_strncpy_to_char(char *dst, const oschar_t *src, size_t size);
int os_makedir(const oschar_t *dir);
int os_rmdir(const oschar_t *dir);

void filepath_init(filepath_t *fpath);
void filepath_copy(filepath_t *fpath, filepath_t *copy);
void filepath_os_append(filepath_t *fpath, oschar_t *path);
void filepath_append(filepath_t *fpath, const char *format, ...);
void filepath_append_n(filepath_t *fpath, uint32_t n, const char *format, ...);
void filepath_set(filepath_t *fpath, const char *path);
oschar_t *filepath_get(filepath_t *fpath);
