/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Logging and Error Handling */

#ifndef LOG_H
#define LOG_H

void* ohmd_allocfn(ohmd_context* ctx, const char* e_msg, size_t size);
#define ohmd_alloc(_ctx, _size) ohmd_allocfn(_ctx, "could not allocate " #_size " bytes of RAM @ " __FILE__ ":" OHMD_STRINGIFY(__LINE__), _size)

#ifndef LOGLEVEL
#define LOGLEVEL 2
#endif

#define LOG(_level, _levelstr, ...) do{ if(_level >= LOGLEVEL){ printf("[%s] ", (_levelstr)); printf(__VA_ARGS__); puts(""); } } while(0)

#if LOGLEVEL == 0
#define LOGD(...) LOG(0, "DD", __VA_ARGS__)
#else
#define LOGD(...)
#endif

#define LOGV(...) LOG(1, "VV", __VA_ARGS__)
#define LOGI(...) LOG(2, "II", __VA_ARGS__)
#define LOGW(...) LOG(3, "WW", __VA_ARGS__)
#define LOGE(...) LOG(4, "EE", __VA_ARGS__)

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#define ohmd_set_error(_ctx, ...) { snprintf((_ctx)->error_msg, OHMD_STR_SIZE, __VA_ARGS__); LOGE(__VA_ARGS__); }

#endif
