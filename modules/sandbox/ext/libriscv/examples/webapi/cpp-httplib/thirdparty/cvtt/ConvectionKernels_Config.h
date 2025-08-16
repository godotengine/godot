#pragma once
#ifndef __CVTT_CONFIG_H__
#define __CVTT_CONFIG_H__

#if (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_X64) || defined(__SSE2__)
#define CVTT_USE_SSE2
#endif

// Define this to compile everything as a single source file
//#define CVTT_SINGLE_FILE

#endif
