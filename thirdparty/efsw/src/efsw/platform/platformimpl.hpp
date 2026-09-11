#ifndef EFSW_PLATFORMIMPL_HPP
#define EFSW_PLATFORMIMPL_HPP

#include <efsw/base.hpp>

#if defined( EFSW_PLATFORM_POSIX )
#include <efsw/platform/posix/SystemImpl.hpp>
#include <efsw/platform/posix/FileSystemImpl.hpp>
#elif EFSW_PLATFORM == EFSW_PLATFORM_WIN32
#include <efsw/platform/win/SystemImpl.hpp>
#include <efsw/platform/win/FileSystemImpl.hpp>
#else
#error Thread, Mutex, and System not implemented for this platform.
#endif

#endif
