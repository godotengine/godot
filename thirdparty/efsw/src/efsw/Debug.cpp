#include <efsw/Debug.hpp>
#include <iostream>

#ifdef EFSW_COMPILER_MSVC
#define WIN32_LEAN_AND_MEAN
#include <crtdbg.h>
#include <windows.h>
#endif

#include <cassert>
#include <cstdarg>
#include <cstdio>

namespace efsw {

#ifdef DEBUG

void efREPORT_ASSERT( const char* File, int Line, const char* Exp ) {
#ifdef EFSW_COMPILER_MSVC
	_CrtDbgReport( _CRT_ASSERT, File, Line, "", Exp );

	DebugBreak();
#else
	std::cout << "ASSERT: " << Exp << " file: " << File << " line: " << Line << std::endl;

#if defined( EFSW_COMPILER_GCC ) && defined( EFSW_32BIT ) && !defined( EFSW_ARM )
	asm( "int3" );
#else
	assert( false );
#endif
#endif
}

void efPRINT( const char* format, ... ) {
	char buf[2048];
	va_list args;

	va_start( args, format );

#ifdef EFSW_COMPILER_MSVC
	_vsnprintf_s( buf, sizeof( buf ), sizeof( buf ) / sizeof( buf[0] ), format, args );
#else
	vsnprintf( buf, sizeof( buf ) / sizeof( buf[0] ), format, args );
#endif

	va_end( args );

#ifdef EFSW_COMPILER_MSVC
	OutputDebugStringA( buf );
#else
	std::cout << buf;
#endif
}

void efPRINTC( unsigned int cond, const char* format, ... ) {
	if ( 0 == cond )
		return;

	char buf[2048];
	va_list args;

	va_start( args, format );

#ifdef EFSW_COMPILER_MSVC
	_vsnprintf_s( buf, efARRAY_SIZE( buf ), efARRAY_SIZE( buf ), format, args );
#else
	vsnprintf( buf, sizeof( buf ) / sizeof( buf[0] ), format, args );
#endif

	va_end( args );

#ifdef EFSW_COMPILER_MSVC
	OutputDebugStringA( buf );
#else
	std::cout << buf;
#endif
}

#endif

} // namespace efsw
