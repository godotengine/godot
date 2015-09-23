#pragma once

#define U16TEXT(t) reinterpret_cast<const unicode16_t*>( t )

#define StringFileInfo U16TEXT("S\0t\0r\0i\0n\0g\0F\0i\0l\0e\0I\0n\0f\0o\0\0")
#define SizeofStringFileInfo sizeof("S\0t\0r\0i\0n\0g\0F\0i\0l\0e\0I\0n\0f\0o\0\0")
#define VarFileInfo U16TEXT("V\0a\0r\0F\0i\0l\0e\0I\0n\0f\0o\0\0")
#define Translation U16TEXT("T\0r\0a\0n\0s\0l\0a\0t\0i\0o\0n\0\0")

#define VarFileInfoAligned U16TEXT("V\0a\0r\0F\0i\0l\0e\0I\0n\0f\0o\0\0\0\0")
#define TranslationAligned U16TEXT("T\0r\0a\0n\0s\0l\0a\0t\0i\0o\0n\0\0\0\0")
#define SizeofVarFileInfoAligned sizeof("V\0a\0r\0F\0i\0l\0e\0I\0n\0f\0o\0\0\0\0")
#define SizeofTranslationAligned sizeof("T\0r\0a\0n\0s\0l\0a\0t\0i\0o\0n\0\0\0\0")
