#include "mmap.hpp"

#ifdef _WIN32
#  include <io.h>
#  include <windows.h>

void* mmap( void* addr, size_t length, int prot, int flags, int fd, off_t offset )
{
    HANDLE hnd;
    void* map = nullptr;

    switch( prot )
    {
    case PROT_READ:
        if( hnd = CreateFileMapping( HANDLE( _get_osfhandle( fd ) ), nullptr, PAGE_READONLY, 0, DWORD( length ), nullptr ) )
        {
            map = MapViewOfFile( hnd, FILE_MAP_READ, 0, 0, length );
            CloseHandle( hnd );
        }
        break;
    case PROT_WRITE:
        if( hnd = CreateFileMapping( HANDLE( _get_osfhandle( fd ) ), nullptr, PAGE_READWRITE, 0, DWORD( length ), nullptr ) )
        {
            map = MapViewOfFile( hnd, FILE_MAP_WRITE, 0, 0, length );
            CloseHandle( hnd );
        }
        break;
    }

    return map ? (char*)map + offset : (void*)-1;
}

int munmap( void* addr, size_t length )
{
    return UnmapViewOfFile( addr ) != 0 ? 0 : -1;
}

#endif
