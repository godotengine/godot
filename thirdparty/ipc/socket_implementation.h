#ifndef GENERIC_IPC_HANDLER_SOCKET_IMPLEMENTATION_H
#define GENERIC_IPC_HANDLER_SOCKET_IMPLEMENTATION_H

#include <stddef.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#endif

/* This is because when WERROR and all errors are enabled you get the warning
 * error: duplicated 'if' condition and is better than disabling the error */
#if EAGAIN == EWOULDBLOCK
#define IPC_AGAIN(x) \
x == EAGAIN
#else
#define IPC_AGAIN(x) \
x == EAGAIN || x == EWOULDBLOCK
#endif


struct SocketImplementation
{
    static void initialize();
    static void finalize();

    /* Limit to 108 chars */
    static int create_af_unix_socket(
            struct sockaddr_un& name,
            const char * file_path );
    /*
     * Sets the socket into non-blocking mode for read/write operations
     * listen() is non-blocking on Linux but not on Mac, on Mac you must poll() first.
     * Windows has a different handler for setting the socket to non-blocking
     * Unix, Linux and Mac uses fcntl.
    */
    static int set_non_blocking( int socket_handle );
    static int connect( int socket_handle, const struct sockaddr *address, socklen_t address_len );
    static int send( int socket_handle, const char * msg, size_t len);
    static int recv( int socket_handle, char * buffer, size_t bufferSize );
    static int poll( int socket_handle );
    static int poll( struct pollfd * pfd, int timeout = 1, int n = 1 );
    static int accept( int socket_handle, struct sockaddr *addr, socklen_t * addrlen);
    static int bind( int socket_handle, const struct sockaddr *addr, size_t len);
    static int listen( int socket_handle, int connection_pool_size);
    static void close( int socket_handle );
    static void unlink( const char * unlink_file );
    static void perror( const char * msg );
};

/*
 * Really because we shouldn't need to tell the OS to start and finalise but we do because Microsoft's
 * API requires WSAStartup and Shutdown
 */
struct InitializerMagic
{
    InitializerMagic()
    {
        SocketImplementation::initialize();
    }

    ~InitializerMagic()
    {
        SocketImplementation::finalize();
    }
};

#endif //GENERIC_IPC_HANDLER_SOCKET_IMPLEMENTATION_H
