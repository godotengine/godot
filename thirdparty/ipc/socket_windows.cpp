// We need different code for Windows sockets compared to other platforms.
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include "socket_implementation.h"

#include <afunix.h>
#include <stdlib.h>
#include <stdio.h>

// danger will robinson danger
InitializerMagic magic;

void SocketImplementation::initialize()
{
    // prevent printf failing this is windows
    setvbuf (stdout, NULL, _IONBF, 0);

    // Initialise winsock
    WSADATA wsaData;
    int err = WSAStartup(MAKEWORD(2,2), &wsaData);
    if(err != 0)
    {
        printf("WSAStartup Failed: %d", err);
    }

}

void SocketImplementation::finalize()
{
    WSACleanup();
}

/* Limit to 108 chars */
int SocketImplementation::create_af_unix_socket(
            struct sockaddr_un& name,
            const char * file_path )
{
    const int socket_id = socket(AF_UNIX, SOCK_STREAM, 0);
    if(socket_id == -1) {
        perror("client socket");
        return -1;
    }
    /* Ensure portable by resetting all to zero */
    memset(&name, 0, sizeof(name));

    /* AF_UNIX */
    name.sun_family = AF_UNIX;
    strncpy(name.sun_path, file_path, sizeof(name.sun_path) - 1);

    return socket_id;
}

/*
 * Sets the socket into non-blocking mode for read/write operations
 * listen() is non-blocking on Linux but not on Mac, on Mac you must poll() first.
 * Windows has a different handler for setting the socket to non-blocking
 * Unix, Linux and Mac uses fcntl.
*/
int SocketImplementation::set_non_blocking( int socket_handle )
{
//    return 0;
    unsigned long enable_non_blocking = 1;
    return ioctlsocket(socket_handle, FIONBIO, &enable_non_blocking ) == 0;
}
int SocketImplementation::connect( int socket_handle, const struct sockaddr *address, socklen_t address_len )
{
    return ::connect(socket_handle, address, address_len);
}
int SocketImplementation::send( int socket_handle, const char * msg, size_t len)
{
    struct pollfd pfd;
    pfd.fd = socket_handle;
    pfd.events = POLLWRNORM;

    if(SocketImplementation::poll(&pfd) == -1 )
    {
        SocketImplementation::perror("poll waiting error");
        return -1;
    }
    else if( pfd.revents & POLLWRNORM )
    {
        printf("waiting for write [%d] %s \n", __LINE__, __FILE__);

        int OK = ::send(socket_handle, msg, len, 0);
        if(OK == -1)
        {
            SocketImplementation::perror("cant send message");
            SocketImplementation::close(socket_handle);
            return -1;
        }
        else
        {
            return 0;
        }

    }

    return 0;
}
int SocketImplementation::recv( int socket_handle, char * buffer, size_t bufferSize )
{
    struct pollfd pfd;
    pfd.fd = socket_handle;
    pfd.events = POLLRDNORM;

    if(SocketImplementation::poll(&pfd) == -1 )
    {
        SocketImplementation::perror("poll read error");
        return -1;
    }
    else if( pfd.revents & POLLRDNORM )
    {
        printf("waiting for recv [%d] %s \n", __LINE__, __FILE__);

        int OK = ::recv(socket_handle, buffer, bufferSize, 0);
        if(OK == -1)
        {
            SocketImplementation::perror("cant read message");
            SocketImplementation::close(socket_handle);
            return -1;
        }
        else
        {
            return 0;
        }
    }

    return 0;
}

int SocketImplementation::poll( struct pollfd * pfd, int timeout, int n )
{
    return ::WSAPoll(pfd, n, timeout);
}

int SocketImplementation::poll( int socket_handle )
{
    struct pollfd pfd;
    pfd.fd = socket_handle;
    pfd.events = POLLIN | POLLOUT;
    pfd.revents = 0;
    return ::WSAPoll(&pfd, 1, 0);
}
int SocketImplementation::accept( int socket_handle, struct sockaddr *addr, socklen_t * addrlen)
{
    return ::accept(socket_handle, addr, addrlen);
}
int SocketImplementation::bind( int socket_handle, const struct sockaddr *addr, size_t len)
{
    return ::bind(socket_handle, addr, len);
}
int SocketImplementation::listen( int socket_handle, int connection_pool_size)
{
    return ::listen(socket_handle, connection_pool_size);
}
void SocketImplementation::close( int socket_handle )
{
    ::closesocket(socket_handle);
}
void SocketImplementation::unlink( const char * unlink_file )
{
    _unlink(unlink_file);
}

void SocketImplementation::perror( const char * msg)
{
    LPVOID lpMsgBuf;
    int e;

    lpMsgBuf = (LPVOID)"Unknown error";
    e = WSAGetLastError();
    if (FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, e,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            // Default language
            (LPTSTR)&lpMsgBuf, 0, NULL)) {
        fprintf(stderr, "%s: Error %d: %s\n", msg, e, (char *)lpMsgBuf);
        LocalFree(lpMsgBuf);
    } else
        fprintf(stderr, "%s: Error %d\n", msg, e);
}

#endif