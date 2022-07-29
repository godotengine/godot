#include "socket_implementation.h"
#ifndef _WIN32

/* Creating AF Unix socket the entire point */
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int SocketImplementation::set_non_blocking(int socket_handle) {
    int flags = fcntl(socket_handle, F_GETFD );
    return fcntl(socket_handle, F_SETFD, flags | O_NONBLOCK);
}

/* A poll method where you don't need the manual
 * Since we are using this specific to AF_UNIX I kept this simple
 */
int SocketImplementation::poll( int socket_handle )
{
    struct pollfd pfd;
    pfd.fd = socket_handle;
    pfd.events = POLLIN | POLLOUT;
    pfd.revents = 0;
    return ::poll(&pfd, 1, 0);
}

int SocketImplementation::poll( struct pollfd * pfd, int timeout, int n )
{
	return ::poll(pfd, n, timeout);
}

/* Why so many proxy functions?
 * Well cotton, the reason is WinSock API has a bunch of differences, so linux is basically just a passthrough of the
 * unix functionality.
 * If M$ used the Posix / IEEE socket implementation and used this API it would be much cleaner, but they don't.
 * The other option is using a bunch of ifdefs inside logic which may need to change per OS
 * In the future when another tech comes along you can use this API to swap in another implementation, so its not
 * a total loss.
 */

int SocketImplementation::connect( int socket_handle, const struct sockaddr *address, socklen_t address_len )
{
    return ::connect(socket_handle, address, address_len);
}

int SocketImplementation::send( int socket_handle, const char * msg, size_t len )
{
    return ::send(socket_handle, msg, len, MSG_DONTWAIT);
}

int SocketImplementation::recv( int socket_handle, char * buffer, size_t bufferSize )
{
    return ::recv(socket_handle, buffer, bufferSize, 0);
}

int SocketImplementation::accept(
        int socket_handle,
        struct sockaddr *addr,
        socklen_t * addrlen )
{
    return ::accept(socket_handle, addr, addrlen);
}

int SocketImplementation::bind( int socket_handle, const struct sockaddr *addr, size_t len)
{
    return ::bind(socket_handle, addr, len);
}

int SocketImplementation::listen( int socket_handle, int connection_pool_size )
{
    return ::listen(socket_handle, connection_pool_size);
}

void SocketImplementation::close(int socket_handle) {
    ::close(socket_handle);
}

void SocketImplementation::unlink(const char *unlink_file) {
    ::unlink(unlink_file);
}

void SocketImplementation::perror(const char *err) {
	::perror(err);
}

#endif