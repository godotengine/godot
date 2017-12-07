#ifndef LIBUSB_POLL_POSIX_H
#define LIBUSB_POLL_POSIX_H

#define usbi_write write
#define usbi_read read
#define usbi_close close
#define usbi_poll poll

int usbi_pipe(int pipefd[2]);

#endif /* LIBUSB_POLL_POSIX_H */
