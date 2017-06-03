/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE libopusfile SOFTWARE CODEC SOURCE CODE. *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE libopusfile SOURCE CODE IS (C) COPYRIGHT 2012                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************/
#if !defined(_opusfile_winerrno_h)
# define _opusfile_winerrno_h (1)

# include <errno.h>
# include <winerror.h>

/*These conflict with the MSVC errno.h definitions, but we don't need to use
   the original ones in any file that deals with sockets.
  We could map the WSA errors to the errno.h ones (most of which are only
   available on sufficiently new versions of MSVC), but they aren't ordered the
   same, and given how rarely we actually look at the values, I don't think
   it's worth a lookup table.*/
# undef EWOULDBLOCK
# undef EINPROGRESS
# undef EALREADY
# undef ENOTSOCK
# undef EDESTADDRREQ
# undef EMSGSIZE
# undef EPROTOTYPE
# undef ENOPROTOOPT
# undef EPROTONOSUPPORT
# undef EOPNOTSUPP
# undef EAFNOSUPPORT
# undef EADDRINUSE
# undef EADDRNOTAVAIL
# undef ENETDOWN
# undef ENETUNREACH
# undef ENETRESET
# undef ECONNABORTED
# undef ECONNRESET
# undef ENOBUFS
# undef EISCONN
# undef ENOTCONN
# undef ETIMEDOUT
# undef ECONNREFUSED
# undef ELOOP
# undef ENAMETOOLONG
# undef EHOSTUNREACH
# undef ENOTEMPTY

# define EWOULDBLOCK     (WSAEWOULDBLOCK-WSABASEERR)
# define EINPROGRESS     (WSAEINPROGRESS-WSABASEERR)
# define EALREADY        (WSAEALREADY-WSABASEERR)
# define ENOTSOCK        (WSAENOTSOCK-WSABASEERR)
# define EDESTADDRREQ    (WSAEDESTADDRREQ-WSABASEERR)
# define EMSGSIZE        (WSAEMSGSIZE-WSABASEERR)
# define EPROTOTYPE      (WSAEPROTOTYPE-WSABASEERR)
# define ENOPROTOOPT     (WSAENOPROTOOPT-WSABASEERR)
# define EPROTONOSUPPORT (WSAEPROTONOSUPPORT-WSABASEERR)
# define ESOCKTNOSUPPORT (WSAESOCKTNOSUPPORT-WSABASEERR)
# define EOPNOTSUPP      (WSAEOPNOTSUPP-WSABASEERR)
# define EPFNOSUPPORT    (WSAEPFNOSUPPORT-WSABASEERR)
# define EAFNOSUPPORT    (WSAEAFNOSUPPORT-WSABASEERR)
# define EADDRINUSE      (WSAEADDRINUSE-WSABASEERR)
# define EADDRNOTAVAIL   (WSAEADDRNOTAVAIL-WSABASEERR)
# define ENETDOWN        (WSAENETDOWN-WSABASEERR)
# define ENETUNREACH     (WSAENETUNREACH-WSABASEERR)
# define ENETRESET       (WSAENETRESET-WSABASEERR)
# define ECONNABORTED    (WSAECONNABORTED-WSABASEERR)
# define ECONNRESET      (WSAECONNRESET-WSABASEERR)
# define ENOBUFS         (WSAENOBUFS-WSABASEERR)
# define EISCONN         (WSAEISCONN-WSABASEERR)
# define ENOTCONN        (WSAENOTCONN-WSABASEERR)
# define ESHUTDOWN       (WSAESHUTDOWN-WSABASEERR)
# define ETOOMANYREFS    (WSAETOOMANYREFS-WSABASEERR)
# define ETIMEDOUT       (WSAETIMEDOUT-WSABASEERR)
# define ECONNREFUSED    (WSAECONNREFUSED-WSABASEERR)
# define ELOOP           (WSAELOOP-WSABASEERR)
# define ENAMETOOLONG    (WSAENAMETOOLONG-WSABASEERR)
# define EHOSTDOWN       (WSAEHOSTDOWN-WSABASEERR)
# define EHOSTUNREACH    (WSAEHOSTUNREACH-WSABASEERR)
# define ENOTEMPTY       (WSAENOTEMPTY-WSABASEERR)
# define EPROCLIM        (WSAEPROCLIM-WSABASEERR)
# define EUSERS          (WSAEUSERS-WSABASEERR)
# define EDQUOT          (WSAEDQUOT-WSABASEERR)
# define ESTALE          (WSAESTALE-WSABASEERR)
# define EREMOTE         (WSAEREMOTE-WSABASEERR)

#endif
