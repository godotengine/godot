#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/time.h>
#include <stdint.h>
#include <fcntl.h>
#include <getopt.h>
#include <errno.h>

#ifndef __WIN32__
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <poll.h>
#define closesocket close
#else
#include <winsock2.h>
#include <ws2tcpip.h>
typedef int socklen_t;
typedef uint32_t in_addr_t;
#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH
#ifdef EWOULDBLOCK
#undef EWOULDBLOCK
#endif
#define EWOULDBLOCK WSAEWOULDBLOCK
#define poll WSAPoll
#endif

#include <zlib.h>
#include <assert.h>

#define ZLIB_CHUNK (16 * 1024)

#define NETLOADER_SERVER_PORT 28280
#define NETLOADER_CLIENT_PORT 28771


static char cmdbuf[3072];
static uint32_t cmdlen=0;

//---------------------------------------------------------------------------------
static void shutdownSocket(int socket, int flags) {
//---------------------------------------------------------------------------------
	if (flags)
		shutdown(socket, flags);
	closesocket(socket);
}

//---------------------------------------------------------------------------------
static int setSocketNonblocking(int sock) {
//---------------------------------------------------------------------------------

#ifndef __WIN32__
	int flags = fcntl(sock, F_GETFL);

	if (flags == -1) return -1;

	int rc = fcntl(sock, F_SETFL, flags | O_NONBLOCK);

	if (rc != 0) return -1;
#else
	u_long iMode = 1; // non-blocking

	int rc = ioctlsocket(sock, FIONBIO, &iMode);

	if (rc != NO_ERROR) return -1;
#endif

	return 0;
}

//---------------------------------------------------------------------------------
static int socketError(const char *msg) {
//---------------------------------------------------------------------------------
#ifndef _WIN32
	int ret = errno;
	if (ret == EAGAIN)
		ret = EWOULDBLOCK;
	perror(msg);
#else
	int ret = WSAGetLastError();
	wchar_t *s = NULL;
	FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
               NULL, ret,
               MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
               (LPWSTR)&s, 0, NULL);
	fprintf(stderr, "%S\n", s);
	LocalFree(s);
	if (ret == WSAEWOULDBLOCK)
		ret = EWOULDBLOCK;
#endif

	return ret;
}

//---------------------------------------------------------------------------------
int pollSocket(int fd, int events, int timeout) {
//---------------------------------------------------------------------------------
#ifndef __WIN32__
	struct pollfd pfd;
#else
	WSAPOLLFD pfd;
#endif

	pfd.fd = fd;
	pfd.events = events;
	pfd.revents = 0;

	int ret = poll(&pfd, 1, timeout);
	if (ret < 0) {
		socketError("poll");
		return -1;
	}

	if (ret == 0)
		return -1;

	if (!(pfd.revents & events)) {
		int err = 0;
		int len = sizeof(err);
		getsockopt(fd, SOL_SOCKET, SO_ERROR, (char*)&err, &len);
		fprintf(stderr, "socket error 0x%x on poll\n", err);
		return -1;
	}

	return 0;
}

//---------------------------------------------------------------------------------
static struct in_addr findSwitch(int retries) {
//---------------------------------------------------------------------------------

	printf("pinging switch\n");

	struct sockaddr_in s, remote, rs;
	char recvbuf[256];
	char mess[] = "nxboot";

	int broadcastSock = socket(PF_INET, SOCK_DGRAM, 0);
	if (broadcastSock < 0) socketError("create send socket");

	int optval = 1, len;
	setsockopt(broadcastSock, SOL_SOCKET, SO_BROADCAST, (char *)&optval, sizeof(optval));

	memset(&s, '\0', sizeof(struct sockaddr_in));
	s.sin_family = AF_INET;
	s.sin_port = htons(NETLOADER_SERVER_PORT);
	s.sin_addr.s_addr = INADDR_BROADCAST;

	memset(&rs, '\0', sizeof(struct sockaddr_in));
	rs.sin_family = AF_INET;
	rs.sin_port = htons(NETLOADER_CLIENT_PORT);
	rs.sin_addr.s_addr = INADDR_ANY;

	int recvSock = socket(PF_INET, SOCK_DGRAM, 0);

	if (recvSock < 0)  socketError("create receive socket");

	if (bind(recvSock, (struct sockaddr*) &rs, sizeof(rs)) < 0) socketError("bind receive socket");
	setSocketNonblocking(recvSock);

	while (retries) {
		if (sendto(broadcastSock, mess, strlen(mess), 0, (struct sockaddr *)&s, sizeof(s)) < 0)
			socketError("sendto");

		if (pollSocket(recvSock, POLLIN, 150) == 0) {
			socklen_t socklen = sizeof(remote);
			len = recvfrom(recvSock, recvbuf, sizeof(recvbuf), 0, (struct sockaddr *)&remote, &socklen);
			if (len != -1) {
				if (strncmp("bootnx", recvbuf, strlen("bootnx")) == 0) {
					break;
				}
			}
		}

		--retries;
	}

	if (retries == 0)
		remote.sin_addr.s_addr =  INADDR_NONE;

	shutdownSocket(broadcastSock, 0);
	shutdownSocket(recvSock, SHUT_RD);

	return remote.sin_addr;
}

//---------------------------------------------------------------------------------
static int sendData(int sock, int sendsize, void *buffer) {
//---------------------------------------------------------------------------------
	char *buf = (char*)buffer;
	while (sendsize) {
		if (pollSocket(sock, POLLOUT, -1))
			return 1;

		int len = send(sock, buf, sendsize, 0);
		if (len == 0)
			return 1;

		if (len == -1) {
			if (socketError("send") != EWOULDBLOCK)
				return 1;
		} else {
			sendsize -= len;
			buf += len;
		}
	}

	return sendsize != 0;
}

//---------------------------------------------------------------------------------
static int recvData(int sock, void *buffer, int size, int flags) {
//---------------------------------------------------------------------------------
	int len, sizeleft = size;
	char *buf = (char*)buffer;
	while (sizeleft) {
		if (pollSocket(sock, POLLIN, -1))
			return 0;

		len = recv(sock,buf,sizeleft,flags);
		if (len == 0)
			return 0;

		if (len == -1) {
			if (socketError("recv") != EWOULDBLOCK)
				return 0;
		} else {
			sizeleft -=len;
			buf +=len;
		}
	}

	return size;
}


//---------------------------------------------------------------------------------
static int sendInt32LE(int socket, uint32_t size) {
//---------------------------------------------------------------------------------
	unsigned char lenbuf[4];
	lenbuf[0] = size & 0xff;
	lenbuf[1] = (size >>  8) & 0xff;
	lenbuf[2] = (size >> 16) & 0xff;
	lenbuf[3] = (size >> 24) & 0xff;

	return sendData(socket, 4, lenbuf);
}

//---------------------------------------------------------------------------------
static int recvInt32LE(int socket, int32_t *data) {
//---------------------------------------------------------------------------------
	unsigned char intbuf[4];
	int len = recvData(socket, intbuf, 4, 0);

	if (len == 4) {
		*data = intbuf[0] & 0xff + (intbuf[1] <<  8) + (intbuf[2] <<  16) + (intbuf[3] <<  24);
		return 0;
	}

	return -1;
}

static unsigned char in[ZLIB_CHUNK];
static unsigned char out[ZLIB_CHUNK];

//---------------------------------------------------------------------------------
static int sendNROFile(in_addr_t nxaddr, char *name, size_t filesize, FILE *fh) {
//---------------------------------------------------------------------------------

	int retval = 0;

	int ret, flush;
	unsigned have;
	z_stream strm;

	/* allocate deflate state */
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	ret = deflateInit(&strm, Z_DEFAULT_COMPRESSION);
	if (ret != Z_OK) return ret;

	int sock = socket(AF_INET,SOCK_STREAM,0);
	if (sock < 0) {
		socketError("create connection socket");
		return -1;
	}

	struct sockaddr_in s;
	memset(&s, '\0', sizeof(struct sockaddr_in));
	s.sin_family = AF_INET;
	s.sin_port = htons(NETLOADER_SERVER_PORT);
	s.sin_addr.s_addr = nxaddr;

	if (connect(sock,(struct sockaddr *)&s,sizeof(s)) < 0) {
		struct in_addr address;
		address.s_addr = nxaddr;
		fprintf(stderr,"Connection to %s failed\n",inet_ntoa(address));
		return -1;
	}

	int namelen = strlen(name);

	if (sendInt32LE(sock,namelen)) {
		fprintf(stderr,"Failed sending filename length\n");
		retval = -1;
		goto error;
	}

	if (sendData(sock,namelen,name)) {
		fprintf(stderr,"Failed sending filename\n");
		retval = -1;
		goto error;
	}

	if (sendInt32LE(sock,filesize)) {
		fprintf(stderr,"Failed sending file length\n");
		retval = -1;
		goto error;
	}

	int response;

	if (recvInt32LE(sock,&response)!=0) {
		fprintf(stderr,"Invalid response\n");
		retval = 1;
		goto error;
	}

	if (response!=0) {
		switch(response) {
			case -1:
				fprintf(stderr,"Failed to create file\n");
				break;
			case -2:
				fprintf(stderr,"Insufficient space\n");
				break;
			case -3:
				fprintf(stderr,"Insufficient memory\n");
				break;
		}
		retval = 1;
		goto error;
	}

	printf("Sending %s, %zd bytes\n",name, filesize);

	size_t totalsent = 0, blocks = 0;


	do {
		strm.avail_in = fread(in, 1, ZLIB_CHUNK, fh);
		if (ferror(fh)) {
			(void)deflateEnd(&strm);
			return Z_ERRNO;
		}
		flush = feof(fh) ? Z_FINISH : Z_NO_FLUSH;
		strm.next_in = in;
		/* run deflate() on input until output buffer not full, finish
		   compression if all of source has been read in */
		do {
			strm.avail_out = ZLIB_CHUNK;
			strm.next_out = out;
			ret = deflate(&strm, flush);    /* no bad return value */
			assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
			have = ZLIB_CHUNK - strm.avail_out;

			if (have != 0) {
				if (sendInt32LE(sock,have)) {
					fprintf(stderr,"Failed sending chunk size\n");
					retval = -1;
					goto error;
				}

				if (sendData(sock,have,out)) {
					fprintf(stderr,"Failed sending %s\n", name);
					retval = 1;
					(void)deflateEnd(&strm);
					goto error;
				}

				totalsent += have;
				blocks++;
			}
		} while (strm.avail_out == 0);
		assert(strm.avail_in == 0);     /* all input will be used */
		/* done when last data in file processed */
	} while (flush != Z_FINISH);
	assert(ret == Z_STREAM_END);        /* stream will be complete */
	(void)deflateEnd(&strm);

	printf("%zu sent (%.2f%%), %zd blocks\n",totalsent, (float)(totalsent * 100.0)/ filesize, blocks);

	if (recvInt32LE(sock,&response)!=0) {
		fprintf(stderr,"Failed sending %s\n",name);
		retval = 1;
		goto error;
	}


	if (sendData(sock,cmdlen+4,(unsigned char*)cmdbuf)) {

		fprintf(stderr,"Failed sending command line\n");
		retval = 1;

	}

error:
	shutdownSocket(sock, SHUT_WR);
	return retval;
}

//---------------------------------------------------------------------------------
static void showHelp() {
//---------------------------------------------------------------------------------
	puts("Usage: nxlink [options] nrofile\n");
	puts("--help,    -h   Display this information");
	puts("--address, -a   Hostname or IPv4 address of Switch");
	puts("--retries, -r   number of times to ping before giving up");
	puts("--path   , -p   set upload path for file");
	puts("--args          args to send to nro");
	puts("--server , -s   start server after completed upload");
	puts("\n");
}


//---------------------------------------------------------------------------------
static int addExtraArgs(int len, char *buf, char *extra_args) {
//---------------------------------------------------------------------------------

	if (NULL==extra_args) return len;


	int extra_len = strlen(extra_args);

	char *dst = &buf[len];
	char *src = extra_args;

	do {
		int c;

		do {
			c = *src++;
			extra_len--;
		} while (c ==' ' && extra_len >= 0);

		if (c == '\"' || c == '\'') {
			int quote = c;
			do {
				c = *src++;
				if (c != quote) *dst++ = c;
				extra_len--;
			} while (c != quote && extra_len >= 0);

			*dst++ = '\0';

			continue;
		}
		do {
			*dst++ = c;
			extra_len--;
			c = *src++;
		} while (c != ' ' && extra_len >= 0);

		*dst++ = '\0';
	} while (extra_len >= 0);

	return dst - buf;
}

#define NRO_ARGS	1000

#ifdef __WIN32__
static void win32_socket_cleanup(void) {
	WSACleanup();
}
#endif

//---------------------------------------------------------------------------------
// -- GODOT start --
int nxlink_main(int argc, char **argv) {
// -- GODOT end --
//---------------------------------------------------------------------------------
	char *address = NULL;
	char *basepath = NULL;
	char *finalpath = NULL;
	char *endarg = NULL;
	char *extra_args = NULL;
	int retries = 10;
	static int server = 0;

	if (argc < 2) {
		showHelp();
		return EXIT_FAILURE;
	}

	while (1) {
		static struct option long_options[] = {
			{"address", required_argument, 0,	'a'},
			{"retries", required_argument, 0,	'r'},
			{"path",    required_argument, 0,	'p'},
			{"args",    required_argument, 0,  NRO_ARGS},
			{"help",    no_argument,       0,	'h'},
			{"server",  no_argument,       &server,  1 },
			{0, 0, 0, 0}
		};

		/* getopt_long stores the option index here. */
		int option_index = 0, c;

		c = getopt_long (argc, argv, "a:r:hp:s", long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch(c) {

		case 'a':
			address = optarg;
			break;
		case 'r':
			errno = 0;
			retries = strtoul(optarg, &endarg, 0);
			if (endarg == optarg) errno = EINVAL;
			if (errno != 0) {
				perror("--retries");
				exit(1);
			}
			break;
		case 'p':
			basepath = optarg;
			break;
		case 's':
			server = 1;
			break;
		case 'h':
			showHelp();
			return EXIT_FAILURE;
		case NRO_ARGS:
			extra_args=optarg;
			break;
		}

	}

	char *filename = argv[optind++];
	if (filename== NULL) {
		showHelp();
		return EXIT_FAILURE;
	}

	memset(cmdbuf, '\0', sizeof(cmdbuf));

	FILE *fh = fopen(filename,"rb");
	if (fh == NULL) {
		fprintf(stderr,"Failed to open %s\n",filename);
		return EXIT_FAILURE;
	}

#ifdef _WIN32
	setvbuf(stdout, 0, _IONBF, 0);
#endif

	fseek(fh,0,SEEK_END);
	size_t filesize = ftell(fh);
	fseek(fh,0,SEEK_SET);

	char *basename = NULL;
	if ((basename=strrchr(filename,'/'))!=NULL) {
		basename++;
	} else if ((basename=strrchr(filename,'\\'))!=NULL) {
		basename++;
	} else {
		basename = filename;
	}

	if (basepath) {
		size_t finalpath_len = strlen(basepath);
		if (basepath[finalpath_len] == '/') {
			finalpath_len += (strlen(basename) + 1);
			finalpath = malloc(finalpath_len);
			sprintf(finalpath, "%s%s", basepath, basename);
		} else {
			finalpath = basepath;
		}
	} else {
		finalpath = basename;
	}

	cmdlen = 0;

	for (int index = optind; index < argc; index++) {
		int len=strlen(argv[index]);
		if ((cmdlen + len + 5) >= (sizeof(cmdbuf) - 2)) break;
		strcpy(&cmdbuf[cmdlen+4],argv[index]);
		cmdlen+= len + 1;
	}

	cmdlen = addExtraArgs(cmdlen, &cmdbuf[4], extra_args);

	cmdbuf[0] = cmdlen & 0xff;
	cmdbuf[1] = (cmdlen>>8) & 0xff;
	cmdbuf[2] = (cmdlen>>16) & 0xff;
	cmdbuf[3] = (cmdlen>>24) & 0xff;

#ifdef __WIN32__
	WSADATA wsa_data;
	if (WSAStartup (MAKEWORD(2,2), &wsa_data)) {
		printf ("WSAStartup failed\n");
		return EXIT_FAILURE;
	}
	atexit(&win32_socket_cleanup);
#endif

	struct in_addr nxaddr;
	nxaddr.s_addr  =  INADDR_NONE;

	if (address == NULL) {
		nxaddr = findSwitch(retries);

		if (nxaddr.s_addr == INADDR_NONE) {
			printf("No response from Switch!\n");
			return EXIT_FAILURE;
		}

	} else {
		struct addrinfo *info;
		if (getaddrinfo(address, NULL, NULL, &info) == 0) {
			nxaddr = ((struct sockaddr_in*)info->ai_addr)->sin_addr;
			freeaddrinfo(info);
		}
	}

	if (nxaddr.s_addr == INADDR_NONE) {
		fprintf(stderr,"Invalid address\n");
		return EXIT_FAILURE;
	}

	int res = sendNROFile(nxaddr.s_addr,finalpath,filesize,fh);

	fclose(fh);

	if (res != 0)
		return EXIT_FAILURE;

	if (!server)
		return EXIT_SUCCESS;

	printf("starting server\n");

	struct sockaddr_in serv_addr;

	memset(&serv_addr, '0', sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(NETLOADER_CLIENT_PORT);

	int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	if (listenfd < 0) {
		socketError("socket");
		return EXIT_FAILURE;
	}

	int rc = bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
	if (rc != 0) {
		socketError("bind listen socket");
		shutdownSocket(listenfd, 0);
		return EXIT_FAILURE;
	}

	rc = setSocketNonblocking(listenfd);
	if (rc == -1) {
		socketError("listen fcntl");
		shutdownSocket(listenfd, 0);
		return EXIT_FAILURE;
	}

	rc = listen(listenfd, 10);
	if (rc != 0) {
		socketError("listen");
		shutdownSocket(listenfd, 0);
		return EXIT_FAILURE;
	}

	printf("server active ...\n");

	int datafd = -1;

	while (listenfd != -1 || datafd != -1) {
		struct sockaddr_in sa_remote;

		if (pollSocket(listenfd >= 0 ? listenfd : datafd, POLLIN, -1))
			break;

		if (listenfd >= 0) {
			socklen_t addrlen = sizeof(sa_remote);
			datafd = accept(listenfd, (struct sockaddr*)&sa_remote, &addrlen);

			if (datafd < 0 && socketError("accept") != EWOULDBLOCK)
				break;

			if (datafd >= 0) {
				shutdownSocket(listenfd, 0);
				listenfd = -1;
			}
		} else {
			char recvbuf[256];
			int len = recv(datafd, recvbuf, sizeof(recvbuf), 0);

			if (len == 0 || (len < 0 && socketError("recv") != EWOULDBLOCK)) {
				shutdownSocket(datafd, 0);
				datafd = -1;
				break;
			}

			if (len > 0)
				fwrite(recvbuf, 1, len, stdout);
		}
	}

	if (listenfd >= 0)
		shutdownSocket(listenfd, 0);
	if (datafd >= 0)
		shutdownSocket(datafd, SHUT_RD);

	printf("exiting ... \n");

	return EXIT_SUCCESS;
}
