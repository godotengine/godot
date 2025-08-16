#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

void error(const char *msg) {
	perror(msg);
	exit(1);
}

int main(int argc, char** argv)
{
	uint16_t server_port = 8081;
	if (argc > 1) {
		server_port = atoi(argv[1]);
	}

	/*
	* socket: create TCP stream fd
	*/
	const int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	if (listenfd < 0)
		error("Error creating TCP socket");

	const int optval = 1;
	setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR,
		&optval, sizeof(int));

	struct sockaddr_in serveraddr = {};
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serveraddr.sin_port = htons(server_port);

	/*
	* bind: associate the parent socket with a port
	*/
	if (bind(listenfd, (struct sockaddr *) &serveraddr,
		sizeof(serveraddr)) < 0)
		error("Bind error. Port already in use?");

	/*
	* listen: listen for incoming TCP connection requests
	*/
	if (listen(listenfd, 5) < 0)
		error("ERROR on listen");

	printf("Listening on port %u\n", server_port);

	/*
	* main loop: wait for a connection request, echo input line,
	* then close connection.
	*/
	while (1) {
		/*
		 * accept: wait for a connection request
		 */
		struct sockaddr_in clientaddr;
		socklen_t clientlen = sizeof(clientaddr);
		const int clientfd =
			accept(listenfd, (struct sockaddr *)&clientaddr, &clientlen);
		if (clientfd < 0) {
			printf("ERROR %s on server accept\n", strerror(errno));
			continue;
		}

		/*
		 * inet_ntoa: print who sent the message
		 */
		char* ipstr = inet_ntoa(clientaddr.sin_addr);
		if (ipstr == NULL)
			error("ERROR on inet_ntoa\n");
		printf("Server established connection with %s\n",
			ipstr);

		/*
		 * read: read input string from the client
		 */
		char buffer[8192];
		const ssize_t rb = read(clientfd, buffer, sizeof(buffer));
		if (rb < 0) {
			printf("ERROR %s reading from socket\n", strerror(errno));
			close(clientfd);
		} else if (rb == 0) {
			close(clientfd);
			continue;
		}
		printf("Server received %u bytes: %.*s",
			(unsigned)rb, (int)rb, buffer);

		/*
		 * write: echo the input string back to the client
		 */
		const ssize_t wb = write(clientfd, buffer, rb);
		if (wb < 0) {
			printf("ERROR %s writing to socket\n", strerror(errno));
			close(clientfd);
			continue;
		}

		close(clientfd);
	}
}
