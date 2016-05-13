#ifndef EDITOR_FILE_SERVER_H
#define EDITOR_FILE_SERVER_H

#include "object.h"
#include "os/thread.h"
#include "io/tcp_server.h"
#include "io/packet_peer.h"
#include "io/file_access_network.h"

class EditorFileServer : public Object {

	OBJ_TYPE(EditorFileServer,Object);

	enum Command {
		CMD_NONE,
		CMD_ACTIVATE,
		CMD_STOP,
	};


	struct ClientData {

		Thread *thread;
		Ref<StreamPeerTCP> connection;
		Map<int,FileAccess*> files;
		EditorFileServer *efs;
		bool quit;

	};

	Ref<TCP_Server> server;
	Set<Thread*> to_wait;

	static void _close_client(ClientData *cd);
	static void _subthread_start(void*s);

	Mutex *wait_mutex;
	Thread *thread;
	static void _thread_start(void*);
	bool quit;
	Command cmd;

	String password;
	int port;
	bool active;


public:

	void start();
	void stop();

	bool is_active() const;

	EditorFileServer();
	~EditorFileServer();
};

#endif // EDITOR_FILE_SERVER_H
