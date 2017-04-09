#include <sailfishapp.h>

#include "main/main.h"
#include "os_sailfish.h"

int main(int argc, char* argv[]) {
	OS_Sailfish os;
	os.application = SailfishApp::application(argc, argv);
	os.window = new QOpenGLWindow();

	Error error = Main::setup(argv[0], argc-1, &argv[1]);
	if (error != OK) {
		return 255;
	}

	if (Main::start()) {
		os.run(); // it is actually the OS that decides how to run
	}

	delete os.window;
	delete os.application;

	Main::cleanup();

	return os.get_exit_code();
}
