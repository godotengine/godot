/*************************************************************************/
/*  godot_main_osx.mm                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "main/main.h"

#include "os_osx.h"

#if CATCH_TESTS
#include <catch2/catch_session.hpp>
int run_catch_tests(int argc, char **argv) {

Catch::Session session; // There must be exactly one instance

  bool test_external = false;

  // Build a new parser on top of Catch2's
  using namespace Catch::Clara;
	// Register the command line option for running Catch2 tests, so that its
	// command line parser won't halt because it's unrecognized.
  auto cli
    = session.cli()           // Get Catch2's command line parser
    | Opt(test_external) // bind variable to a new option, with a hint string
        ["--test-external"]    // the option names it will respond to
        ("The option that tells Godot to run external tests.");

  // Now pass the new composite back to Catch2 so it uses that
  session.cli(cli);

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine(argc, argv);
	if (returnCode != 0) {
		fprintf(stderr, "Error parsing command line arguments for Catch2.");
		return returnCode;
	}
	return session.run();
}
#endif

int main(int argc, char **argv) {
	int first_arg = 1;
	const char *dbg_arg = "-NSDocumentRevisionsDebugMode";
	printf("arguments\n");
	for (int i = 0; i < argc; i++) {
		if (strcmp(dbg_arg, argv[i]) == 0)
			first_arg = i + 2;
		printf("%i: %s\n", i, argv[i]);
	};

#ifdef DEBUG_ENABLED
	// lets report the path we made current after all that
	char cwd[4096];
	getcwd(cwd, 4096);
	printf("Current path: %s\n", cwd);
#endif

	OS_OSX os;
	Error err;

	if (os.open_with_filename != "") {
		char *argv_c = (char *)malloc(os.open_with_filename.utf8().size());
		memcpy(argv_c, os.open_with_filename.utf8().get_data(), os.open_with_filename.utf8().size());
		err = Main::setup(argv[0], 1, &argv_c);
		free(argv_c);
	} else {
		err = Main::setup(argv[0], argc - first_arg, &argv[first_arg]);
	}

	if (err == ERR_HELP) { // Returned by --help and --version, so success.
		return 0;
	} else if (err != OK) {
		return 255;
	}

	if (Main::start()) {
		os.run(); // it is actually the OS that decides how to run
	} else {
		List<String> args = os.get_cmdline_args();
		if (args.find("--test-external") != nullptr) {
			#if CATCH_TESTS
			run_catch_tests(argc, argv);
			#endif
		}
	}
	Main::cleanup();

	return os.get_exit_code();
};
