/* 
 * Helper function for running tests on external modules.
 * Copyright 2023 the halcyon authors
 */

#ifndef MAIN_TESTS_CATCH_TESTING_H
#define MAIN_TESTS_CATCH_TESTING_H

#ifdef CATCH_TESTS
#include <catch2/catch_session.hpp>

void run_catch_tests(int argc, char **argv) {

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
		return;
	}
	// This returns an int that's meant to propagate to the `main()` return value,
	// but trying to return it leads to SIGABRT.
	session.run();
}
#endif  // CATCH_TESTS

#endif  // MAIN_TESTS_CATCH_TESTING_H
