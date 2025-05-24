/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/**
 *  Test suite related functions of SDL test framework.
 *
 *  This code is a part of the SDL test library, not the main SDL library.
 */

/*
  Defines types for test case definitions and the test execution harness API.

  Based on original GSOC code by Markus Kauppila <markus.kauppila@gmail.com>
*/

#ifndef SDL_test_h_arness_h
#define SDL_test_h_arness_h

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_test_common.h> /* SDLTest_CommonState */

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* ! Definitions for test case structures */
#define TEST_ENABLED  1
#define TEST_DISABLED 0

/* ! Definition of all the possible test return values of the test case method */
#define TEST_ABORTED        -1
#define TEST_STARTED         0
#define TEST_COMPLETED       1
#define TEST_SKIPPED         2

/* ! Definition of all the possible test results for the harness */
#define TEST_RESULT_PASSED              0
#define TEST_RESULT_FAILED              1
#define TEST_RESULT_NO_ASSERT           2
#define TEST_RESULT_SKIPPED             3
#define TEST_RESULT_SETUP_FAILURE       4

/* !< Function pointer to a test case setup function (run before every test) */
typedef void (SDLCALL *SDLTest_TestCaseSetUpFp)(void **arg);

/* !< Function pointer to a test case function */
typedef int (SDLCALL *SDLTest_TestCaseFp)(void *arg);

/* !< Function pointer to a test case teardown function (run after every test) */
typedef void  (SDLCALL *SDLTest_TestCaseTearDownFp)(void *arg);

/*
 * Holds information about a single test case.
 */
typedef struct SDLTest_TestCaseReference {
    /* !< Func2Stress */
    SDLTest_TestCaseFp testCase;
    /* !< Short name (or function name) "Func2Stress" */
    const char *name;
    /* !< Long name or full description "This test pushes func2() to the limit." */
    const char *description;
    /* !< Set to TEST_ENABLED or TEST_DISABLED (test won't be run) */
    int enabled;
} SDLTest_TestCaseReference;

/*
 * Holds information about a test suite (multiple test cases).
 */
typedef struct SDLTest_TestSuiteReference {
    /* !< "PlatformSuite" */
    const char *name;
    /* !< The function that is run before each test. NULL skips. */
    SDLTest_TestCaseSetUpFp testSetUp;
    /* !< The test cases that are run as part of the suite. Last item should be NULL. */
    const SDLTest_TestCaseReference **testCases;
    /* !< The function that is run after each test. NULL skips. */
    SDLTest_TestCaseTearDownFp testTearDown;
} SDLTest_TestSuiteReference;


/*
 * Generates a random run seed string for the harness. The generated seed
 * will contain alphanumeric characters (0-9A-Z).
 *
 * \param buffer Buffer in which to generate the random seed. Must have a capacity of at least length + 1 characters.
 * \param length Number of alphanumeric characters to write to buffer, must be >0
 *
 * \returns A null-terminated seed string and equal to the in put buffer on success, NULL on failure
 */
char * SDLCALL SDLTest_GenerateRunSeed(char *buffer, int length);

/*
 * Holds information about the execution of test suites.
 * */
typedef struct SDLTest_TestSuiteRunner SDLTest_TestSuiteRunner;

/*
 * Create a new test suite runner, that will execute the given test suites.
 * It will register the harness cli arguments to the common SDL state.
 *
 * \param state Common SDL state on which to register CLI arguments.
 * \param testSuites NULL-terminated test suites containing test cases.
 *
 * \returns the test run result: 0 when all tests passed, 1 if any tests failed.
 */
SDLTest_TestSuiteRunner * SDLCALL SDLTest_CreateTestSuiteRunner(SDLTest_CommonState *state, SDLTest_TestSuiteReference *testSuites[]);

/*
 * Destroy a test suite runner.
 * It will unregister the harness cli arguments to the common SDL state.
 *
 * \param runner The runner that should be destroyed.
 */
void SDLCALL SDLTest_DestroyTestSuiteRunner(SDLTest_TestSuiteRunner *runner);

/*
 * Execute a test suite, using the configured run seed, execution key, filter, etc.
 *
 * \param runner The runner that should be executed.
 *
 * \returns the test run result: 0 when all tests passed, 1 if any tests failed.
 */
int SDLCALL SDLTest_ExecuteTestSuiteRunner(SDLTest_TestSuiteRunner *runner);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_test_h_arness_h */
