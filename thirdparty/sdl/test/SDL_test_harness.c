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
#include <SDL3/SDL_test.h>

#include <stdlib.h> /* Needed for exit() */

/* Enable to have color in logs */
#if 1
#define COLOR_RED       "\033[0;31m"
#define COLOR_GREEN     "\033[0;32m"
#define COLOR_YELLOW    "\033[0;93m"
#define COLOR_BLUE      "\033[0;94m"
#define COLOR_END       "\033[0m"
#else
#define COLOR_RED       ""
#define COLOR_GREEN     ""
#define COLOR_BLUE      ""
#define COLOR_YELLOW    ""
#define COLOR_END       ""
#endif

/* Invalid test name/description message format */
#define SDLTEST_INVALID_NAME_FORMAT "(Invalid)"

/* Log summary message format */
#define SDLTEST_LOG_SUMMARY_FORMAT     "%s Summary: Total=%d " COLOR_GREEN "Passed=%d" COLOR_END " " COLOR_RED "Failed=%d" COLOR_END " " COLOR_BLUE "Skipped=%d" COLOR_END
#define SDLTEST_LOG_SUMMARY_FORMAT_OK  "%s Summary: Total=%d " COLOR_GREEN "Passed=%d" COLOR_END " " COLOR_GREEN "Failed=%d" COLOR_END " " COLOR_BLUE "Skipped=%d" COLOR_END

/* Final result message format */
#define SDLTEST_FINAL_RESULT_FORMAT COLOR_YELLOW ">>> %s '%s':" COLOR_END " %s\n"

struct SDLTest_TestSuiteRunner {
    struct
    {
        SDLTest_TestSuiteReference **testSuites;
        char *runSeed;
        Uint64 execKey;
        char *filter;
        int testIterations;
        bool randomOrder;
    } user;

    SDLTest_ArgumentParser argparser;
};

/* ! Timeout for single test case execution */
static Uint32 SDLTest_TestCaseTimeout = 3600;

static const char *common_harness_usage[] = {
    "[--iterations #]",
    "[--execKey #]",
    "[--seed string]",
    "[--filter suite_name|test_name]",
    "[--random-order]",
    NULL
};

char *SDLTest_GenerateRunSeed(char *buffer, int length)
{
    Uint64 randomContext = SDL_GetPerformanceCounter();
    int counter;

    if (!buffer) {
        SDLTest_LogError("Input buffer must not be NULL.");
        return NULL;
    }

    /* Sanity check input */
    if (length <= 0) {
        SDLTest_LogError("The length of the harness seed must be >0.");
        return NULL;
    }

    /* Generate a random string of alphanumeric characters */
    for (counter = 0; counter < length; counter++) {
        char ch;
        int v = SDL_rand_r(&randomContext, 10 + 26);
        if (v < 10) {
            ch = (char)('0' + v);
        } else {
            ch = (char)('A' + v - 10);
        }
        buffer[counter] = ch;
    }
    buffer[length] = '\0';

    return buffer;
}

/**
 * Generates an execution key for the fuzzer.
 *
 * \param runSeed        The run seed to use
 * \param suiteName      The name of the test suite
 * \param testName       The name of the test
 * \param iteration      The iteration count
 *
 * \returns The generated execution key to initialize the fuzzer with.
 *
 */
static Uint64 SDLTest_GenerateExecKey(const char *runSeed, const char *suiteName, const char *testName, int iteration)
{
    SDLTest_Md5Context md5Context;
    Uint64 *keys;
    char iterationString[16];
    size_t runSeedLength;
    size_t suiteNameLength;
    size_t testNameLength;
    size_t iterationStringLength;
    size_t entireStringLength;
    char *buffer;

    if (!runSeed || runSeed[0] == '\0') {
        SDLTest_LogError("Invalid runSeed string.");
        return 0;
    }

    if (!suiteName || suiteName[0] == '\0') {
        SDLTest_LogError("Invalid suiteName string.");
        return 0;
    }

    if (!testName || testName[0] == '\0') {
        SDLTest_LogError("Invalid testName string.");
        return 0;
    }

    if (iteration <= 0) {
        SDLTest_LogError("Invalid iteration count.");
        return 0;
    }

    /* Convert iteration number into a string */
    SDL_memset(iterationString, 0, sizeof(iterationString));
    (void)SDL_snprintf(iterationString, sizeof(iterationString) - 1, "%d", iteration);

    /* Combine the parameters into single string */
    runSeedLength = SDL_strlen(runSeed);
    suiteNameLength = SDL_strlen(suiteName);
    testNameLength = SDL_strlen(testName);
    iterationStringLength = SDL_strlen(iterationString);
    entireStringLength = runSeedLength + suiteNameLength + testNameLength + iterationStringLength + 1;
    buffer = (char *)SDL_malloc(entireStringLength);
    if (!buffer) {
        SDLTest_LogError("Failed to allocate buffer for execKey generation.");
        return 0;
    }
    (void)SDL_snprintf(buffer, entireStringLength, "%s%s%s%d", runSeed, suiteName, testName, iteration);

    /* Hash string and use half of the digest as 64bit exec key */
    SDLTest_Md5Init(&md5Context);
    SDLTest_Md5Update(&md5Context, (unsigned char *)buffer, (unsigned int)entireStringLength);
    SDLTest_Md5Final(&md5Context);
    SDL_free(buffer);
    keys = (Uint64 *)md5Context.digest;

    return keys[0];
}

/**
 * Set timeout handler for test.
 *
 * \param timeout Timeout interval in seconds.
 * \param callback Function that will be called after timeout has elapsed.
 *
 * \return Timer id or -1 on failure.
 */
static SDL_TimerID SDLTest_SetTestTimeout(int timeout, SDL_TimerCallback callback)
{
    Uint32 timeoutInMilliseconds;
    SDL_TimerID timerID;

    if (!callback) {
        SDLTest_LogError("Timeout callback can't be NULL");
        return 0;
    }

    if (timeout < 0) {
        SDLTest_LogError("Timeout value must be bigger than zero.");
        return 0;
    }

    /* Set timer */
    timeoutInMilliseconds = timeout * 1000;
    timerID = SDL_AddTimer(timeoutInMilliseconds, callback, 0x0);
    if (timerID == 0) {
        SDLTest_LogError("Creation of SDL timer failed: %s", SDL_GetError());
        return 0;
    }

    return timerID;
}

/**
 * Timeout handler. Aborts test run and exits harness process.
 */
static Uint32 SDLCALL SDLTest_BailOut(void *userdata, SDL_TimerID timerID, Uint32 interval)
{
    SDLTest_LogError("TestCaseTimeout timer expired. Aborting test run.");
    exit(TEST_ABORTED); /* bail out from the test */
    return 0;
}

/**
 * Execute a test using the given execution key.
 *
 * \param testSuite Suite containing the test case.
 * \param testCase Case to execute.
 * \param execKey Execution key for the fuzzer.
 * \param forceTestRun Force test to run even if test was disabled in suite.
 *
 * \returns Test case result.
 */
static int SDLTest_RunTest(SDLTest_TestSuiteReference *testSuite, const SDLTest_TestCaseReference *testCase, Uint64 execKey, bool forceTestRun)
{
    SDL_TimerID timer = 0;
    int testCaseResult = 0;
    int testResult = 0;
    int fuzzerCount;
    void *data = NULL;

    if (!testSuite || !testCase || !testSuite->name || !testCase->name) {
        SDLTest_LogError("Setup failure: testSuite or testCase references NULL");
        return TEST_RESULT_SETUP_FAILURE;
    }

    if (!testCase->enabled && forceTestRun == false) {
        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, "Skipped (Disabled)");
        return TEST_RESULT_SKIPPED;
    }

    /* Initialize fuzzer */
    SDLTest_FuzzerInit(execKey);

    /* Reset assert tracker */
    SDLTest_ResetAssertSummary();

    /* Set timeout timer */
    timer = SDLTest_SetTestTimeout(SDLTest_TestCaseTimeout, SDLTest_BailOut);

    /* Maybe run suite initializer function */
    if (testSuite->testSetUp) {
        testSuite->testSetUp(&data);
        if (SDLTest_AssertSummaryToTestResult() == TEST_RESULT_FAILED) {
            SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Suite Setup", testSuite->name, COLOR_RED "Failed" COLOR_END);
            return TEST_RESULT_SETUP_FAILURE;
        }
    }

    /* Run test case function */
    testCaseResult = testCase->testCase(data);

    /* Convert test execution result into harness result */
    if (testCaseResult == TEST_SKIPPED) {
        /* Test was programmatically skipped */
        testResult = TEST_RESULT_SKIPPED;
    } else if (testCaseResult == TEST_STARTED) {
        /* Test did not return a TEST_COMPLETED value; assume it failed */
        testResult = TEST_RESULT_FAILED;
    } else if (testCaseResult == TEST_ABORTED) {
        /* Test was aborted early; assume it failed */
        testResult = TEST_RESULT_FAILED;
    } else {
        /* Perform failure analysis based on asserts */
        testResult = SDLTest_AssertSummaryToTestResult();
    }

    /* Maybe run suite cleanup function (ignore failed asserts) */
    if (testSuite->testTearDown) {
        testSuite->testTearDown(data);
    }

    /* Cancel timeout timer */
    if (timer) {
        SDL_RemoveTimer(timer);
    }

    /* Report on asserts and fuzzer usage */
    fuzzerCount = SDLTest_GetFuzzerInvocationCount();
    if (fuzzerCount > 0) {
        SDLTest_Log("Fuzzer invocations: %d", fuzzerCount);
    }

    /* Final log based on test execution result */
    if (testCaseResult == TEST_SKIPPED) {
        /* Test was programmatically skipped */
        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, COLOR_BLUE "Skipped (Programmatically)" COLOR_END);
    } else if (testCaseResult == TEST_STARTED) {
        /* Test did not return a TEST_COMPLETED value; assume it failed */
        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, COLOR_RED "Failed (test started, but did not return TEST_COMPLETED)" COLOR_END);
    } else if (testCaseResult == TEST_ABORTED) {
        /* Test was aborted early; assume it failed */
        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", testCase->name, COLOR_RED "Failed (Aborted)" COLOR_END);
    } else {
        SDLTest_LogAssertSummary();
    }

    return testResult;
}

/* Prints summary of all suites/tests contained in the given reference */
#if 0
static void SDLTest_LogTestSuiteSummary(SDLTest_TestSuiteReference *testSuites)
{
    int suiteCounter;
    int testCounter;
    SDLTest_TestSuiteReference *testSuite;
    SDLTest_TestCaseReference *testCase;

    /* Loop over all suites */
    suiteCounter = 0;
    while (&testSuites[suiteCounter]) {
        testSuite=&testSuites[suiteCounter];
        suiteCounter++;
        SDLTest_Log("Test Suite %i - %s\n", suiteCounter,
            (testSuite->name) ? testSuite->name : SDLTEST_INVALID_NAME_FORMAT);

        /* Loop over all test cases */
        testCounter = 0;
        while (testSuite->testCases[testCounter]) {
            testCase=(SDLTest_TestCaseReference *)testSuite->testCases[testCounter];
            testCounter++;
            SDLTest_Log("  Test Case %i - %s: %s", testCounter,
                (testCase->name) ? testCase->name : SDLTEST_INVALID_NAME_FORMAT,
                (testCase->description) ? testCase->description : SDLTEST_INVALID_NAME_FORMAT);
        }
    }
}
#endif

/* Gets a timer value in seconds */
static float GetClock(void)
{
    float currentClock = SDL_GetPerformanceCounter() / (float)SDL_GetPerformanceFrequency();
    return currentClock;
}

/**
 * Execute a test suite using the given run seed and execution key.
 *
 * The filter string is matched to the suite name (full comparison) to select a single suite,
 * or if no suite matches, it is matched to the test names (full comparison) to select a single test.
 *
 * \param runner The runner to execute.
 *
 * \returns Test run result; 0 when all tests passed, 1 if any tests failed.
 */
int SDLTest_ExecuteTestSuiteRunner(SDLTest_TestSuiteRunner *runner)
{
    int totalNumberOfTests = 0;
    int failedNumberOfTests = 0;
    int suiteCounter;
    int testCounter;
    int iterationCounter;
    SDLTest_TestSuiteReference *testSuite;
    const SDLTest_TestCaseReference *testCase;
    const char *runSeed = NULL;
    const char *currentSuiteName;
    const char *currentTestName;
    Uint64 execKey;
    float runStartSeconds;
    float suiteStartSeconds;
    float testStartSeconds;
    float runEndSeconds;
    float suiteEndSeconds;
    float testEndSeconds;
    float runtime;
    int suiteFilter = 0;
    const char *suiteFilterName = NULL;
    int testFilter = 0;
    const char *testFilterName = NULL;
    bool forceTestRun = false;
    int testResult = 0;
    int runResult = 0;
    int totalTestFailedCount = 0;
    int totalTestPassedCount = 0;
    int totalTestSkippedCount = 0;
    int testFailedCount = 0;
    int testPassedCount = 0;
    int testSkippedCount = 0;
    int countSum = 0;
    const SDLTest_TestCaseReference **failedTests;
    char generatedSeed[16 + 1];
    int nbSuites = 0;
    int i = 0;
    int *arraySuites = NULL;

    /* Sanitize test iterations */
    if (runner->user.testIterations < 1) {
        runner->user.testIterations = 1;
    }

    /* Generate run see if we don't have one already */
    if (!runner->user.runSeed || runner->user.runSeed[0] == '\0') {
        runSeed = SDLTest_GenerateRunSeed(generatedSeed, 16);
        if (!runSeed) {
            SDLTest_LogError("Generating a random seed failed");
            return 2;
        }
    } else {
        runSeed = runner->user.runSeed;
    }

    /* Reset per-run counters */
    totalTestFailedCount = 0;
    totalTestPassedCount = 0;
    totalTestSkippedCount = 0;

    /* Take time - run start */
    runStartSeconds = GetClock();

    /* Log run with fuzzer parameters */
    SDLTest_Log("::::: Test Run /w seed '%s' started\n", runSeed);

    /* Count the total number of tests */
    suiteCounter = 0;
    while (runner->user.testSuites[suiteCounter]) {
        testSuite = runner->user.testSuites[suiteCounter];
        suiteCounter++;
        testCounter = 0;
        while (testSuite->testCases[testCounter]) {
            testCounter++;
            totalNumberOfTests++;
        }
    }

    if (totalNumberOfTests == 0) {
        SDLTest_LogError("No tests to run?");
        return -1;
    }

    /* Pre-allocate an array for tracking failed tests (potentially all test cases) */
    failedTests = (const SDLTest_TestCaseReference **)SDL_malloc(totalNumberOfTests * sizeof(SDLTest_TestCaseReference *));
    if (!failedTests) {
        SDLTest_LogError("Unable to allocate cache for failed tests");
        return -1;
    }

    /* Initialize filtering */
    if (runner->user.filter && runner->user.filter[0] != '\0') {
        /* Loop over all suites to check if we have a filter match */
        suiteCounter = 0;
        while (runner->user.testSuites[suiteCounter] && suiteFilter == 0) {
            testSuite = runner->user.testSuites[suiteCounter];
            suiteCounter++;
            if (testSuite->name && SDL_strcasecmp(runner->user.filter, testSuite->name) == 0) {
                /* Matched a suite name */
                suiteFilter = 1;
                suiteFilterName = testSuite->name;
                SDLTest_Log("Filtering: running only suite '%s'", suiteFilterName);
                break;
            }

            /* Within each suite, loop over all test cases to check if we have a filter match */
            testCounter = 0;
            while (testSuite->testCases[testCounter] && testFilter == 0) {
                testCase = testSuite->testCases[testCounter];
                testCounter++;
                if (testCase->name && SDL_strcasecmp(runner->user.filter, testCase->name) == 0) {
                    /* Matched a test name */
                    suiteFilter = 1;
                    suiteFilterName = testSuite->name;
                    testFilter = 1;
                    testFilterName = testCase->name;
                    SDLTest_Log("Filtering: running only test '%s' in suite '%s'", testFilterName, suiteFilterName);
                    break;
                }
            }
        }

        if (suiteFilter == 0 && testFilter == 0) {
            SDLTest_LogError("Filter '%s' did not match any test suite/case.", runner->user.filter);
            for (suiteCounter = 0; runner->user.testSuites[suiteCounter]; ++suiteCounter) {
                testSuite = runner->user.testSuites[suiteCounter];
                if (testSuite->name) {
                    SDLTest_Log("Test suite: %s", testSuite->name);
                }

                /* Within each suite, loop over all test cases to check if we have a filter match */
                for (testCounter = 0; testSuite->testCases[testCounter]; ++testCounter) {
                    testCase = testSuite->testCases[testCounter];
                    SDLTest_Log("      test: %s%s", testCase->name, testCase->enabled ? "" : " (disabled)");
                }
            }
            SDLTest_Log("Exit code: 2");
            SDL_free((void *)failedTests);
            return 2;
        }

        runner->user.randomOrder = false;
    }

    /* Number of test suites */
    while (runner->user.testSuites[nbSuites]) {
        nbSuites++;
    }

    arraySuites = SDL_malloc(nbSuites * sizeof(int));
    if (!arraySuites) {
        SDL_free((void *)failedTests);
        return SDL_OutOfMemory();
    }
    for (i = 0; i < nbSuites; i++) {
        arraySuites[i] = i;
    }

    /* Mix the list of suites to run them in random order */
    {
        /* Exclude last test "subsystemsTestSuite" which is said to interfere with other tests */
        nbSuites--;

        if (runner->user.execKey != 0) {
            execKey = runner->user.execKey;
        } else {
            /* dummy values to have random numbers working */
            execKey = SDLTest_GenerateExecKey(runSeed, "random testSuites", "initialisation", 1);
        }

        /* Initialize fuzzer */
        SDLTest_FuzzerInit(execKey);

        i = 100;
        while (i--) {
            int a, b;
            int tmp;
            a = SDLTest_RandomIntegerInRange(0, nbSuites - 1);
            b = SDLTest_RandomIntegerInRange(0, nbSuites - 1);
            /*
             * NB: prevent swapping here to make sure the tests start with the same
             * random seed (whether they are run in order or not).
             * So we consume same number of SDLTest_RandomIntegerInRange() in all cases.
             *
             * If some random value were used at initialization before the tests start, the --seed wouldn't do the same with or without randomOrder.
             */
            /* Swap */
            if (runner->user.randomOrder) {
                tmp = arraySuites[b];
                arraySuites[b] = arraySuites[a];
                arraySuites[a] = tmp;
            }
        }

        /* re-add last lest */
        nbSuites++;
    }

    /* Loop over all suites */
    for (i = 0; i < nbSuites; i++) {
        suiteCounter = arraySuites[i];
        testSuite = runner->user.testSuites[suiteCounter];
        currentSuiteName = (testSuite->name ? testSuite->name : SDLTEST_INVALID_NAME_FORMAT);
        suiteCounter++;

        /* Filter suite if flag set and we have a name */
        if (suiteFilter == 1 && suiteFilterName && testSuite->name &&
            SDL_strcasecmp(suiteFilterName, testSuite->name) != 0) {
            /* Skip suite */
            SDLTest_Log("===== Test Suite %i: '%s' " COLOR_BLUE "skipped" COLOR_END "\n",
                        suiteCounter,
                        currentSuiteName);
        } else {

            int nbTestCases = 0;
            int *arrayTestCases;
            int j;
            while (testSuite->testCases[nbTestCases]) {
                nbTestCases++;
            }

            arrayTestCases = SDL_malloc(nbTestCases * sizeof(int));
            if (!arrayTestCases) {
                SDL_free(arraySuites);
                SDL_free((void *)failedTests);
                return SDL_OutOfMemory();
            }
            for (j = 0; j < nbTestCases; j++) {
                arrayTestCases[j] = j;
            }

            /* Mix the list of testCases to run them in random order */
            j = 100;
            while (j--) {
                int a, b;
                int tmp;
                a = SDLTest_RandomIntegerInRange(0, nbTestCases - 1);
                b = SDLTest_RandomIntegerInRange(0, nbTestCases - 1);
                /* Swap */
                /* See previous note */
                if (runner->user.randomOrder) {
                    tmp = arrayTestCases[b];
                    arrayTestCases[b] = arrayTestCases[a];
                    arrayTestCases[a] = tmp;
                }
            }

            /* Reset per-suite counters */
            testFailedCount = 0;
            testPassedCount = 0;
            testSkippedCount = 0;

            /* Take time - suite start */
            suiteStartSeconds = GetClock();

            /* Log suite started */
            SDLTest_Log("===== Test Suite %i: '%s' started\n",
                        suiteCounter,
                        currentSuiteName);

            /* Loop over all test cases */
            for (j = 0; j < nbTestCases; j++) {
                testCounter = arrayTestCases[j];
                testCase = testSuite->testCases[testCounter];
                currentTestName = (testCase->name ? testCase->name : SDLTEST_INVALID_NAME_FORMAT);
                testCounter++;

                /* Filter tests if flag set and we have a name */
                if (testFilter == 1 && testFilterName && testCase->name &&
                    SDL_strcasecmp(testFilterName, testCase->name) != 0) {
                    /* Skip test */
                    SDLTest_Log("===== Test Case %i.%i: '%s' " COLOR_BLUE "skipped" COLOR_END "\n",
                                suiteCounter,
                                testCounter,
                                currentTestName);
                } else {
                    /* Override 'disabled' flag if we specified a test filter (i.e. force run for debugging) */
                    if (testFilter == 1 && !testCase->enabled) {
                        SDLTest_Log("Force run of disabled test since test filter was set");
                        forceTestRun = true;
                    }

                    /* Take time - test start */
                    testStartSeconds = GetClock();

                    /* Log test started */
                    SDLTest_Log(COLOR_YELLOW "----- Test Case %i.%i: '%s' started" COLOR_END,
                                suiteCounter,
                                testCounter,
                                currentTestName);
                    if (testCase->description && testCase->description[0] != '\0') {
                        SDLTest_Log("Test Description: '%s'",
                                    (testCase->description) ? testCase->description : SDLTEST_INVALID_NAME_FORMAT);
                    }

                    /* Loop over all iterations */
                    iterationCounter = 0;
                    while (iterationCounter < runner->user.testIterations) {
                        iterationCounter++;

                        if (runner->user.execKey != 0) {
                            execKey = runner->user.execKey;
                        } else {
                            execKey = SDLTest_GenerateExecKey(runSeed, testSuite->name, testCase->name, iterationCounter);
                        }

                        SDLTest_Log("Test Iteration %i: execKey %" SDL_PRIu64, iterationCounter, execKey);
                        testResult = SDLTest_RunTest(testSuite, testCase, execKey, forceTestRun);

                        if (testResult == TEST_RESULT_PASSED) {
                            testPassedCount++;
                            totalTestPassedCount++;
                        } else if (testResult == TEST_RESULT_SKIPPED) {
                            testSkippedCount++;
                            totalTestSkippedCount++;
                        } else {
                            testFailedCount++;
                            totalTestFailedCount++;
                        }
                    }

                    /* Take time - test end */
                    testEndSeconds = GetClock();
                    runtime = testEndSeconds - testStartSeconds;
                    if (runtime < 0.0f) {
                        runtime = 0.0f;
                    }

                    if (runner->user.testIterations > 1) {
                        /* Log test runtime */
                        SDLTest_Log("Runtime of %i iterations: %.1f sec", runner->user.testIterations, runtime);
                        SDLTest_Log("Average Test runtime: %.5f sec", runtime / (float)runner->user.testIterations);
                    } else {
                        /* Log test runtime */
                        SDLTest_Log("Total Test runtime: %.1f sec", runtime);
                    }

                    /* Log final test result */
                    switch (testResult) {
                    case TEST_RESULT_PASSED:
                        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Test", currentTestName, COLOR_GREEN "Passed" COLOR_END);
                        break;
                    case TEST_RESULT_FAILED:
                        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", currentTestName, COLOR_RED "Failed" COLOR_END);
                        break;
                    case TEST_RESULT_NO_ASSERT:
                        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Test", currentTestName, COLOR_BLUE "No Asserts" COLOR_END);
                        break;
                    }

                    /* Collect failed test case references for repro-step display */
                    if (testResult == TEST_RESULT_FAILED) {
                        failedTests[failedNumberOfTests] = testCase;
                        failedNumberOfTests++;
                    }
                }
            }

            /* Take time - suite end */
            suiteEndSeconds = GetClock();
            runtime = suiteEndSeconds - suiteStartSeconds;
            if (runtime < 0.0f) {
                runtime = 0.0f;
            }

            /* Log suite runtime */
            SDLTest_Log("Total Suite runtime: %.1f sec", runtime);

            /* Log summary and final Suite result */
            countSum = testPassedCount + testFailedCount + testSkippedCount;
            if (testFailedCount == 0) {
                SDLTest_Log(SDLTEST_LOG_SUMMARY_FORMAT_OK, "Suite", countSum, testPassedCount, testFailedCount, testSkippedCount);
                SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Suite", currentSuiteName, COLOR_GREEN "Passed" COLOR_END);
            } else {
                SDLTest_LogError(SDLTEST_LOG_SUMMARY_FORMAT, "Suite", countSum, testPassedCount, testFailedCount, testSkippedCount);
                SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Suite", currentSuiteName, COLOR_RED "Failed" COLOR_END);
            }

            SDL_free(arrayTestCases);
        }
    }

    SDL_free(arraySuites);

    /* Take time - run end */
    runEndSeconds = GetClock();
    runtime = runEndSeconds - runStartSeconds;
    if (runtime < 0.0f) {
        runtime = 0.0f;
    }

    /* Log total runtime */
    SDLTest_Log("Total Run runtime: %.1f sec", runtime);

    /* Log summary and final run result */
    countSum = totalTestPassedCount + totalTestFailedCount + totalTestSkippedCount;
    if (totalTestFailedCount == 0) {
        runResult = 0;
        SDLTest_Log(SDLTEST_LOG_SUMMARY_FORMAT_OK, "Run", countSum, totalTestPassedCount, totalTestFailedCount, totalTestSkippedCount);
        SDLTest_Log(SDLTEST_FINAL_RESULT_FORMAT, "Run /w seed", runSeed, COLOR_GREEN "Passed" COLOR_END);
    } else {
        runResult = 1;
        SDLTest_LogError(SDLTEST_LOG_SUMMARY_FORMAT, "Run", countSum, totalTestPassedCount, totalTestFailedCount, totalTestSkippedCount);
        SDLTest_LogError(SDLTEST_FINAL_RESULT_FORMAT, "Run /w seed", runSeed, COLOR_RED "Failed" COLOR_END);
    }

    /* Print repro steps for failed tests */
    if (failedNumberOfTests > 0) {
        SDLTest_Log("Harness input to repro failures:");
        for (testCounter = 0; testCounter < failedNumberOfTests; testCounter++) {
            SDLTest_Log(COLOR_RED " --seed %s --filter %s" COLOR_END, runSeed, failedTests[testCounter]->name);
        }
    }
    SDL_free((void *)failedTests);

    SDLTest_Log("Exit code: %d", runResult);
    return runResult;
}

static int SDLCALL SDLTest_TestSuiteCommonArg(void *data, char **argv, int index)
{
    SDLTest_TestSuiteRunner *runner = data;

    if (SDL_strcasecmp(argv[index], "--iterations") == 0) {
        if (argv[index + 1]) {
            runner->user.testIterations = SDL_atoi(argv[index + 1]);
            if (runner->user.testIterations < 1) {
                runner->user.testIterations = 1;
            }
            return 2;
        }
    }
    else if (SDL_strcasecmp(argv[index], "--execKey") == 0) {
        if (argv[index + 1]) {
            (void)SDL_sscanf(argv[index + 1], "%" SDL_PRIu64, &runner->user.execKey);
            return 2;
        }
    }
    else if (SDL_strcasecmp(argv[index], "--seed") == 0) {
        if (argv[index + 1]) {
            runner->user.runSeed = SDL_strdup(argv[index + 1]);
            return 2;
        }
    }
    else if (SDL_strcasecmp(argv[index], "--filter") == 0) {
        if (argv[index + 1]) {
            runner->user.filter = SDL_strdup(argv[index + 1]);
            return 2;
        }
    }
    else if (SDL_strcasecmp(argv[index], "--random-order") == 0) {
        runner->user.randomOrder = true;
        return 1;
    }
    return 0;
}

SDLTest_TestSuiteRunner *SDLTest_CreateTestSuiteRunner(SDLTest_CommonState *state, SDLTest_TestSuiteReference *testSuites[])
{
    SDLTest_TestSuiteRunner *runner;
    SDLTest_ArgumentParser *argparser;

    if (!state) {
        SDLTest_LogError("SDL Test Suites require a common state");
        return NULL;
    }

    runner = SDL_calloc(1, sizeof(SDLTest_TestSuiteRunner));
    if (!runner) {
        SDLTest_LogError("Failed to allocate memory for test suite runner");
        return NULL;
    }
    runner->user.testSuites = testSuites;

    runner->argparser.parse_arguments = SDLTest_TestSuiteCommonArg;
    runner->argparser.usage = common_harness_usage;
    runner->argparser.data = runner;

    /* Find last argument description and append our description */
    argparser = state->argparser;
    for (;;) {
        if (argparser->next == NULL) {
            argparser->next = &runner->argparser;
            break;
        }
        argparser = argparser->next;

    }

    return runner;
}

void SDLTest_DestroyTestSuiteRunner(SDLTest_TestSuiteRunner *runner) {

    SDL_free(runner->user.filter);
    SDL_free(runner->user.runSeed);
    SDL_free(runner);
}
