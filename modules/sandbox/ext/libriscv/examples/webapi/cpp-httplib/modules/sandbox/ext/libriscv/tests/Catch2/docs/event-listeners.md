<a id="top"></a>
# Event Listeners

An event listener is a bit like a reporter, in that it responds to various
reporter events in Catch2, but it is not expected to write any output.
Instead, an event listener performs actions within the test process, such
as performing global initialization (e.g. of a C library), or cleaning out
in-memory logs if they are not needed (the test case passed).

Unlike reporters, each registered event listener is always active. Event
listeners are always notified before reporter(s).

To write your own event listener, you should derive from `Catch::TestEventListenerBase`,
as it provides empty stubs for all reporter events, allowing you to
only override events you care for. Afterwards you have to register it
with Catch2 using `CATCH_REGISTER_LISTENER` macro, so that Catch2 knows
about it and instantiates it before running tests.

Example event listener:
```cpp
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

class testRunListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(Catch::TestRunInfo const&) override {
        lib_foo_init();
    }
};

CATCH_REGISTER_LISTENER(testRunListener)
```

_Note that you should not use any assertion macros within a Listener!_

[You can find the list of events that the listeners can react to on its
own page](reporter-events.md#top).


---

[Home](Readme.md#top)
