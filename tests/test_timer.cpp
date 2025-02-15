#ifndef TEST_STRING_H
#define TEST_STRING_H

#include "tests/test_macros.h"
#include "scene/main/timer.h"


TEST_CASE("[SceneTree][Timer] Testing Autostart and Pause Behavior") {
	Timer timer;

	// starting the timer
    timer.set_autostart(true);

	// checking if the timer is not running (it should be running)
	CHECK_FALSE(timer.is_stopped() == true);
    
	// pauses the timer and checks if it pauses
    timer.set_paused(true);
	CHECK_FALSE(timer.is_paused() == false);
    
	// unpauses the time and checks if it resumes 
    timer.set_paused(false);
	CHECK_FALSE(timer.is_paused() == true);
}

#endif 