#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include "core/os/main_loop.h"

class TestRunner : public MainLoop {
	GDCLASS(TestRunner, MainLoop);

public:
	virtual void init();
	virtual bool iteration(float p_time);
	virtual void finish();

protected:
	static void _bind_methods();
};

#endif // TEST_RUNNER_H
