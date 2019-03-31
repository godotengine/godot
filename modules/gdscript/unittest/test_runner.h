#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include "test_result.h"
#include "test_suite.h"

#include "scene/main/scene_tree.h"

class TestRunner : public SceneTree {
	GDCLASS(TestRunner, SceneTree);

public:
	virtual void init();
	virtual bool iteration(float p_time);
	virtual void finish();

protected:
	static void _bind_methods();

private:
	Ref<TestResult> m_test_result;
	Ref<TestSuite> m_test_suite;
};

#endif // TEST_RUNNER_H
