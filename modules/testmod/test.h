#ifndef TEST_NODE_H
#define TEST_NODE_H

#include "scene/main/node.h"

class TestNode : public Object {
	GDCLASS(TestNode, Node);

protected:
	static void _bind_methods();

public:
	TestNode();
	~TestNode();

	int add(int a, int b);
};

#endif // TEST_NODE_H
