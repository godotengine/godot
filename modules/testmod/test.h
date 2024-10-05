#ifndef TEST_NODE_H
#define TEST_NODE_H

#include "scene/main/node.h"

class TestNode : public Node {
	GDCLASS(TestNode, Node);

private:
	int count;

protected:
	static void _bind_methods();

public:
	TestNode();
	~TestNode();
	void add(int p_value);
	void reset();
	int get_total() const;
};

#endif // TEST_NODE_H
