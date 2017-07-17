/* priority_queue.h */
#ifndef PRIORITY_QUEUE_H
#define PRIOTITY_QUEUE_H

#include "reference.h"

class PriorityQueue : public Reference {
	GDCLASS(PriorityQueue, Reference);

	Array items;
	Array costs;

protected:
	static void _bind_methods();

public:
	PriorityQueue();

	Variant dequeue();
	void enqueue(float cost, Variant item);

	
};

#endif