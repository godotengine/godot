/* priority_queue.cpp */

#include "priority_queue.h"

Variant PriorityQueue::dequeue()
{
	int bestIndex = 0;


	for( int i = 0; i < items.size(); i++ )
	{
		if (costs[i] < costs[bestIndex])
			bestIndex = i;
	}

	Variant bestItem = items[bestIndex];
	items.remove(bestIndex);

	return bestItem;
}


void PriorityQueue::enqueue(float cost, Variant value)
{
	items.append(value);
	costs.append(cost);
}

void PriorityQueue::_bind_methods() {

	//ClassDB::bind_method("add", &Sumator::add);
	ClassDB::bind_method(D_METHOD("enqueue", "cost", "value"), &PriorityQueue::enqueue);
	ClassDB::bind_method(D_METHOD("dequeue"), &PriorityQueue::dequeue);
}


PriorityQueue::PriorityQueue() {

}