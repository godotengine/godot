/* summator.h */
#ifndef SUMMATOR_H
#define SUMMATOR_H

#include "core/reference.h"

class Summator : public Reference {
	GDCLASS(Summator, Reference);

	int count;

	protected:
	static void_bind_methods();

public:
	void add(int value);
		void reset();
	int get_total() const;

	Summator();
};

 #endif
