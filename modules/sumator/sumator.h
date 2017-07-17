/* sumator.h */
#ifndef SUMATOR_H
#define SUMATOR_H

#include "reference.h"
#include "../core/ustring.h"

class Sumator : public Reference {
	GDCLASS(Sumator, Reference);

	String name;
	int count;

protected:
	static void _bind_methods();

public:
	void add(int value);
	void reset();
	int get_total() const;

	String get_name() const;

	Sumator();
};

#endif