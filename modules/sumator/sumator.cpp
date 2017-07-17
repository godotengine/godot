/* sumator.cpp */

#include "sumator.h"

void Sumator::add(int value) {

	count += value;
}

void Sumator::reset() {

	count = 0;
}

int Sumator::get_total() const {

	return count;
}

String Sumator::get_name() const {

	return name;
}

void Sumator::_bind_methods() {

	ClassDB::bind_method("add", &Sumator::add);
	ClassDB::bind_method("reset", &Sumator::reset);
	ClassDB::bind_method("get_total", &Sumator::get_total);
	ClassDB::bind_method("get_name", &Sumator::get_name);
}

Sumator::Sumator() : name("Sumator"){
	count = 0;
}