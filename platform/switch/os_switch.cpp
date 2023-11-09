#include "os_switch.h"

void OS_Switch::initialize() {
}

void OS_Switch::finalize() {
}

void OS_Switch::initialize_core() {
	OS_Unix::initialize_core();
}

void OS_Switch::finalize_core() {
	OS_Unix::finalize_core();
}

void OS_Switch::initialize_joypads() {}

void OS_Switch::delete_main_loop() {}

bool OS_Switch::_check_internal_feature_support(const String &p_feature) {}

OS_Switch::OS_Switch() {
}

OS_Switch::~OS_Switch() {}