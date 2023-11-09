#include "os_switch.h"

void OS_Switch::initialize() {}
void OS_Switch::finalize() {}
void OS_Switch::finalize_core() {}

void OS_Switch::initialize_joypads() {}

void OS_Switch::delete_main_loop() {}

bool OS_Switch::_check_internal_feature_support(const String &p_feature) {}
Vector<String> OS_Switch::get_video_adapter_driver_info() const {}

String OS_Switch::get_stdin_string() {}
Error OS_Switch::get_entropy(uint8_t *r_buffer, int p_bytes) {}

String OS_Switch::get_version() const {}

OS::DateTime OS_Switch::get_datetime(bool utc) const {}
OS::TimeZoneInfo OS_Switch::get_time_zone_info() const {}

void OS_Switch::delay_usec(uint32_t p_usec) const {}

uint64_t OS_Switch::get_ticks_usec() const {}

OS_Switch::OS_Switch() {
}

OS_Switch::~OS_Switch() {}