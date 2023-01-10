/**************************************************************************/
/*  power_x11.h                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef POWER_X11_H
#define POWER_X11_H

#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"

class PowerX11 {
private:
	int nsecs_left;
	int percent_left;
	OS::PowerState power_state;

	FileAccessRef open_power_file(const char *base, const char *node, const char *key);
	bool read_power_file(const char *base, const char *node, const char *key, char *buf, size_t buflen);
	bool make_proc_acpi_key_val(char **_ptr, char **_key, char **_val);
	void check_proc_acpi_battery(const char *node, bool *have_battery, bool *charging);
	void check_proc_acpi_ac_adapter(const char *node, bool *have_ac);
	bool GetPowerInfo_Linux_proc_acpi();
	bool next_string(char **_ptr, char **_str);
	bool int_string(char *str, int *val);
	bool GetPowerInfo_Linux_proc_apm();
	bool GetPowerInfo_Linux_sys_class_power_supply();
	bool UpdatePowerInfo();

public:
	PowerX11();
	virtual ~PowerX11();

	OS::PowerState get_power_state();
	int get_power_seconds_left();
	int get_power_percent_left();
};

#endif // POWER_X11_H
