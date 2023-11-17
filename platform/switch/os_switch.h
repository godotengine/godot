/**************************************************************************/
/*  os_switch.h                                                           */
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

#ifndef OS_SWITCH_H
#define OS_SWITCH_H

#include "core/crypto/crypto_core.h"
#include "drivers/unix/os_unix.h"

#include "switch_wrapper.h"

#include <string>
#include <vector>

class OS_Switch : public OS_Unix {
	MainLoop *_main_loop = nullptr;
	CryptoCore::RandomGenerator _random_generator;
	std::vector<std::string> _args;

private:
protected:
	virtual void initialize() override;
	virtual void finalize() override;

	virtual void initialize_core() override;
	virtual void finalize_core() override;

	virtual void initialize_joypads() override;

	Error get_entropy(uint8_t *r_buffer, int p_bytes) override;

	virtual void set_main_loop(MainLoop *p_main_loop) override { _main_loop = p_main_loop; }
	virtual void delete_main_loop() override;

public:
	virtual bool _check_internal_feature_support(const String &p_feature) override;

	String get_executable_path() const override { return String(_args[0].c_str()); }

	String get_data_path() const override;
	String get_config_path() const override;
	String get_cache_path() const override;
	String get_user_data_dir() const override;

	// actual switch appends here
	void run();

	// we do not care about process, we won't run any on the switch
	virtual Error execute(const String &p_path, const List<String> &p_arguments, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr, bool p_open_console = false) override { return ERR_UNAVAILABLE; }
	virtual Error create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id = nullptr, bool p_open_console = false) override { return ERR_UNAVAILABLE; }
	virtual Error kill(const ProcessID &p_pid) override { return ERR_UNAVAILABLE; }
	virtual bool is_process_running(const ProcessID &p_pid) const override { return false; }

	// we do not care about environment, we won't use any on the switch
	virtual bool has_environment(const String &p_var) const override { return false; }
	virtual String get_environment(const String &p_var) const override { return ""; }
	virtual void set_environment(const String &p_var, const String &p_value) const override {}
	virtual void unset_environment(const String &p_var) const override {}

	virtual String get_name() const override { return "Switch"; }
	virtual String get_distribution_name() const override { return "Horizon"; }
	virtual String get_version() const override { return ""; };

	virtual MainLoop *get_main_loop() const override { return _main_loop; }

	OS_Switch(const std::vector<std::string> &args);
	virtual ~OS_Switch();
};

#endif // OS_SWITCH_H
