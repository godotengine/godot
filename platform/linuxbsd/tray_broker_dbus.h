/*************************************************************************/
/*  tray_broker_dbus.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TRAY_BROKER_DBUS_H
#define TRAY_BROKER_DBUS_H

#ifdef DBUS_ENABLED

#include "core/error/error_list.h"
#include "core/io/tray_broker.h"
#include "core/templates/hash_map.h"
#include "core/templates/optional.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"
#include <dbus/dbus.h>
#include <cstdint>

struct _dbus_fixed_arr {
	void *data;
	int32_t len;
};

struct _dbus_msg_tuple {
	int32_t type;
	int32_t category;
	void *value;
	const char *signature;
	struct _dbus_msg_tuple *sub;
};

enum ValueCategory {
	DBUS,
	GODOT,
};

enum ValueType {
	IMAGE,
	BITMAP,
};

Error _append_dbus_container(DBusMessageIter *iter, struct _dbus_msg_tuple *t);
Error _append_dbus_msg_tuple(DBusMessageIter *iter, struct _dbus_msg_tuple *args);
Error _unmarshall_dbus_basic(DBusMessageIter *iter, Vector<Variant> &res);
Error _unmarshall_dbus_container(DBusMessageIter *iter, Vector<Variant> &res);

typedef int ReadActuator;

typedef int WriteActuator;

typedef int ErrorActuator;

class TrayBrokerDBus : public TrayBroker {
	GDCLASS(TrayBrokerDBus, TrayBroker);

public:
	typedef void (*Callback)(DBusMessage *msg, void *p_userdata);

private:
	class Actuator {
	public:
		DBusConnection *bus;
		DBusWatch *watch;
		short flag;

		virtual dbus_bool_t handle() = 0;

		Actuator(DBusConnection *b, DBusWatch *w, short f) :
				bus(b), watch(w), flag(f) {}
		virtual ~Actuator() {}
	};

	class ReadActuator : public Actuator {
	public:
		virtual dbus_bool_t handle() override;

		ReadActuator(DBusConnection *b, DBusWatch *w, short f) :
				Actuator(b, w, f) {}
		virtual ~ReadActuator() {}
	};

	class WriteActuator : public Actuator {
	public:
		virtual dbus_bool_t handle() override;

		WriteActuator(DBusConnection *b, DBusWatch *w, short f) :
				Actuator(b, w, f) {}
		virtual ~WriteActuator() {}
	};

	struct WatchActuator {
		Optional<TrayBrokerDBus::ReadActuator> r;
		Optional<TrayBrokerDBus::WriteActuator> w;
	};

	struct SignalDBus {
		Callback cb;
		void *p_userdata;

		void emit(DBusMessage *msg);

		SignalDBus() :
				cb(nullptr), p_userdata(nullptr) {}

		SignalDBus(Callback c, void *p) :
				cb(c), p_userdata(p) {}
	};

private:
	DBusConnection *bus;
	DBusError error;
	HashMap<int, WatchActuator> acts;
	HashMap<String, SignalDBus> signals;

	static TrayBroker *_create_dbus();

	static dbus_bool_t dispatch(DBusConnection *bus);
	static void dispatch_status(DBusConnection *bus, DBusDispatchStatus status, void *user_data);
	static void send_dispatch(Actuator &act, short flag);

	static dbus_bool_t add_watch(DBusWatch *w, void *user_data);
	static void remove_watch(DBusWatch *w, void *user_data);
	static void toggle_watch(DBusWatch *w, void *user_data);

	static DBusHandlerResult signal_filter(DBusConnection *bus, DBusMessage *msg, void *user_data);

public:
	DBusConnection *get_bus();
	static void make_default();
	void process_messages();

	bool add_signal(const String &path, Callback cb, void *user_data);
	void remove_signal(const String &path);

	virtual void request_notification_permission() override;

	TrayBrokerDBus();
	~TrayBrokerDBus();
};

#endif // DBUS_ENABLED

#endif // TRAY_BROKER_DBUS_H
