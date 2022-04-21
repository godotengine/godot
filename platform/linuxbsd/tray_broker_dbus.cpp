/*************************************************************************/
/*  tray_broker_dbus.cpp                                                 */
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

#include "tray_broker_dbus.h"

#ifdef DBUS_ENABLED

#include "core/templates/hash_map.h"
#include "core/variant/dictionary.h"

#include <dbus/dbus.h>

#include <poll.h>

Error _append_dbus_container(DBusMessageIter *iter, struct _dbus_msg_tuple *t) {
	Error err = OK;
	auto sub_iter = reinterpret_cast<DBusMessageIter *>(t->value);
	int32_t type = t->type;
	bool dict = false;

	if (type == DBUS_TYPE_DICT_ENTRY) {
		dict = true;
		type = DBUS_TYPE_ARRAY;
	}

	if (!dbus_message_iter_open_container(iter, type, t->signature, sub_iter)) {
		return ERR_OUT_OF_MEMORY;
	}

	if (t->sub) {
		dbus_bool_t container = dbus_type_is_container(t->sub->type);

		if (dict) {
			if (t->sub->type == DBUS_TYPE_INVALID) {
				dbus_message_iter_close_container(iter, sub_iter);
				return OK;
			}

			if (!(t->sub + 1)->value || (t->sub + 1)->type == DBUS_TYPE_INVALID) {
				dbus_message_iter_close_container(iter, sub_iter);
				return ERR_INVALID_PARAMETER;
			}

			container = dbus_type_is_container((t->sub + 1)->type);

			for (_dbus_msg_tuple *s = t->sub; s->value && s->type != DBUS_TYPE_INVALID; s += 2) {
				DBusMessageIter pair_iter;
				_dbus_msg_tuple *n = s + 1;

				if (!n->value || n->type == DBUS_TYPE_INVALID) {
					dbus_message_iter_close_container(iter, sub_iter);
					return ERR_INVALID_PARAMETER;
				}

				if (!dbus_message_iter_open_container(sub_iter, DBUS_TYPE_DICT_ENTRY, t->signature, &pair_iter)) {
					dbus_message_iter_close_container(iter, sub_iter);
					return ERR_OUT_OF_MEMORY;
				}

				if (!dbus_message_iter_append_basic(&pair_iter, s->type, s->value)) {
					dbus_message_iter_close_container(sub_iter, &pair_iter);
					dbus_message_iter_close_container(iter, sub_iter);
					return ERR_OUT_OF_MEMORY;
				}

				if (container) {
					auto val_iter = reinterpret_cast<DBusMessageIter *>(n->value);
					if (!dbus_message_iter_open_container(&pair_iter, n->type, n->signature, val_iter)) {
						dbus_message_iter_close_container(sub_iter, &pair_iter);
						dbus_message_iter_close_container(iter, sub_iter);
						return err;
					}
					err = _append_dbus_container(val_iter, n->sub);
					dbus_message_iter_close_container(&pair_iter, val_iter);
					if (err) {
						dbus_message_iter_close_container(sub_iter, &pair_iter);
						dbus_message_iter_close_container(iter, sub_iter);
						return err;
					}
				} else {
					if (!dbus_message_iter_append_basic(&pair_iter, n->type, n->value)) {
						dbus_message_iter_close_container(&pair_iter, iter);
						dbus_message_iter_close_container(iter, sub_iter);
						return ERR_OUT_OF_MEMORY;
					}
				}

				dbus_message_iter_close_container(&pair_iter, iter);
			}
		} else if (type == DBUS_TYPE_ARRAY) {
			for (_dbus_msg_tuple *s = t->sub; s->value; ++s) {
				if (container) {
					err = _append_dbus_container(sub_iter, t->sub);
					if (err) {
						return err;
					}
				} else {
					if (!dbus_message_iter_append_basic(iter, s->type, s->value)) {
						dbus_message_iter_close_container(iter, sub_iter);
						return ERR_OUT_OF_MEMORY;
					}
				}
			}
		} else if (type == DBUS_TYPE_VARIANT) {
			if (container) {
				err = _append_dbus_container(sub_iter, t->sub);
				if (err) {
					return err;
				}
			} else {
				if (!dbus_message_iter_append_basic(iter, t->type, t->value)) {
					dbus_message_iter_close_container(iter, sub_iter);
					return ERR_OUT_OF_MEMORY;
				}
			}
		} else {
			return ERR_INVALID_PARAMETER;
		}
	}

	dbus_message_iter_close_container(iter, sub_iter);

	return OK;
}

Error _append_dbus_msg_tuple(DBusMessageIter *iter, struct _dbus_msg_tuple *args) {
	for (_dbus_msg_tuple *t = args; t->value; ++t) {
		int type = t->type;
		if (t->category == ValueCategory::DBUS) {
			if (type == DBUS_TYPE_ARRAY) {
				if (t->sub && dbus_type_is_basic(t->sub->type)) {
					auto arr = reinterpret_cast<_dbus_fixed_arr *>(t->sub->value);
					dbus_message_iter_append_fixed_array(reinterpret_cast<DBusMessageIter *>(t->value), t->sub->type, arr->data, arr->len);
				} else {
					_append_dbus_container(iter, t);
				}
			} else if (dbus_type_is_container(t->type)) {
				_append_dbus_container(iter, t);
			} else if (!dbus_message_iter_append_basic(iter, type, t->value)) {
				return ERR_OUT_OF_MEMORY;
			}
		} else if (t->category == ValueCategory::GODOT) {
			// TODO: implement serialization of objects like images
			ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "serialization of Godot classes has yet to be implemented");
		}
	}

	return OK;
}

Error _unmarshall_dbus_basic(DBusMessageIter *iter, Variant &res) {
	int32_t type = dbus_message_iter_get_arg_type(iter);
	switch (type) {
		case DBUS_TYPE_OBJECT_PATH:
		case DBUS_TYPE_SIGNATURE:
		case DBUS_TYPE_STRING: {
			const char *str;
			dbus_message_iter_get_basic(iter, &str);
			res = String(str);
			break;
		}
		case DBUS_TYPE_BYTE: {
			uint8_t byte;
			dbus_message_iter_get_basic(iter, &byte);
			res = byte;
			break;
		}
		case DBUS_TYPE_BOOLEAN: {
			bool boolean;
			dbus_message_iter_get_basic(iter, &boolean);
			res = boolean;
			break;
		}
		case DBUS_TYPE_INT16: {
			int16_t num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		case DBUS_TYPE_UINT16: {
			uint16_t num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		case DBUS_TYPE_INT32: {
			int32_t num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		case DBUS_TYPE_UINT32: {
			uint32_t num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		case DBUS_TYPE_INT64: {
			int64_t num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		case DBUS_TYPE_UINT64: {
			uint64_t num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		case DBUS_TYPE_DOUBLE: {
			double num;
			dbus_message_iter_get_basic(iter, &num);
			res = num;
			break;
		}
		default: {
			return ERR_INVALID_PARAMETER;
		}
	}

	return OK;
}

Error _unmarshall_dbus_dictionary(DBusMessageIter *iter, Dictionary &res) {
	Error err = OK;
	DBusMessageIter pair_iter, var_iter, sub_iter;

	int32_t type = dbus_message_iter_get_arg_type(iter);
	while (type != DBUS_TYPE_INVALID) {
		const char *key;

		dbus_message_iter_recurse(iter, &pair_iter);

		if (dbus_message_iter_get_arg_type(&pair_iter) != DBUS_TYPE_STRING) {
			return ERR_INVALID_PARAMETER;
		}
		dbus_message_iter_get_basic(&pair_iter, &key);
		dbus_message_iter_next(&pair_iter);

		int32_t sub_type = dbus_message_iter_get_arg_type(&pair_iter);
		if (sub_type == DBUS_TYPE_INVALID) {
			res[String(key)] = Variant();
		} else if (sub_type == DBUS_TYPE_VARIANT) {
			dbus_message_iter_recurse(&pair_iter, &var_iter);
			int32_t var_type = dbus_message_iter_get_arg_type(&var_iter);
			if (var_type == DBUS_TYPE_INVALID) {
				res[String(key)] = Variant();
			} else if (dbus_type_is_container(var_type)) {
				Vector<Variant> sub_res;
				dbus_message_iter_recurse(&var_iter, &sub_iter);
				err = _unmarshall_dbus_container(&sub_iter, sub_res);
				res[String(key)] = sub_res;
			} else {
				Variant sub_res;
				err = _unmarshall_dbus_basic(&var_iter, sub_res);
				res[String(key)] = sub_res;
			}
		} else if (dbus_type_is_container(sub_type)) {
			Vector<Variant> sub_res;
			dbus_message_iter_recurse(&pair_iter, &sub_iter);
			err = _unmarshall_dbus_container(&sub_iter, sub_res);
			res[String(key)] = sub_res;
		} else {
			Variant sub_res;
			err = _unmarshall_dbus_basic(&sub_iter, sub_res);
			res[String(key)] = sub_res;
		}

		if (err) {
			break;
		}

		dbus_message_iter_next(iter);
		type = dbus_message_iter_get_arg_type(iter);
	}

	return err;
}

Error _unmarshall_dbus_container(DBusMessageIter *iter, Vector<Variant> &res) {
	Error err = OK;

	int32_t type = dbus_message_iter_get_arg_type(iter);
	while (type != DBUS_TYPE_INVALID) {
		switch (type) {
			case DBUS_TYPE_DICT_ENTRY: {
				// although unlikely to reach this case dict entries are only contained by arrays
				err = ERR_INVALID_PARAMETER;
				break;
			}
			case DBUS_TYPE_ARRAY: {
				DBusMessageIter sub_iter;
				dbus_message_iter_recurse(iter, &sub_iter);
				int32_t sub_type = dbus_message_iter_get_arg_type(&sub_iter);
				if (sub_type == DBUS_TYPE_INVALID) {
					res.append(Vector<Variant>());
				} else if (sub_type == DBUS_TYPE_DICT_ENTRY) {
					Dictionary sub;
					err = _unmarshall_dbus_dictionary(&sub_iter, sub);
					res.append(sub);
				} else if (dbus_type_is_container(sub_type)) {
					err = _unmarshall_dbus_container(&sub_iter, res);
				} else {
					Variant sub_res;
					err = _unmarshall_dbus_basic(&sub_iter, sub_res);
					res.append(sub_res);
				}
				break;
			}
			case DBUS_TYPE_VARIANT: {
				DBusMessageIter sub_iter;
				dbus_message_iter_recurse(iter, &sub_iter);
				int32_t sub_type = dbus_message_iter_get_arg_type(&sub_iter);
				if (sub_type == DBUS_TYPE_INVALID) {
					res.append(Variant());
				} else if (dbus_type_is_container(sub_type)) {
					err = _unmarshall_dbus_container(&sub_iter, res);
				} else {
					Variant sub_res;
					err = _unmarshall_dbus_basic(&sub_iter, sub_res);
					res.append(sub_res);
				}
				break;
			}
			case DBUS_TYPE_STRUCT: {
				DBusMessageIter sub_iter;
				Vector<Variant> sub;
				dbus_message_iter_recurse(iter, &sub_iter);
				err = _unmarshall_dbus_container(&sub_iter, sub);
				if (err) {
					return err;
				}
				res.append(sub);
				break;
			}
			default: {
				if (dbus_type_is_basic(type)) {
					Variant sub_res;
					err = _unmarshall_dbus_basic(iter, sub_res);
					res.append(sub_res);
				}
				break;
			}
		}

		if (err) {
			break;
		}

		dbus_message_iter_next(iter);
		type = dbus_message_iter_get_arg_type(iter);
	}

	return err;
}

dbus_bool_t TrayBrokerDBus::ReadActuator::handle() {
	if (!dbus_watch_handle(watch, DBUS_WATCH_READABLE)) {
		return 0;
	}

	return TrayBrokerDBus::dispatch(bus);
}

dbus_bool_t TrayBrokerDBus::WriteActuator::handle() {
	return dbus_watch_handle(watch, DBUS_WATCH_WRITABLE);
}

void TrayBrokerDBus::SignalDBus::emit(DBusMessage *msg) {
	cb(msg, p_userdata);
}

dbus_bool_t TrayBrokerDBus::dispatch(DBusConnection *bus) {
	DBusDispatchStatus status = dbus_connection_get_dispatch_status(bus);

	while (status == DBUS_DISPATCH_DATA_REMAINS) {
		status = dbus_connection_dispatch(bus);
	}

	if (status == DBUS_DISPATCH_NEED_MEMORY) {
		return 0;
	}

	return 1;
}

void TrayBrokerDBus::dispatch_status(DBusConnection *bus, DBusDispatchStatus status, void *user_data) {
	if (status == DBUS_DISPATCH_DATA_REMAINS) {
		dispatch(bus);
	}
}

void TrayBrokerDBus::send_dispatch(Actuator &act, short flag) {
	act.handle();
}

dbus_bool_t TrayBrokerDBus::add_watch(DBusWatch *w, void *user_data) {
	auto tb = reinterpret_cast<TrayBrokerDBus *>(user_data);
	uint32_t flags = dbus_watch_get_flags(w);
	int32_t fd = dbus_watch_get_unix_fd(w);

	WatchActuator *act = tb->acts.getptr(fd);
	if (!act) {
		act = &tb->acts.set(fd, WatchActuator{ nullopt, nullopt })->value();
	}
	if (flags & DBUS_WATCH_READABLE) {
		act->r = ReadActuator(tb->bus, w, POLLIN);
	}
	if (flags & DBUS_WATCH_WRITABLE) {
		act->w = WriteActuator(tb->bus, w, POLLOUT);
	}

	return 1;
}

void TrayBrokerDBus::remove_watch(DBusWatch *w, void *p_userdata) {
	auto tb = reinterpret_cast<TrayBrokerDBus *>(p_userdata);
	uint32_t flags = dbus_watch_get_flags(w);
	int32_t fd = dbus_watch_get_unix_fd(w);

	WatchActuator &act = tb->acts[fd];
	if (flags & DBUS_WATCH_READABLE) {
		if (!act.w) {
			tb->acts.erase(fd);
		} else {
			act.r = nullopt;
		}
	}
	if (flags & DBUS_WATCH_WRITABLE) {
		if (!act.r) {
			tb->acts.erase(fd);
		} else {
			act.w = nullopt;
		}
	}
}

void TrayBrokerDBus::toggle_watch(DBusWatch *w, void *user_data) {
	if (dbus_watch_get_enabled(w)) {
		add_watch(w, user_data);
	} else {
		remove_watch(w, user_data);
	}
}

DBusHandlerResult TrayBrokerDBus::signal_filter(DBusConnection *bus, DBusMessage *msg, void *p_userdata) {
	auto tb = reinterpret_cast<TrayBrokerDBus *>(p_userdata);
	const char *path = dbus_message_get_path(msg);

	switch (dbus_message_get_type(msg)) {
		case DBUS_MESSAGE_TYPE_SIGNAL: {
			auto signal = tb->signals.getptr(path);
			if (signal) {
				signal->emit(msg);
				return DBUS_HANDLER_RESULT_HANDLED;
			}
			break;
		}
			// TODO: Handle method calls for things like the tray icon
	}

	return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

void TrayBrokerDBus::make_default() {
	_create = _create_dbus;
}

TrayBroker *TrayBrokerDBus::_create_dbus() {
	return memnew(TrayBrokerDBus);
}

bool TrayBrokerDBus::add_signal(const String &path, Callback cb, void *p_userdata) {
	return signals.set(path, SignalDBus(cb, p_userdata));
}

void TrayBrokerDBus::remove_signal(const String &path) {
	signals.erase(path);
}

void TrayBrokerDBus::request_notification_permission() {
}

void TrayBrokerDBus::process_messages() {
	if (!acts.is_empty()) {
		int ret;
		struct timespec to = { 0, 10000 };

		List<int> act_l;
		acts.get_key_list(&act_l);

		size_t nfds = 0;
		for (List<int>::Element *a = act_l.front(); a; a = a->next()) {
			int fd = a->get();
			if (fd < 0) {
				continue;
			}
			auto act = acts[fd];
			if (act.r) {
				++nfds;
			}
			if (act.w) {
				++nfds;
			}
		}

		Vector<pollfd> pfds;
		pfds.resize(nfds);
		for (List<int>::Element *a = act_l.front(); a; a = a->next()) {
			int fd = a->get();
			if (fd < 0) {
				continue;
			}
			const WatchActuator &act = acts[fd];
			if (act.r) {
				pollfd pfd = { fd, act.r->flag, 0 };
				pfds.push_back(pfd);
			}
			if (act.w) {
				pollfd pfd = { fd, act.w->flag, 0 };
				pfds.push_back(pfd);
			}
		}

		ret = ppoll(pfds.ptrw(), pfds.size(), &to, NULL);

		if (ret > 0) {
			for (int f = 0; f < pfds.size(); ++f) {
				const struct pollfd &pfd = pfds[f];
				if (pfd.revents == 0 || pfd.revents & POLLNVAL) {
					continue;
				}
				auto it = acts.getptr(pfd.fd);
				if (!it) {
					continue;
				}
				if (pfd.revents & POLLIN) {
					send_dispatch(*it->r, pfd.revents);
				}
				if (pfd.revents & POLLOUT) {
					send_dispatch(*it->w, pfd.revents);
				}
			}
		}
	}
}

TrayBrokerDBus::TrayBrokerDBus() {
	dbus_error_init(&error);

	bus = dbus_bus_get(DBUS_BUS_SESSION, &error);
	if (!bus) {
		ERR_FAIL_MSG("Couldn't create DBus connection");
	}
	if (dbus_error_is_set(&error)) {
		ERR_FAIL_MSG(String("Getting DBus bus returned ") + error.message);
	}

	dbus_connection_set_dispatch_status_function(bus, dispatch_status, this, nullptr);

	dbus_connection_set_watch_functions(bus, add_watch, remove_watch, toggle_watch, this, nullptr);

	dbus_connection_add_filter(bus, signal_filter, this, nullptr);
}

DBusConnection *TrayBrokerDBus::get_bus() {
	return bus;
}

TrayBrokerDBus::~TrayBrokerDBus() {
	if (bus) {
		dbus_connection_flush(bus);
		dbus_connection_unref(bus);
	}
}

#endif // DBUS_ENABLED
