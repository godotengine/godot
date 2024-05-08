#include "luaCoroutine.h"

#include "luaAPI.h"
#include "luaTuple.h"

#ifdef LAPI_GDEXTENSION
#include <godot_cpp/classes/file_access.hpp>
#endif

void LuaCoroutine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind", "lua"), &LuaCoroutine::bind);
	ClassDB::bind_method(D_METHOD("set_hook", "Hook", "HookMask", "Count"), &LuaCoroutine::setHook);
	ClassDB::bind_method(D_METHOD("resume", "Args"), &LuaCoroutine::resume);
	ClassDB::bind_method(D_METHOD("yield_await", "Args"), &LuaCoroutine::yieldAwait);
	ClassDB::bind_method(D_METHOD("yield_state", "Args"), &LuaCoroutine::yield);

	ClassDB::bind_method(D_METHOD("load_string", "Code"), &LuaCoroutine::loadString);
	ClassDB::bind_method(D_METHOD("load_file", "FilePath"), &LuaCoroutine::loadFile);
	ClassDB::bind_method(D_METHOD("is_done"), &LuaCoroutine::isDone);

	ClassDB::bind_method(D_METHOD("call_function", "LuaFunctionName", "Args"), &LuaCoroutine::callFunction);
	ClassDB::bind_method(D_METHOD("function_exists", "LuaFunctionName"), &LuaCoroutine::luaFunctionExists);
	ClassDB::bind_method(D_METHOD("push_variant", "Name", "var"), &LuaCoroutine::pushGlobalVariant);
	ClassDB::bind_method(D_METHOD("pull_variant", "Name"), &LuaCoroutine::pullVariant);
	ClassDB::bind_method(D_METHOD("get_registry_value", "Name"), &LuaCoroutine::getRegistryValue);
	ClassDB::bind_method(D_METHOD("set_registry_value", "Name", "var"), &LuaCoroutine::setRegistryValue);

	// This signal is only meant to be used by await when yield_await is called.
	ADD_SIGNAL(MethodInfo("coroutine_resume"));
}

// binds the thread to a lua object
void LuaCoroutine::bind(LuaAPI*lua) {
	parent = lua;
	tState = lua->newThreadState();
	state.setState(tState, lua, false);

	// register the yield method
	lua_register(tState, "yield", luaYield);
}

// binds the thread to a lua object
void LuaCoroutine::bindExisting(LuaAPI*lua, lua_State *L) {
	done = false;
	parent = lua;
	this->tState = L;
	state.setState(tState, lua, false);

	// register the yield method
	lua_register(tState, "yield", luaYield);
}

void LuaCoroutine::setHook(Callable hook, int mask, int count) {
	return state.setHook(hook, mask, count);
}

Signal LuaCoroutine::yieldAwait(Array args) {
	lua_pop(tState, 1); // Pop function off top of stack. 
	for (int i = 0; i < args.size(); i++) {
		Ref<LuaError> err = state.pushVariant(args[i]);
		if (!err.is_null()) {
			// Raise an error on the lua state, error should be passed to reumse()
			state.pushVariant(err);
		}
	}
	return Signal(this, "coroutine_resume");
}

// Calls LuaState::luaFunctionExists()
bool LuaCoroutine::luaFunctionExists(String functionName) {
	return state.luaFunctionExists(functionName);
}

// Calls LuaState::pullVariant()
Variant LuaCoroutine::pullVariant(String name) {
	return state.pullVariant(name);
}

// Calls LuaState::pushGlobalVariant()
Ref<LuaError> LuaCoroutine::pushGlobalVariant(String name, Variant var) {
	return state.pushGlobalVariant(name, var);
}

// Calls LuaState::callFunction()
Variant LuaCoroutine::callFunction(String functionName, Array args) {
	return state.callFunction(functionName, args);
}

// Calls LuaState::getRegistryValue()
Variant LuaCoroutine::getRegistryValue(String name) {
	return state.getRegistryValue(name);
}

// Calls LuaState::setRegistryValue()
Ref<LuaError> LuaCoroutine::setRegistryValue(String name, Variant var) {
	return state.setRegistryValue(name, var);
}

// loads a string into the threads state
Ref<LuaError> LuaCoroutine::loadString(String code) {
	done = false;
	int ret = luaL_loadstring(tState, code.utf8().get_data());
	if (ret != LUA_OK) {
		return state.handleError(ret);
	}
	return nullptr;
}

Ref<LuaError> LuaCoroutine::loadFile(String fileName) {
#ifndef LAPI_GDEXTENSION
	done = false;
	Error error;
	Ref<FileAccess> file = FileAccess::open(fileName, FileAccess::READ, &error);
	if (error != Error::OK) {
		return LuaError::newError(vformat("error '%s' while opening file '%s'", error_names[error], fileName), LuaError::ERR_FILE);
	}
#else
	done = false;
	Ref<FileAccess> file = FileAccess::open(fileName, FileAccess::READ);
	if (!file.is_valid()) {
		return LuaError::newError(vformat("error while opening file '%s'", fileName), LuaError::ERR_FILE);
	}
#endif

	String path = file->get_path_absolute();
	int ret = luaL_loadfile(tState, path.utf8().get_data());
	if (ret != LUA_OK) {
		return state.handleError(ret);
	}
	return nullptr;
}

Ref<LuaError> LuaCoroutine::yield(Array args) {
	Array ret;
	if (int count = lua_gettop(tState); count > 0) {
		lua_pop(tState, count);
	}
	for (int i = 0; i < args.size(); i++) {
		Ref<LuaError> err = state.pushVariant(args[i]);
		if (!err.is_null()) {
			return err;
		}
	}

	lua_yield(tState, args.size());
	return nullptr;
}

#ifndef LAPI_GDEXTENSION

Variant LuaCoroutine::resume(Array args) {
	if (done) {
		return LuaError::newError("Thread is done executing", LuaError::ERR_RUNTIME);
	}

	List<Connection> resume_connections;
	get_signal_connection_list("coroutine_resume", &resume_connections);

	if (resume_connections.size() > 0) {
		if (resume_connections.size() != 1) {
			return LuaError::newError("Cannot have more than one connection to the coroutine_resume signal", LuaError::ERR_RUNTIME);
		}

		Callable callback = resume_connections.begin()->callable;
		if (!callback.is_valid()) {
			return LuaError::newError("Invalid callable connected to the coroutine_resume signal", LuaError::ERR_RUNTIME);
		}

		disconnect("coroutine_resume", callback);

		Vector<const Variant *> mem_args;
		mem_args.resize(args.size());
		for (int i = 0; i < args.size(); i++) {
			mem_args.write[i] = &args[i];
		}

		const Variant **p_args = (const Variant **)mem_args.ptr();

		Variant returned;
		Callable::CallError error;
		callback.callp(p_args, args.size(), returned, error);
		if (error.error != Callable::CallError::CALL_OK) {
			return state.handleError(callback.get_method(), error, p_args, args.size());
		}

		args.clear();
		args.append(returned);
	}

	for (int i = 0; i < args.size(); i++) {
		Ref<LuaError> err = state.pushVariant(args[i]);
		if (!err.is_null()) {
			return err;
		}
	}

#ifndef LAPI_LUAJIT
	int argc = 0;
	int ret = lua_resume(tState, nullptr, args.size(), &argc);
#else
	int ret = lua_resume(tState, args.size());
	int argc = lua_gettop(tState);
#endif

	if (ret == LUA_OK) {
		done = true; // thread is finished
	} else if (ret != LUA_YIELD) {
		done = true;
		return state.handleError(ret);
	}

	Array toReturn;
	for (int i = 1; i <= argc; i++) {
		toReturn.append(state.getVar(i));
	}

	return toReturn;
}

#else

Variant LuaCoroutine::resume(Array args) {
	if (done) {
		return LuaError::newError("Thread is done executing", LuaError::ERR_RUNTIME);
	}

	TypedArray<Dictionary> resume_connections = get_signal_connection_list("coroutine_resume");
	if (resume_connections.size() > 0) {
		if (resume_connections.size() != 1) {
			return LuaError::newError("Cannot have more than one connection to the coroutine_resume signal", LuaError::ERR_RUNTIME);
		}

		bool valid = false;
		Callable callback = resume_connections.pop_back().get("callable", &valid);
		if (!valid || !callback.is_valid()) {
			return LuaError::newError("Invalid callable connected to the coroutine_resume signal", LuaError::ERR_RUNTIME);
		}

		disconnect("coroutine_resume", callback);

		Variant returned = callback.callv(args);
		args.clear();
		args.append(returned);
	}

	for (int i = 0; i < args.size(); i++) {
		Ref<LuaError> err = state.pushVariant(args[i]);
		if (!err.is_null()) {
			return err;
		}
	}

#ifndef LAPI_LUAJIT
	int argc = 0;
	int ret = lua_resume(tState, nullptr, args.size(), &argc);
#else
	int ret = lua_resume(tState, args.size());
	int argc = lua_gettop(tState);
#endif

	if (ret == LUA_OK) {
		done = true; // thread is finished
	} else if (ret != LUA_YIELD) {
		done = true;
		return state.handleError(ret);
	}

	Array toReturn;
	for (int i = 1; i <= argc; i++) {
		toReturn.append(state.getVar(i));
	}

	return toReturn;
}

#endif

bool LuaCoroutine::isDone() {
	return done;
}

int LuaCoroutine::luaYield(lua_State *state) {
	int argc = lua_gettop(state);
	return lua_yield(state, argc);
}
