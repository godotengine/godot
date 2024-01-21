#include "luaAPI.h"

#include "luaCoroutine.h"
#include "luaObjectMetatable.h"

#include <luaState.h>
#include <lua/lua.hpp>
#include "luaError.h"


#ifdef LAPI_GDEXTENSION
#include <godot_cpp/classes/file_access.hpp>
#endif
LuaAPI* LuaAPI::singleton = nullptr;
LuaAPI::LuaAPI() {
	// lState = lua_newstate(&LuaAPI::luaAlloc, (void *)&luaAllocData);
	// // 设置为多线程模式
	// //luaJIT_setmode(lState, 0, LUAJIT_MODE_ENGINE | LUAJIT_MODE_ON );
	// Ref<LuaDefaultObjectMetatable> mt;
	// mt.instantiate();
	// objectMetatable = mt;

	// // Creating lua state instance
	// state->setState(lState, this, true);

	state = memnew(LuaState);
	singleton = this;
}

LuaAPI::~LuaAPI() {
	close_lua_state();
	singleton = nullptr;
	if(state != nullptr)
	{
		memdelete(state);
	}
	
}

LuaAPI* LuaAPI::get_singleton()
{
	return singleton;
}
void LuaAPI::create_lua_state()
{
	if(state == nullptr)
	{
		state = memnew(LuaState);
	}
	if(lState == nullptr)
	{
		lState = lua_newstate(&LuaAPI::luaAlloc, (void *)&luaAllocData);
		// 设置为多线程模式
		if(is_jit)
		{
			luaJIT_setmode(lState, 0, LUAJIT_MODE_ENGINE | LUAJIT_MODE_ON);
		}
		else
		{
			luaJIT_setmode(lState, 0, LUAJIT_MODE_ENGINE | LUAJIT_MODE_OFF);
		}
		Ref<LuaDefaultObjectMetatable> mt;
		mt.instantiate();
		objectMetatable = mt;
		// Creating lua state instance
		state->setState(lState, this, true);
		if(FileAccess::exists(startLuaFile))
		{
			doFile(startLuaFile);
		}
		++version;
	}

}
void LuaAPI::close_lua_state()
{
	if(lState != nullptr)
	{
		lua_close(lState);
		lState = nullptr;
		state->setState(lState, this, true);		
	}
}
// Bind C++ functions to GDScript
void LuaAPI::_bind_methods() {
	

	ClassDB::bind_method(D_METHOD("do_file", "FilePath"), &LuaAPI::doFile);
	ClassDB::bind_method(D_METHOD("do_string", "Code"), &LuaAPI::doString);

	ClassDB::bind_method(D_METHOD("add_search_path", "Path"), &LuaAPI::add_search_path);
	ClassDB::bind_method(D_METHOD("get_search_paths"), &LuaAPI::get_search_paths);
	ClassDB::bind_method(D_METHOD("set_jit", "value"), &LuaAPI::set_jit);

	ClassDB::bind_method(D_METHOD("bind_libraries", "Array"), &LuaAPI::bindLibraries);
	ClassDB::bind_method(D_METHOD("set_hook", "Hook", "HookMask", "Count"), &LuaAPI::setHook);
	ClassDB::bind_method(D_METHOD("configure_gc", "What", "Data"), &LuaAPI::configureGC);
	ClassDB::bind_method(D_METHOD("get_memory_usage"), &LuaAPI::getMemoryUsage);
	ClassDB::bind_method(D_METHOD("push_variant", "Name", "var"), &LuaAPI::pushGlobalVariant);
	ClassDB::bind_method(D_METHOD("pull_variant", "Name"), &LuaAPI::pullVariant);
	ClassDB::bind_method(D_METHOD("get_registry_value", "Name"), &LuaAPI::getRegistryValue);
	ClassDB::bind_method(D_METHOD("set_registry_value", "Name", "var"), &LuaAPI::setRegistryValue);
	ClassDB::bind_method(D_METHOD("call_function", "LuaFunctionName", "Args"), &LuaAPI::callFunction);
	ClassDB::bind_method(D_METHOD("function_exists", "LuaFunctionName"), &LuaAPI::luaFunctionExists);

	ClassDB::bind_method(D_METHOD("new_coroutine"), &LuaAPI::newCoroutine);
	ClassDB::bind_method(D_METHOD("get_running_coroutine"), &LuaAPI::getRunningCoroutine);

	ClassDB::bind_method(D_METHOD("set_use_callables", "value"), &LuaAPI::setUseCallables);
	ClassDB::bind_method(D_METHOD("get_use_callables"), &LuaAPI::getUseCallables);

	ClassDB::bind_method(D_METHOD("set_object_metatable", "value"), &LuaAPI::setObjectMetatable);
	ClassDB::bind_method(D_METHOD("get_object_metatable"), &LuaAPI::getObjectMetatable);

	ClassDB::bind_method(D_METHOD("set_memory_limit", "limit"), &LuaAPI::setMemoryLimit);
	ClassDB::bind_method(D_METHOD("get_memory_limit"), &LuaAPI::getMemoryLimit);

	ClassDB::bind_method(D_METHOD("reload_lua_state"), &LuaAPI::reload_lua_state);
	ClassDB::bind_method(D_METHOD("close_lua_state"), &LuaAPI::close_lua_state);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_callables"), "set_use_callables", "get_use_callables");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "object_metatable"), "set_object_metatable", "get_object_metatable");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "memory_limit"), "set_memory_limit", "get_memory_limit");

	// BIND_ENUM_CONSTANT(HOOK_MASK_CALL);
	// BIND_ENUM_CONSTANT(HOOK_MASK_RETURN);
	// BIND_ENUM_CONSTANT(HOOK_MASK_LINE);
	// BIND_ENUM_CONSTANT(HOOK_MASK_COUNT);

	// BIND_ENUM_CONSTANT(GC_STOP);
	// BIND_ENUM_CONSTANT(GC_RESTART);
	// BIND_ENUM_CONSTANT(GC_COLLECT);
	// BIND_ENUM_CONSTANT(GC_COUNT);
	// BIND_ENUM_CONSTANT(GC_COUNTB);
	// BIND_ENUM_CONSTANT(GC_STEP);
	// BIND_ENUM_CONSTANT(GC_SETPAUSE);
	// BIND_ENUM_CONSTANT(GC_SETSTEPMUL);
}

// Calls LuaState::bindLibs()
Ref<LuaError> LuaAPI::bindLibraries(TypedArray<String> libs) {
	return state->bindLibraries(libs);
}

void LuaAPI::setHook(Callable hook, int mask, int count) {
	return state->setHook(hook, mask, count);
}

int LuaAPI::configureGC(int what, int data) {
	return lua_gc(lState, what, data);
}

void LuaAPI::setUseCallables(bool value) {
	useCallables = value;
}

bool LuaAPI::getUseCallables() const {
	return useCallables;
}

void LuaAPI::setObjectMetatable(Ref<LuaObjectMetatable> value) {
	objectMetatable = value;
}

Ref<LuaObjectMetatable> LuaAPI::getObjectMetatable() const {
	return objectMetatable;
}

void LuaAPI::setMemoryLimit(uint64_t limit) {
	luaAllocData.memoryLimit = limit;
}

uint64_t LuaAPI::getMemoryLimit() const {
	return luaAllocData.memoryLimit;
}

Variant LuaAPI::getRegistryValue(String name) {
	return state->getRegistryValue(name);
}

Ref<LuaError> LuaAPI::setRegistryValue(String name, Variant var) {
	return state->setRegistryValue(name, var);
}

uint64_t LuaAPI::getMemoryUsage() const {
	return luaAllocData.memoryUsed;
}

// Calls LuaState::luaFunctionExists()
bool LuaAPI::luaFunctionExists(String functionName) {
	return state->luaFunctionExists(functionName);
}

// Calls LuaState::pullVariant()
Variant LuaAPI::pullVariant(String name) {
	return state->pullVariant(name);
}

// Calls LuaState::callFunction()
Variant LuaAPI::callFunction(String functionName, Array args) {
	chack_load();
	return state->callFunction(functionName, args);
}
// 获取全局表中的函数
Ref<LuaError> LuaAPI::getTableFuncMap(String name, HashMap<StringName, Callable>& func_map,HashMap<StringName, Variant> & member_map)
{
	chack_load();
	return state->getTableFuncMap(name, func_map,member_map);
}

void LuaAPI::getLuaClassTable(String table_name,Object* _this)
{	
	chack_load();
	LuaClassTable* ct = memnew(LuaClassTable);
	ct->init(this,table_name,_this);
}

// Calls LuaState::pushGlobalVariant()
Ref<LuaError> LuaAPI::pushGlobalVariant(String name, Variant var) {
	return state->pushGlobalVariant(name, var);
}

// addFile() calls luaL_loadfille with the absolute file path
Ref<LuaError> LuaAPI::doFile(String fileName) {
	// push the error handler onto the stack
	lua_pushcfunction(lState, LuaState::luaErrorHandler);

	String path;
	// fileAccess never unrefs without this
	{
#ifndef LAPI_GDEXTENSION
		Error error;
		Ref<FileAccess> file = FileAccess::open(fileName, FileAccess::READ, &error);
		if (error != Error::OK) {
			String error_str = vformat("error '%s' while opening file '%s'", error_names[error], fileName);
			ERR_PRINT(error_str);
			return LuaError::newError(error_str, LuaError::ERR_FILE);
		}
#else
		Ref<FileAccess> file = FileAccess::open(fileName, FileAccess::READ);
		if (!file.is_valid()) {
			String error_str = vformat("error '%s' while opening file '%s'", error_names[error], fileName);
			ERR_PRINT(error_str);
			return LuaError::newError(error_str, LuaError::ERR_FILE);
		}
#endif


		Vector<uint8_t> data = file->get_buffer(file->get_length());	
		int ret = luaL_loadbuffer(lState, (const char*)data.ptr(), data.size(), fileName.utf8().get_data());
		if (ret != LUA_OK) {
			return state->handleError(ret);
		}

	}
	Ref<LuaError> err = execute(-2);
	// pop the error handler from the stack
	lua_pop(lState, 1);
	return err;
}

// Loads string into lua state and executes the top of the stack
Ref<LuaError> LuaAPI::doString(String code) {
	// push the error handler onto the stack
	lua_pushcfunction(lState, LuaState::luaErrorHandler);
	int ret = luaL_loadstring(lState, code.utf8().get_data());
	if (ret != LUA_OK) {
		return state->handleError(ret);
	}

	Ref<LuaError> err = execute(-2);
	// pop the error handler from the stack
	lua_pop(lState, 1);
	return err;
}

// Execute the current lua stack, return error as string if one occurs, otherwise return String()
Ref<LuaError> LuaAPI::execute(int handlerIndex) {
	int ret = lua_pcall(lState, 0, 0, handlerIndex);
	if (ret != LUA_OK) {
		return state->handleError(ret);
	}
	return nullptr;
}

Ref<LuaCoroutine> LuaAPI::newCoroutine() {
	Ref<LuaCoroutine> thread;
	thread.instantiate();
	thread->bind(this);
	return thread;
}

Ref<LuaCoroutine> LuaAPI::getRunningCoroutine() {
	Variant top = state->getVar();
	if (top.get_type() != Variant::Type::OBJECT) {
		return nullptr;
	}

#ifndef LAPI_GDEXTENSION
	Ref<LuaCoroutine> thread = Object::cast_to<LuaCoroutine>(top);
#else
	Ref<LuaCoroutine> thread = dynamic_cast<LuaCoroutine *>(top.operator Object *());
#endif
	return thread;
}

// Creates a new thread staee
lua_State *LuaAPI::newThreadState() {
	return lua_newthread(lState);
}

// returns state
lua_State *LuaAPI::getState() {
	return lState;
}

void *LuaAPI::luaAlloc(void *ud, void *ptr, size_t osize, size_t nsize) {
	LuaAllocData *data = (LuaAllocData *)ud;
	if (nsize == 0) {
		if (ptr != nullptr) {
			data->memoryUsed -= osize;
			memfree(ptr);
		}
		return nullptr;
	}

	if (ptr == nullptr) {
		if (data->memoryLimit != 0 && data->memoryUsed + (uint64_t)nsize > data->memoryLimit) {
			return nullptr;
		}

		data->memoryUsed += (uint64_t)nsize;
		return memalloc(nsize);
	}

	if (data->memoryLimit != 0 && data->memoryUsed - (uint64_t)osize + (uint64_t)nsize > data->memoryLimit) {
		return nullptr;
	}

	data->memoryUsed -= (uint64_t)osize;
	data->memoryUsed += (uint64_t)nsize;
	return memrealloc(ptr, nsize);
}





void LuaClassTable::init(LuaAPI *lua,StringName name,Object* _self)
{
	this->objectID = lua->get_instance_id();
	this->self.set_obj(_self);
	_self->set_master_script_instance(this);
	lua->getTableFuncMap(name, funcMap,propertyMap);

	if(funcMap.has("__init"))
	{
		Callable cb = funcMap["__init"];
		cb.call(_self);
	}
}


bool LuaClassTable::is_valid()const
{
	Object* lua_ptr = ObjectDB::get_instance(objectID);
	if(lua_ptr == nullptr)
	{
		return false;
	}
	LuaAPI *lua = Object::cast_to<LuaAPI>(lua_ptr);
	Object* self_ptr = self.get_ref();
	if(self_ptr == nullptr)
	{
		return false;
	}
	if(lua->is_load())
	{
		lua->chack_load();
		funcMap.clear();
		propertyMap.clear();
		lua->getTableFuncMap(table_name, funcMap,propertyMap);
		is_init = true;
	}
	else if(!is_init || version != lua->get_version())
	{
		is_init = true;
		LuaClassTable* _this = (LuaClassTable*)this;
		funcMap.clear();
		propertyMap.clear();
		lua->getTableFuncMap(table_name, funcMap,propertyMap);
	}
	return is_init;
}

Variant LuaClassTable::callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) 
{
	Object* self_ptr = self.get_ref();
	if(!is_valid() || self_ptr == nullptr)
	{
		return Variant();
	}
	const Variant** args = (const Variant**)alloca(p_argcount * sizeof(void *));
	Variant obj = self_ptr;
	args[0] = &obj;
	for(int i = 0;i < p_argcount;i++)
	{
		args[i + 1] = (Variant*)p_args[i];
	}
	if(funcMap.has(p_method))
	{
		Callable cb = funcMap[p_method];
		Variant rs;
			cb.callp(args, p_argcount + 1, rs,r_error);
			return rs;
	}
	return Variant();
}
Variant LuaClassTable::call_const(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) 
{
	Object* self_ptr = self.get_ref();
	if(!is_valid() || self_ptr == nullptr)
	{
		return Variant();
	}
	const Variant** args = (const Variant**)alloca(p_argcount * sizeof(void *));
	Variant obj = self_ptr;
	args[0] = &obj;
	for(int i = 0;i < p_argcount;i++)
	{
		args[i + 1] = p_args[i];
	}
	if(funcMap.has(p_method))
	{
		Callable cb = funcMap[p_method];
		Variant rs;
			cb.callp(args, p_argcount + 1, rs,r_error);
			return rs;
	}
	return Variant();
}

