#include <cstdio>
#include <cstring>
#include <string>
extern "C" {
	#include <lauxlib.h>
	#include <lua.h>
	#include <lualib.h>
}
static constexpr bool VERBOSE = false;
static lua_State *L;

static int api_print(lua_State *L) {
	const char *text = luaL_checkstring(L, 1);
	printf("[luajit] %s\n", text);
	fflush(stdout);
	return 0;
}
static int api_pause(lua_State *L) {
	if constexpr (VERBOSE) {
		printf("[luajit] Pausing VM...\n");
		fflush(stdout);
	}
	asm("wfi"); // Wait for interrupt
	if constexpr (VERBOSE) {
		printf("[luajit] Resuming VM...\n");
		fflush(stdout);
	}
	return 0;
}

static std::string result; // Global variable to hold the result
extern "C" __attribute__((used, retain))
const char *run(const char* code)
{
	if (code != nullptr) {
		// Load a string as a script
		if (luaL_loadbuffer(L, code, strlen(code), "@code") != LUA_OK) {
			fprintf(stderr, "Error loading Lua code: %s\n", lua_tostring(L, -1));
			return "(error)";
		}
	}

	// Run the script (0 arguments, 1 result)
	lua_pcall(L, 0, 1, 0);

	// Get the result type
	const int type = lua_type(L, -1);
	switch (type) {
		case LUA_TNIL:
			result = "(nil)";
			break;
		case LUA_TBOOLEAN:
			result = lua_toboolean(L, -1) ? "true" : "false";
			break;
		case LUA_TNUMBER:
			result = std::to_string(lua_tonumber(L, -1));
			break;
		case LUA_TSTRING:
			result = std::string(lua_tostring(L, -1));
			break;
		default:
			result = "(unknown)";
			break;
	}
	return result.c_str();
}
extern "C" __attribute__((used, retain))
int compile(const char* code)
{
	// Load a string as a script
	const int res = luaL_loadbuffer(L, code, strlen(code), "@code");
	if (res != LUA_OK) {
		fprintf(stderr, "Error loading Lua code: %s\n", lua_tostring(L, -1));
		return -1;
	}
	return 0;
}

int main()
{
	printf("LuaJIT WebAssembly Example\n");
	// Initialize LuaJIT
	L = luaL_newstate();
	if (!L) {
		fprintf(stderr, "Failed to create Lua state\n");
		return 1;
	}
	// Set the error handler
	lua_atpanic(L, [](lua_State *L) -> int {
		fprintf(stderr, "Lua panic: %s\n", lua_tostring(L, -1));
		return 0; // Return 0 to avoid further panic
	});
	luaL_openlibs(L); /* Load Lua libraries */
	if constexpr (VERBOSE) {
		printf("[luajit] Lua state initialized\n");
		fflush(stdout);
	}

	// API bindings
	lua_register(L, "print", api_print);
	lua_register(L, "pause", api_pause);

	asm("wfi"); // Pause the VM
}
