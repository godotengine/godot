import os

def code_gen(luaJIT=False):
    lua_libraries = [
        "base",
        "coroutine",
        "debug",
        "io",
        "math",
        "os",
        "package",
        "string",
        "table",
        "utf8"
    ]

    luajit_libraries = [
        "base",
        "bit",
        "debug",
        "ffi",
        "io",
        "jit",
        "math",
        "os",
        "package",
        "string",
        "string_buffer",
        "table"
    ]

    libraries = lua_libraries
    lib_source_files = []

    if luaJIT:
        libraries = luajit_libraries

    for library in os.listdir("./"):
        if not os.path.isdir(library) or library == "__pycache__" or library == "bin":
            continue
        
        libraries.append(library)

        for source_file in os.listdir("./%s" % library):
            if source_file.endswith(".cpp") or source_file.endswith(".c"):
                lib_source_files.append(os.path.join(library, source_file))

    luaLibraries_gen_cpp = "#include \"lua_libraries.h\"\n#include <map>\n#include <string>\n\n"
    
    if len(lib_source_files) > 0:
        for source_file in lib_source_files:
            luaLibraries_gen_cpp += "#include \"%s\"\n" % source_file
        luaLibraries_gen_cpp += "\n"
    
    luaLibraries_gen_cpp += "std::map<std::string, lua_CFunction> luaLibraries = {\n"

    for library in libraries:
        luaLibraries_gen_cpp += "\t{ \"%s\", luaopen_%s },\n" % (library, library)

    luaLibraries_gen_cpp += "};\n"
    if luaJIT:
        luaLibraries_gen_cpp += """
bool loadLuaLibrary(lua_State *L, String libraryName) {
	const char *lib_c_str = libraryName.ascii().get_data();
	if (luaLibraries[lib_c_str] == nullptr) {
		return false;
	}
	
    lua_pushcfunction(L, luaLibraries[lib_c_str]);
    if (libraryName == "base") {
        lua_pushstring(L, "");
    } else {
	    lua_pushstring(L, lib_c_str);
    }
	lua_call(L, 1, 0);
	return true;
}
"""
    else:
        luaLibraries_gen_cpp += """
bool loadLuaLibrary(lua_State *L, String libraryName) {
	const char *lib_c_str = libraryName.ascii().get_data();
	if (luaLibraries[lib_c_str] == nullptr) {
		return false;
	}

	luaL_requiref(L, lib_c_str, luaLibraries[lib_c_str], 1);
	lua_pop(L, 1);
	return true;
}
"""

    gen_file = open("lua_libraries.gen.cpp", "w")
    gen_file.write(luaLibraries_gen_cpp)
    gen_file.close()