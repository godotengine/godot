-- bare-bones luac in Lua
-- usage: lua luac.lua file.lua

assert(arg[1]~=nil and arg[2]==nil,"usage: lua luac.lua file.lua")
f=assert(io.open("luac.out","wb"))
assert(f:write(string.dump(assert(loadfile(arg[1])))))
assert(f:close())
