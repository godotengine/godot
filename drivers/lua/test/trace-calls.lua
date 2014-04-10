-- trace calls
-- example: lua -ltrace-calls bisect.lua

local level=0

local function hook(event)
 local t=debug.getinfo(3)
 io.write(level," >>> ",string.rep(" ",level))
 if t~=nil and t.currentline>=0 then io.write(t.short_src,":",t.currentline," ") end
 t=debug.getinfo(2)
 if event=="call" then
  level=level+1
 else
  level=level-1 if level<0 then level=0 end
 end
 if t.what=="main" then
  if event=="call" then
   io.write("begin ",t.short_src)
  else
   io.write("end ",t.short_src)
  end
 elseif t.what=="Lua" then
-- table.foreach(t,print)
  io.write(event," ",t.name or "(Lua)"," <",t.linedefined,":",t.short_src,">")
 else
 io.write(event," ",t.name or "(C)"," [",t.what,"] ")
 end
 io.write("\n")
end

debug.sethook(hook,"cr")
level=0
