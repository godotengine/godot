-- reads luac listings and reports global variable usage
-- lines where a global is written to are marked with "*"
-- typical usage: luac -p -l file.lua | lua globals.lua | sort | lua table.lua

while 1 do
 local s=io.read()
 if s==nil then break end
 local ok,_,l,op,g=string.find(s,"%[%-?(%d*)%]%s*([GS])ETGLOBAL.-;%s+(.*)$")
 if ok then
  if op=="S" then op="*" else op="" end
  io.write(g,"\t",l,op,"\n")
 end
end
