-- read environment variables as if they were global variables

local f=function (t,i) return os.getenv(i) end
setmetatable(getfenv(),{__index=f})

-- an example
print(a,USER,PATH)
