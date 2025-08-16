----------------------------------------------------------------------------
-- LuaJIT profiler zones.
--
-- Copyright (C) 2005-2023 Mike Pall. All rights reserved.
-- Released under the MIT license. See Copyright Notice in luajit.h
----------------------------------------------------------------------------
--
-- This module implements a simple hierarchical zone model.
--
-- Example usage:
--
--   local zone = require("jit.zone")
--   zone("AI")
--   ...
--     zone("A*")
--     ...
--     print(zone:get()) --> "A*"
--     ...
--     zone()
--   ...
--   print(zone:get()) --> "AI"
--   ...
--   zone()
--
----------------------------------------------------------------------------

local remove = table.remove

return setmetatable({
  flush = function(t)
    for i=#t,1,-1 do t[i] = nil end
  end,
  get = function(t)
    return t[#t]
  end
}, {
  __call = function(t, zone)
    if zone then
      t[#t+1] = zone
    else
      return (assert(remove(t), "empty zone stack"))
    end
  end
})

