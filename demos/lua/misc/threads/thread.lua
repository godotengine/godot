local node = extends "Node2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"

local thread = Thread:new()

--this function runs in a thread!
--threads always take one userdata argument
function node:_bg_load(path)
	print("THREAD FUNC!")
	--load the resource
	local tex = ResourceLoader:load(path)
	--call _bg_load_done on main thread	
	self:call_deferred("_bg_load_done")
	return tex --return it
end

function node:_bg_load_done()
	--wait for the thread to complete, get the returned value
    print("THREAD DONE")
	local tex = thread:wait_to_finish()
	--set to the sprite
	self:get_node("sprite"):set_texture(tex)
end

function node:_on_load_pressed()
	if thread:is_active() then
		--already working
		return
    end
	print("START THREAD!")
	thread:start(self, "_bg_load", "res://mona.png")
end
