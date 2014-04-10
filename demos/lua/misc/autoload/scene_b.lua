local node = extends "Panel"

-- member variables here, example:
-- var a=2
-- var b="textvar"

function node:_ready()
	-- Initalization here
    --print(self, '_ready')
end

function node:_on_goto_scene_pressed()
	self:get_node("/root/global"):goto_scene("res://scene_a.scn")
	-- replace with function body
end
