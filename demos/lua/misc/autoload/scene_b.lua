local node = extends "Panel"

function node:_on_goto_scene_pressed()
	self:get_node("/root/global"):goto_scene("res://scene_a.xml")
	-- replace with function body
end
