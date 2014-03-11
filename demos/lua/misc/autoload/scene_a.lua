local node = extends "Panel"

-- member variables here, example:
-- var a=2
-- var b="textvar"

function node:_ready()
	-- Initalization here
    --print(self, '_ready')
end

function node:_on_goto_scene_pressed()
	self:get_node("/root/global"):goto_scene("res://scene_b.scn")
	-- replace with function body
end

function node:_on_goto_scene_toggled( pressed )
	-- replace with function body
	print(pressed)
end

function node:_on_anim_trigered(path)
    print('terigered', path, self:get_node(path))
	self:get_node("node/btn"):set_text("trigered")
end

