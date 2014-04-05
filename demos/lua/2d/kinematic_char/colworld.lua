local node = extends "Node2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"

function node:_ready()
	-- Initalization here
end

function node:_on_princess_body_enter(body)
	-- the name of this editor-generated callback is unfortunate
	self:get_node("youwin"):show()
end
