local node = extends "Area2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"

--var taken=false

function node:_on_body_enter(body)
	if not self.taken and body:extends("res://player.lua") then
		self:get_node("anim"):play("taken")
		taken = true
    end
end

function node:_ready()
	-- Initalization here
    self.taken = false
end
