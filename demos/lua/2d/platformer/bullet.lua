local node = extends "RigidBody2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"

--  in lua script, global scope's local variables is not a owned variables of object instance
--  declare owned variables is initialized in object's construct(_init or _ready) function
--local disabled = false

function node:_init()
    self.disabled = false
end

function node:disable()
	if self.disabled then
		return
    end
	self:get_node("anim"):play("shutdown")
	self.disabled = true
end

function node:_ready()
	-- Initalization here
	self:get_node("Timer"):start()
end
