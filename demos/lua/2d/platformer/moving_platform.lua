local node = extends "Node2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"

export("motion", "Vector2", Vector2())
export("cycle", "Real", 1.0)

function node:_fixed_process(delta)
	self.accum = self.accum + (delta * (1.0 / self.cycle) * PI * 2.0)
	accum = math.fmod(self.accum, PI * 2.0)
	local d = math.sin(self.accum)
	local xf = Matrix32()
	xf[2]= self.motion * d 
	self:get_node("platform"):set_transform(xf)
end

function node:_ready()
	-- Initalization here
	self:set_fixed_process(true)
    self.accum = 0.0
end



