local node = extends "RigidBody2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"

local STATE_WALKING = 0
local STATE_DYING = 1
local WALK_SPEED = 50

local bullet_class = "res://bullet.lua"

function node:_die()
	self:queue_free()
end

function node:_pre_explode()
	--stay there
	self:clear_shapes()
	self:set_mode(RigidBody2D.MODE_STATIC)
	self:get_node("sound"):play("explode")
end	

local function sign(n)
    if n > 1 then
        n = 1
    elseif n < -1 then
        n = -1
    end
    return n
end

function node:_integrate_forces(s)
	local lv = s:get_linear_velocity()
	local new_anim = self.anim

	if self.state == STATE_DYING then
		new_anim = "explode"
	elseif self.state == STATE_WALKING then
		new_anim="walk"

	    local wall_side = 0.0

        for i = 0, s:get_contact_count() - 1 do
			local cc = s:get_contact_collider_object(i)
			local dp = s:get_contact_local_normal(i)

			if cc then
                if cc:extends(bullet_class) and not cc.disabled then
					self:set_mode(RigidBody2D.MODE_RIGID)
					self.state = STATE_DYING
					--lv=s.get_contact_local_normal(i)*400
					s:set_angular_velocity(sign(dp.x) * 33.0)
					self:set_friction(true)
					cc:disable()
					self:get_node("sound"):play("hit")
					break
                end
            end

			if dp.x > 0.9 then
				wall_side = 1.0
			elseif dp.x<-0.9 then
				wall_side = -1.0
            end
        end

        if wall_side ~= 0 and wall_side ~= self.direction then
            self.direction = -self.direction
            self:get_node("sprite"):set_scale(Vector2(-self.direction, 1))			
        end
        if self.direction < 0 and not self.rc_left:is_colliding() and self.rc_right:is_colliding() then
            self.direction = -self.direction
            self:get_node("sprite"):set_scale(Vector2(-self.direction, 1))
        elseif self.direction > 0 and not self.rc_right:is_colliding() and self.rc_left:is_colliding() then
            self.direction = -self.direction
            self:get_node("sprite"):set_scale(Vector2(-self.direction, 1))
        end
        lv.x = self.direction * WALK_SPEED
    end

	if self.anim ~= new_anim then
		self.anim = new_anim
		self:get_node("anim"):play(self.anim)
    end
	s:set_linear_velocity(lv)
end

function node:_ready()
	-- Initalization here
	self.rc_left = self:get_node("raycast_left")
	self.rc_right = self:get_node("raycast_right")
    self.state = STATE_WALKING
    self.direction = -1
    self.anim = ""
end
