local node = extends "KinematicBody2D"

-- This is a simple collision demo showing how
-- the kinematic cotroller works.
-- move() will allow to move the node, and will
-- always move it to a non-colliding spot, 
-- as long as it starts from a non-colliding spot too.


--pixels / second
local GRAVITY = 500.0

-- Angle in degrees towards either side that the player can 
-- consider "floor".
local FLOOR_ANGLE_TOLERANCE = 40
local WALK_FORCE = 600
local WALK_MAX_SPEED = 200
local STOP_FORCE = 1300
local JUMP_SPEED = 200
local JUMP_MAX_AIRBORNE_TIME = 0.2

local velocity = Vector2()
local on_air_time = 100
local jumping = false

local prev_jump_pressed = false

local function sign(n)
    if n > 1 then
        n = 1
    elseif n < -1 then
        n = -1
    end
    return n
end

function node:_fixed_process(delta)
	--create forces
	local force = Vector2(0, GRAVITY)

	local stop = velocity.x ~= 0.0
	
	local walk_left = Input:is_action_pressed("move_left")
	local walk_right = Input:is_action_pressed("move_right")
	local jump = Input:is_action_pressed("jump")

	local stop=true
	
	if walk_left then
		if velocity.x <= 0 and velocity.x > -WALK_MAX_SPEED then
			force.x = force.x - WALK_FORCE			
			stop = false
        end
    elseif walk_right then
		if velocity.x >= 0 and velocity.x < WALK_MAX_SPEED then
			force.x = force.x + WALK_FORCE
			stop = false
        end
    end

	if stop then
		local vsign = sign(velocity.x)
		local vlen = math.abs(velocity.x)
		
		vlen = vlen - (STOP_FORCE * delta)
		if vlen < 0 then
			vlen = 0
        end			
		velocity.x = vlen * vsign
    end

		
	--integrate forces to velocity
	velocity = velocity + (force * delta)
	
	--integrate velocity into motion and move
	local motion = velocity * delta

	--move and consume motion
	motion = self:move(motion)

	local floor_velocity = Vector2()

	if self:is_colliding() then
		--ran against something, is it the floor? get normal
		local n = self:get_collision_normal()

		if math.deg(math.acos(n:dot(Vector2(0,-1)))) < FLOOR_ANGLE_TOLERANCE then
			--if angle to the "up" vectors is < angle tolerance
			--char is on floor
			on_air_time = 0
			floor_velocity = self:get_collider_velocity()
			--velocity.y = 0 
        end
		-- But we were moving and our motion was interrupted, 
		-- so try to complete the motion by "sliding"
		-- by the normal
		motion = n:slide(motion)
		velocity = n:slide(velocity)
		
		--then move again
		self:move(motion)
    end

	if floor_velocity~=Vector2() then
		--if floor moves, move with floor
		self:move(floor_velocity * delta)
    end
	if jumping and velocity.y > 0 then
		jumping=false
    end
	if on_air_time < JUMP_MAX_AIRBORNE_TIME and jump and not prev_jump_pressed and not jumping then
		velocity.y = velocity.y - JUMP_SPEED
		jumping=true
    end
	on_air_time = on_air_time + delta
	prev_jump_pressed = jump	
end

function node:_ready()
	-- Initalization here
	self:set_fixed_process(true)
end
