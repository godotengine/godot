local node = extends "RigidBody2D"

-- Character Demo, written by Juan Linietsky.
--
-- Implementation of a 2D Character controller.
-- This implementation uses the physics engine for
-- controlling a character, in a very similar way
-- than a 3D character controller would be implemented.
--
-- Using the physics engine for this has the main
-- advantages:
-- -Easy to write.
-- -Interaction with other physics-based objects is free
-- -Only have to deal with the object linear velocity, not position
-- -All collision/area framework available
-- 
-- But also has the following disadvantages:
--  
-- -Objects may bounce a little bit sometimes
-- -Going up ramps sends the chracter flying up, small hack is needed.
-- -A ray collider is needed to avoid sliding down on ramps and  
--   undesiderd bumps, small steps and rare numerical precision errors.
--   (another alternative may be to turn on friction when the character is not moving).
-- -Friction cant be used, so floor velocity must be considered
--  for moving platforms.

local WALK_ACCEL = 800.0
local WALK_DEACCEL = 800.0
local WALK_MAX_VELOCITY = 200.0
local GRAVITY = 700.0
local AIR_ACCEL = 200.0
local AIR_DEACCEL = 200.0
local JUMP_VELOCITY = 460
local STOP_JUMP_FORCE = 900.0

local MAX_FLOOR_AIRBORNE_TIME = 0.15

local MAX_SHOOT_POSE_TIME = 0.3

local bullet = ResourceLoader:load("res://bullet.xml")

local enemy = ResourceLoader:load("res://enemy.xml")

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
	local step = s:get_step()
	
	local new_anim = self.anim
	local new_siding_left = self.siding_left
	
	-- Get the controls
	local move_left = Input:is_action_pressed("move_left")
	local move_right = Input:is_action_pressed("move_right")
	local jump = Input:is_action_pressed("jump")
	local shoot = Input:is_action_pressed("shoot")
	local spawn = Input:is_action_pressed("spawn")
	
	if spawn then
		local e = enemy:instance()
		local p = self:get_pos()
		p.y = p.y - 100
		e:set_pos(p)
		self:get_parent():add_child(e)
    end
	
	--deapply prev floor velocity
	lv.x = lv.x - self.floor_h_velocity
	self.floor_h_velocity = 0.0
	
	
	-- Find the floor (a contact with upwards facing collision normal)
	local found_floor = false
	local floor_index = -1
	
	for x = 0, s:get_contact_count() - 1 do
		local ci = s:get_contact_local_normal(x)
		if ci:dot(Vector2(0,-1)) > 0.6 then
			found_floor = true
			floor_index = x
        end
    end
	-- A good idea when impementing characters of all kinds,
	-- Compensates for physics imprecission, as well as human
	-- reaction delay.

	if shoot and not self.shooting then
		self.shoot_time = 0
		local bi = bullet:instance()
		local ss
		if self.siding_left then
			ss = -1.0
		else
			ss = 1.0
        end
		local pos = self:get_pos() + self:get_node("bullet_shoot"):get_pos() * Vector2(ss, 1.0)

		bi:set_pos(pos)
		self:get_parent():add_child(bi)

		bi:set_linear_velocity(Vector2(800.0 * ss, -80))	
		self:get_node("sprite/smoke"):set_emitting(true)	
		self:get_node("sound"):play("shoot")
		PS2D:body_add_collision_exception(bi:get_rid(), self:get_rid()) -- make bullet and this not collide
	else
		self.shoot_time = self.shoot_time + step
    end
	
	if found_floor then
		self.airborne_time = 0.0 
	else
		self.airborne_time = self.airborne_time + step --time it spent in the air
    end
		
	local on_floor = self.airborne_time < MAX_FLOOR_AIRBORNE_TIME

	-- Process jump		
	if self.jumping then
		if lv.y > 0 then
			--set off the self.jumping flag if going down
			self.jumping = false
		elseif not jump then
			self.stopping_jump = true
        end

		if self.stopping_jump then
			lv.y = lv.y + STOP_JUMP_FORCE * step
        end
    end
		
	if on_floor then
		-- Process logic when character is on floor
			
		if move_left and not move_right then
			if lv.x > -WALK_MAX_VELOCITY then
				lv.x = lv.x - WALK_ACCEL * step
            end
		elseif move_right and not move_left then
			if lv.x < WALK_MAX_VELOCITY then
				lv.x = lv.x + WALK_ACCEL * step
            end
		else
			local xv = math.abs(lv.x)
			xv = xv - WALK_DEACCEL * step
			if xv < 0 then
				xv = 0
            end
			lv.x = sign(lv.x) * xv
        end
		--Check jump
		if not self.jumping and jump then
			lv.y = -JUMP_VELOCITY
			self.jumping = true
			self.stopping_jump = false
			self:get_node("sound"):play("jump")
        end
		--check siding
		
		if lv.x < 0 and move_left then
			new_siding_left = true
		elseif lv.x > 0 and move_right then
			new_siding_left = false
        end
		if self.jumping then
			new_anim = "jumping"	
		elseif math.abs(lv.x) < 0.1 then
			if self.shoot_time < MAX_SHOOT_POSE_TIME then
				new_anim = "idle_weapon"
			else
				new_anim = "idle"
            end
		else
			if self.shoot_time < MAX_SHOOT_POSE_TIME then
				new_anim = "run_weapon"
			else
				new_anim = "run"
            end
        end
	else
		-- Process logic when the character is in the air
		if move_left and not move_right then
			if lv.x > -WALK_MAX_VELOCITY then
				lv.x = lv.x - AIR_ACCEL * step			
            end
		elseif move_right and not move_left then
			if lv.x < WALK_MAX_VELOCITY then
				lv.x = lv.x + AIR_ACCEL * step
            end
		else
			local xv = math.abs(lv.x)
			xv = xv - AIR_DEACCEL * step
			if xv < 0 then
				xv = 0
            end
			lv.x = sign(lv.x) * xv
        end
		if lv.y < 0 then
			if self.shoot_time < MAX_SHOOT_POSE_TIME then
				new_anim = "jumping_weapon"
			else
				new_anim = "jumping"
            end
		else
			if self.shoot_time < MAX_SHOOT_POSE_TIME then
				new_anim = "falling_weapon"
			else
				new_anim = "falling"
            end
        end
    end

	--Update siding
	
	if new_siding_left ~= self.siding_left then
		if new_siding_left then
			self:get_node("sprite"):set_scale(Vector2(-1, 1))
		else
			self:get_node("sprite"):set_scale(Vector2(1, 1))
        end
		self.siding_left = new_siding_left
    end
	--Change animation
	if new_anim ~= self.anim then
		self.anim = new_anim
		self:get_node("anim"):play(self.anim)
    end
	self.shooting = shoot

 	-- Apply floor velocity
	if found_floor then
		self.floor_h_velocity = s:get_contact_collider_velocity_at_pos(floor_index).x
		lv.x = lv.x + self.floor_h_velocity
    end

	--Finally, apply gravity and set back the linear velocity
	lv = lv + s:get_total_gravity() * step
	s:set_linear_velocity(lv)
end

function node:_ready()
--	if !Globals.has_singleton("Facebook"):
--        	return
--	local Facebook = Globals.get_singleton("Facebook")
--	local link = Globals.get("facebook/link")
--	local icon = Globals.get("facebook/icon")
--	local msg = "I just sneezed on your wall! Beat my score and Stop the Running nose!"
--	local title = "I just sneezed on your wall!"
--	Facebook.post("feed", msg, title, link, icon)

    -- Initalization here
    self.anim = ""
    self.siding_left = false
    self.jumping = false
    self.stopping_jump = false
    self.shooting = false
    self.floor_h_velocity = 0.0
    self.airborne_time = 1e20
    self.shoot_time = 1e20
end
