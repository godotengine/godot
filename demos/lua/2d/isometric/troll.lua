local node = extends "KinematicBody2D"

-- This is a simple collision demo showing how
-- the kinematic cotroller works.
-- move() will allow to move the node, and will
-- always move it to a non-colliding spot, 
-- as long as it starts from a non-colliding spot too.

--pixels / second
local MOTION_SPEED=160

function node:_fixed_process(delta)
--   local root = self:get_node("/root")
--   local screen = false
--   if Input:is_action_pressed("1") then
--      print("ScreenShot")
--      print(root)
--      root:queue_screen_capture()
--      screen = root:get_screen_capture()
--      screen:save_png("image.png")
--  end
	
	local motion = Vector2()
	
	if Input:is_action_pressed("move_up") then
		motion = motion+Vector2(0,-1)
	end
	if Input:is_action_pressed("move_bottom") then
		motion = motion+Vector2(0,1)
	end
	if Input:is_action_pressed("move_left") then
		motion = motion+Vector2(-1,0)
	end
	if Input:is_action_pressed("move_right") then
		motion = motion+Vector2(1,0)
	end
	
	motion = motion:normalized() * MOTION_SPEED * delta
	motion = self:move(motion)
	
	--make character slide nicely through the world	
	local slide_attempts = 4
	if (self:is_colliding() and slide_attempts>0) then
		motion = self:get_collision_normal():slide(motion)
		motion = self:move(motion)
		slide_attempts = slide_attempts-1
	end
end

function node:_ready()
    -- Initalization here
    self:set_fixed_process(true)
end
