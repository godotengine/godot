local node = extends "Node2D"

-- member variables here, example:
-- var a=2
-- var b="textvar"
local INITIAL_BALL_SPEED = 80
local ball_speed = INITIAL_BALL_SPEED
local screen_size = Vector2(640,400)

-- default ball direction
local direction = Vector2(-1,0)
local pad_size = Vector2(8,32)
local PAD_SPEED = 150

local function randf()
    return math.random(0, 2147483647) / 2147483648
end

function node:_process(delta)
    -- get ball positio and pad rectangles
    local ball_pos = self:get_node("ball"):get_pos()
    local left_rect = Rect2(self:get_node("left"):get_pos() - pad_size * 0.5, pad_size)
    local right_rect = Rect2(self:get_node("right"):get_pos() - pad_size * 0.5, pad_size)
    
    -- integrate new ball postion
    ball_pos = ball_pos + (direction * ball_speed * delta)
    
    -- flip when touching roof or floor
    if (ball_pos.y < 0 and direction.y < 0) or (ball_pos.y > screen_size.y and direction.y > 0) then
        direction.y = -direction.y
    end
        
    -- flip, change direction and increase speed when touching pads    
    if (left_rect:has_point(ball_pos) and direction.x < 0) or (right_rect:has_point(ball_pos) and direction.x > 0) then
        direction.x = -direction.x
        ball_speed = ball_speed * 1.1
        direction.y = randf() * 2.0 - 1
        direction = direction:normalized()
    end

    -- check gameover
    if ball_pos.x < 0 or ball_pos.x > screen_size.x then
        ball_pos = screen_size * 0.5
        ball_speed = INITIAL_BALL_SPEED
        direction = Vector2(-1,0)
    end
                        
    self:get_node("ball"):set_pos(ball_pos)

    -- move left pad    
    local left_pos = self:get_node("left"):get_pos()
    if left_pos.y > 0 and Input:is_action_pressed("left_move_up") then
        left_pos.y = left_pos.y + -PAD_SPEED * delta
    end
    if left_pos.y < screen_size.y and Input:is_action_pressed("left_move_down") then
        left_pos.y = left_pos.y + PAD_SPEED * delta
    end
    self:get_node("left"):set_pos(left_pos)
        
    -- move right pad    
    local right_pos = self:get_node("right"):get_pos()
    if right_pos.y > 0 and Input:is_action_pressed("right_move_up") then
        right_pos.y = right_pos.y + -PAD_SPEED * delta
    end
    if right_pos.y < screen_size.y and Input:is_action_pressed("right_move_down") then
        right_pos.y = right_pos.y + PAD_SPEED * delta
    end
    self:get_node("right"):set_pos(right_pos)
end

function node:_ready()
    screen_size = self:get_viewport_rect().size -- get actual size
    pad_size = self:get_node("left"):get_texture():get_size()
    self:set_process(true)
end
