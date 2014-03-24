local node = extends "Control"

-- Simple Tetris-like demo, (c) 2012 Juan Linietsky
-- Implemented by using a regular Control and drawing on it during the _draw() callback.
-- The drawing surface is updated only when changes happen (by calling update())


local score = 0
local score_label

local MAX_SHAPES = 7

local block = ResourceLoader:load("res://block.png")

local block_colors = {
    Color(1,0.5,0.5),
    Color(0.5,1,0.5),
    Color(0.5,0.5,1),
    Color(0.8,0.4,0.8),
    Color(0.8,0.8,0.4),
    Color(0.4,0.8,0.8),
    Color(0.7,0.7,0.7)
}

local block_shapes = {
    { Vector2(0,-1),Vector2(0,0),Vector2(0,1),Vector2(0,2) }, -- I
    { Vector2(0,0),Vector2(1,0),Vector2(1,1),Vector2(0,1) }, -- O
    { Vector2(-1,1),Vector2(0,1),Vector2(0,0),Vector2(1,0) }, -- S
    { Vector2(1,1),Vector2(0,1),Vector2(0,0),Vector2(-1,0) }, -- Z
    { Vector2(-1,1),Vector2(-1,0),Vector2(0,0),Vector2(1,0) }, -- L
    { Vector2(1,1),Vector2(1,0),Vector2(0,0),Vector2(-1,0) }, -- J
    { Vector2(0,1),Vector2(1,0),Vector2(0,0),Vector2(-1,0) } -- T
}

local block_rotations={
    Matrix32( Vector2(1,0),Vector2(0,1), Vector2() ),
    Matrix32( Vector2(0,1),Vector2(-1,0), Vector2() ),
    Matrix32( Vector2(-1,0),Vector2(0,-1), Vector2() ),
    Matrix32( Vector2(0,-1),Vector2(1,0), Vector2() )
}


local width=0
local height=0

local cells={}

local piece_active=false
local piece_shape=0
local piece_pos=Vector2()
local piece_rot=0

function node:piece_cell_xform(p, er)
    er = er or 0
    local r = (4 + er + piece_rot) % 4
    return piece_pos + block_rotations[r + 1]:xform(p)
end

function node:_draw()
    local sb = self:get_stylebox("bg", "Tree") -- use line edit bg
    self:draw_style_box(sb, Rect2(Vector2(), self:get_size()):grow(3))

    local bs = block:get_size()
    for y = 0, height - 1 do
        for x = 0, width - 1 do
            if cells[tostring(Vector2(x, y))] then
                self:draw_texture_rect(block,
                    Rect2(Vector2(x, y) * bs, bs),
                    false,
                    block_colors[cells[tostring(Vector2(x,y))] + 1]
                )
            end
        end
    end

    if piece_active then
        for _, c in pairs(block_shapes[piece_shape + 1]) do
            self:draw_texture_rect(block,
                Rect2(self:piece_cell_xform(c) * bs, bs),
                false,
                block_colors[piece_shape + 1]
            )
        end
    end
end

function node:piece_check_fit(ofs, er)
    er = er or 0

    for _, c in pairs(block_shapes[piece_shape + 1]) do
        local pos = self:piece_cell_xform(c, er) + ofs
        if pos.x < 0 then
            return false
        elseif pos.y < 0 then
            return false
        elseif pos.x >= width then
            return false
        elseif pos.y >= height then
            return false
        end

        if cells[tostring(pos)] then
            return false
        end
    end

    return true
end


function node:new_piece()
    piece_shape = math.random(0, MAX_SHAPES - 1)
    piece_pos = Vector2(width / 2, 0)
    piece_active = true
    piece_rot = 0
    if piece_shape == 0 then
        piece_pos.y = piece_pos.y + 1
    end

    if not self:piece_check_fit(Vector2()) then
        -- game over
        -- print("GAME OVER!")
        self:game_over()
    end

    self:update()
end

function node:test_collapse_rows()
    local accum_down=0
    for i = 0, height - 1 do
        local y = height - i - 1
        local collapse = true
        for x = 0, width - 1 do
            if cells[tostring(Vector2(x,y))] then
                if accum_down then
                    cells[tostring(Vector2(x, y + accum_down))] = cells[tostring(Vector2(x,y))]
                end
            else
                collapse = false
                if accum_down then
                    cells[tostring(Vector2(x, y + accum_down))] = nil
                end
            end
        end

        if collapse then
            accum_down = accum_down + 1
        end
    end

    score = score + (accum_down * 100)
    score_label:set_text(tostring(score))
end

function node:game_over()
    piece_active=false
    self:get_node("gameover"):set_text("Game Over")
    self:update()
end

function node:restart_pressed()
    score=0
    score_label:set_text("0")
    cells = {}
    self:get_node("gameover"):set_text("")
    piece_active=true
    self:update()
end

function node:piece_move_down()
    if not piece_active then
        return
    end

    if self:piece_check_fit(Vector2(0,1)) then
        piece_pos.y = piece_pos.y + 1
        self:update()
    else
        for _, c in pairs(block_shapes[piece_shape + 1]) do
            local pos = self:piece_cell_xform(c)
            cells[tostring(pos)] = piece_shape
        end
        self:test_collapse_rows()
        self:new_piece()
    end
end

function node:piece_rotate()
    local adv = 1
    if not self:piece_check_fit(Vector2(), 1) then
        return
    end
    piece_rot = (piece_rot + adv) % 4
    self:update()
end

function node:_input(ie)
    if not piece_active then
        return
    end

    if not ie:is_pressed() then
        return
    end

    if ie:is_action("move_left") then
        if self:piece_check_fit(Vector2(-1,0)) then
            piece_pos.x = piece_pos.x - 1
            self:update()
        end
    elseif ie:is_action("move_right") then
        if self:piece_check_fit(Vector2(1,0)) then
            piece_pos.x = piece_pos.x + 1
            self:update()
        end
    elseif ie:is_action("move_down") then
        self:piece_move_down()
    elseif ie:is_action("rotate") then
        self:piece_rotate()
    end
end

function node:setup(w, h)
    width = w
    height = h
    self:set_size(Vector2(w, h) * block:get_size())
    self:new_piece()
    self:get_node("timer"):start()
end

function node:_ready()
    -- Initalization here
    self:setup(10, 20)
    score_label = self:get_node("../score")

    self:set_process_input(true)
end
