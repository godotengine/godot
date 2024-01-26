#ifndef GAME_GUI_IMAGE_H
#define GAME_GUI_IMAGE_H

#include "game_gui_compoent.h"

class GUIImage : public GUIComponent
{
    GDCLASS(GUIImage, GUIComponent);
    static void _bind_methods();
    public:

    Rect2 _texture_region;
    Ref<Texture2D>  texture;
    Vector<Rect2> _piece_rects;
    void set_texture(const Ref<Texture2D> &p_texture) {
        texture = p_texture;
        if (texture == nullptr) return;

		_texture_region = Rect2( 0, 0, texture.get_width(), texture.get_height() );

		_update_piece_rects();
    }
    Ref<Texture2D>  get_texture() {
        return texture;
    }

    bool draw_center = true;
    void set_draw_center(bool value)
    {
        draw_center = value;
        queue_redraw();
    }
    bool get_draw_center() { return draw_center; }

    int left = 0;
    void set_left(int value)
    {
		left = Math::CLAMP( value, 0, _texture_region.size.x );
		_update_piece_rects();
    }
    int get_left() { return left; }

    int right = 0;
    void set_right(int value)
    {
        right = Math::CLAMP( value, 0, _texture_region.size.x );
        _update_piece_rects();
    }
    int get_right() { return right; }

    int top = 0;
    void set_top(int value)
    {
        top = Math::CLAMP( value, 0, _texture_region.size.y );
        _update_piece_rects();
    }
    int get_top() { return top; }

    int bottom = 0;
    void set_bottom(int value)
    {
        bottom = Math::CLAMP( value, 0, _texture_region.size.y );
        _update_piece_rects();
    }

    int get_bottom() { return bottom; }


    FillMode horizontal_fill;
    void set_horizontal_fill(FillMode value)
    {
        horizontal_fill = value;
        queue_redraw();
    }
    FillMode get_horizontal_fill() { return horizontal_fill; }

    FillMode vertical_fill;
    void set_vertical_fill(FillMode value)
    {
        vertical_fill = value;
        queue_redraw();
    } 
    FillMode get_vertical_fill() { return vertical_fill; }

    virtual void _draw() override
    {
        if (texture.is_null()) return;
        if (size.x == 0 or size.y == 0)
         return;

        auto _left = left;
        auto _right = right;
        auto _top = top;
        auto _bottom = bottom;

        if (left + right > size.x || top + bottom > size.y)
        {
            auto scale_x = size.x / (_left + _right);
            auto scale_y = size.y / (_top + _bottom);
            auto scale = MIN( scale_x, scale_y );

            _left = Math::floor( _left * scale );
            _right = Math::ceil( _right * scale );
            _top = Math::floor( _top * scale );
            _bottom = Math::ceil( _bottom * scale );

        }

        auto mid_w = MAX( size.x - (_left+_right), 0 );
        auto mid_h = MAX( size.y - (_top+_bottom), 0 );

        auto pos = position;
        if (_top > 0)
        {
            if (_left > 0)   draw_texture_rect_region( texture, Rect2(pos,Vector2(_left,_top)), _piece_rects[0], modulate );
            pos += Vector2( _left, 0 );
            fill_texture( texture, Rect2(pos,Vector2(mid_w,_top)), _piece_rects[1], horizontal_fill, vertical_fill, modulate );
            pos += Vector2( mid_w, 0 );
            if (_right > 0)  draw_texture_rect_region( texture, Rect2(pos,Vector2(_right,_top)), _piece_rects[2], modulate );

        }

        pos = Vector2( position.x, pos.y + _top );
        if (mid_h > 0)
        {
            fill_texture( texture, Rect2(pos,Vector2(_left,mid_h)), _piece_rects[3], horizontal_fill, vertical_fill, modulate );
            pos += Vector2( _left, 0 );
            if (draw_center && mid_w > 0)  fill_texture( texture, Rect2(pos,Vector2(mid_w,mid_h)), _piece_rects[4], horizontal_fill, vertical_fill, modulate );
            pos += Vector2( mid_w, 0 );
            fill_texture( texture, Rect2(pos,Vector2(_right,mid_h)), _piece_rects[5], horizontal_fill, vertical_fill, modulate );
        }

        pos = Vector2( position.x, pos.y + mid_h );
        if (_bottom > 0)
        {
            if (_left > 0)   draw_texture_rect_region( texture, Rect2(pos,Vector2(_left,_bottom)), _piece_rects[6], modulate );
            pos += Vector2( _left, 0 );
            fill_texture( texture, Rect2(pos,Vector2(mid_w,_bottom)), _piece_rects[7], horizontal_fill, vertical_fill, modulate );
            pos += Vector2( mid_w, 0 );
            if (_right > 0)  draw_texture_rect_region( texture, Rect2(pos,Vector2(_right,_bottom)), _piece_rects[8], modulate );

        }
    }
    
    void _update_piece_rects()
    {
        auto x = _texture_region.position.x;
        auto y = _texture_region.position.y;
        auto w = _texture_region.size.x;
        auto h = _texture_region.size.y;
        auto mid_w = max( w - (left+right), 0 );
        auto mid_h = max( h - (top+bottom), 0 );

        _piece_rects.clear();

        _piece_rects.push_back( Rect2     (      x, y, left,  top) );  # TL
        _piece_rects.push_back( Rect2(      x+left, y, mid_w, top) );  # T
        _piece_rects.push_back( Rect2( x+(w-right), y, right, top) );  # TR

        _piece_rects.push_back( Rect2(           x, y+top, left,  mid_h) );  # L
        _piece_rects.push_back( Rect2(      x+left, y+top, mid_w, mid_h) );  # M
        _piece_rects.push_back( Rect2( x+(w-right), y+top, right, mid_h) );  # R

        _piece_rects.push_back( Rect2(           x, y+(h-bottom), left,  bottom) );  # BL
        _piece_rects.push_back( Rect2(      x+left, y+(h-bottom), mid_w, bottom) );  # B
        _piece_rects.push_back( Rect2( x+(w-right), y+(h-bottom), right, bottom) );  # BR

        queue_redraw();

    }



};

#endif