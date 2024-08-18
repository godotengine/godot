#ifndef _MCURVE_TERRAIN
#define _MCURVE_TERRAIN

#include "../mterrain.h"
#include "../mgrid.h"
#include "mcurve.h"

using namespace godot;

class MCurveTerrain : public RefCounted {
    GDCLASS(MCurveTerrain,RefCounted);

    protected:
    static void _bind_methods();

    private:
    bool apply_tilt = false;
    bool apply_scale = true;

    float deform_offest = 0.0f;
    float deform_radius = 6.0f;
    float deform_falloff = 2.0f;
    MCurve* curve=nullptr;
    MTerrain* terrain=nullptr;
    MGrid* grid=nullptr;
    String terrain_layer_name;
    /// Image
    String terrain_image_name;
    Color paint_color = Color(1.0f,1.0f,1.0f,1.0f);
    Color bg_color = Color(0.0f,0.0f,0.0f,1.0f);
    float paint_radius = 6.0f;
    float paint_falloff = 0.0f;

    public:
    void set_curve(MCurve* input);
    MCurve* get_curve();
    void set_terrain(MTerrain* m_terrain);
    MTerrain* get_terrain();
    void set_terrain_layer_name(const String& input);
    String get_terrain_layer_name();

    void set_apply_tilt(bool input);
    bool get_apply_tilt();
    void set_apply_scale(bool input);
    bool get_apply_scale();

    void set_deform_offest(float input);
    float get_deform_offest();
    void set_deform_radius(float input);
    float get_deform_radius();
    void set_deform_falloff(float input);
    float get_deform_falloff();

    void set_terrain_image_name(const String& input);
    String get_terrain_image_name() const;
    void set_paint_color(const Color& input);
    Color get_paint_color() const;
    void set_bg_color(const Color& input);
    Color get_bg_color() const;
    void set_paint_radius(float input);
    float get_paint_radius();
    void set_paint_falloff(float input);
    float get_paint_falloff();

    void clear_deform_aabb(AABB aabb);
    void clear_deform(const PackedInt64Array& conn_ids);
    void deform_on_conns(const PackedInt64Array& conn_ids);

    void clear_paint_aabb(AABB aabb);
    void clear_paint(const PackedInt64Array& conn_ids);
    void paint_on_conns(const PackedInt64Array& conn_ids);

    void clear_grass_aabb(MGrass* grass,AABB aabb,float radius_plus_offset);
    void clear_grass(const PackedInt64Array& conn_ids,MGrass* grass,float radius_plus_offset);
    void modify_grass(const PackedInt64Array& conn_ids,MGrass* grass,float g_start_offset,float g_radius,bool add);
};
#endif