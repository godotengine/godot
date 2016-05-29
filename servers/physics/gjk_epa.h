/*************************************************/
/*  gjk_epa.h                                    */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef GJK_EPA_H
#define GJK_EPA_H

#include "shape_sw.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
#include "collision_solver_sw.h"

bool gjk_epa_calculate_penetration(const ShapeSW *p_shape_A, const Transform& p_transform_A, const ShapeSW *p_shape_B, const Transform& p_transform_B, CollisionSolverSW::CallbackResult p_result_callback,void *p_userdata, bool p_swap=false);
bool gjk_epa_calculate_distance(const ShapeSW *p_shape_A, const Transform& p_transform_A, const ShapeSW *p_shape_B, const Transform& p_transform_B, Vector3& r_result_A, Vector3& r_result_B);

#endif
