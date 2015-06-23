/*************************************************************************/
/*  register_server_types.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "register_server_types.h"
#include "globals.h"

#include "visual_server.h"
#include "audio_server.h"
#include "physics_server.h"
#include "physics_2d_server.h"
#include "spatial_sound_server.h"
#include "spatial_sound_2d_server.h"

void register_server_types() {

	Globals::get_singleton()->add_singleton( Globals::Singleton("VisualServer",VisualServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("VS",VisualServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("AudioServer",AudioServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("AS",AudioServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("PhysicsServer",PhysicsServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("PS",PhysicsServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("Physics2DServer",Physics2DServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("PS2D",Physics2DServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("SpatialSoundServer",SpatialSound2DServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("SS",SpatialSound2DServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("SpatialSound2DServer",SpatialSound2DServer::get_singleton()) );
	Globals::get_singleton()->add_singleton( Globals::Singleton("SS2D",SpatialSound2DServer::get_singleton()) );


	ObjectTypeDB::register_virtual_type<Physics2DDirectBodyState>();
	ObjectTypeDB::register_virtual_type<Physics2DDirectSpaceState>();
	ObjectTypeDB::register_virtual_type<Physics2DShapeQueryResult>();
	ObjectTypeDB::register_type<Physics2DTestMotionResult>();
	ObjectTypeDB::register_type<Physics2DShapeQueryParameters>();

	ObjectTypeDB::register_type<PhysicsShapeQueryParameters>();
	ObjectTypeDB::register_virtual_type<PhysicsDirectBodyState>();
	ObjectTypeDB::register_virtual_type<PhysicsDirectSpaceState>();
	ObjectTypeDB::register_virtual_type<PhysicsShapeQueryResult>();

}

void unregister_server_types(){


}
