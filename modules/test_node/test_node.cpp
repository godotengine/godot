/* test_node.cpp */

#include "test_node.h"

void TestNode::_bind_methods() 
{
//	ClassDB::bind_method(D_METHOD("isCool"), &TestNode::isCool);
	ClassDB::bind_method(D_METHOD("setCoolPath", "newPath"), &TestNode::setPath);
	ClassDB::bind_method(D_METHOD("getCoolPath"), &TestNode::getPath);
//
//	//Add a property to the editor
//								//Type		   label in editor   setter    getter
////	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "isCool"), "setCool", "isCool");
	ADD_PROPERTYNZ(PropertyInfo(Variant::NODE_PATH, "path"), "setCoolPath", "getCoolPath");
}

TestNode::TestNode()
{
	print_line("Path given: " + path);

	Node2D* node = dynamic_cast<Node2D*>( get_node(path) );

	if (node)
		print_line("Found node! :) " + node->get_class());
	else
		print_line("Couldn't find node! :(");

}


void TestNode::setPath(const NodePath& p_path)
{
	path = p_path;
	update();
}

NodePath TestNode::getPath() const
{
	return path;
}




void BunnyPosition::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("get_bunny_position"), &BunnyPosition::getPosition);
	ClassDB::bind_method(D_METHOD("set_bunny_position"), &BunnyPosition::setPosition);
	ClassDB::bind_method(D_METHOD("updateBunny", "deltaTime"), &BunnyPosition::updatePosition);
}

BunnyPosition::BunnyPosition() : position(50, 50)
{
	m_vx = Math::rand() % 200 + 50;
	m_vy = Math::rand() % 200 + 50;
	m_ay = 980;
}

Vector2 BunnyPosition::getPosition() const
{
	return position;
}

void BunnyPosition::setPosition(const Vector2& p_position)
{
	if (position != p_position)
		position = p_position;
}


void BunnyPosition::updatePosition( const float& deltaTime )
{
	position.x += m_vx * deltaTime;
	position.y += m_vy * deltaTime;

	m_vy += m_ay * deltaTime;

	if(position.x > 800)
	{
		m_vx *= -1;
		position.x = 800;
	}
	if(position.x < 0 )
	{
		m_vx = Math::abs(m_vx);
		position.x = 0;
	}

	if(position.y > 600)
	{
		m_vy = -0.85 * m_vy;
		position.y = 600;
		if ( Math::randf() > 0.5)
		{
			m_vy = -(Math::rand() % 1100 + 50);
		}
	}
	if( position.y < 0)
	{
		m_vy = 0;
		position.y = 0;
	}	
}


//-------------------

Bunny::Bunny()
{
	set_position( Point2(50, 50) );

	m_vx = Math::rand() % 200 + 50;
	m_vy = Math::rand() % 200 + 50;
	m_ay = 980;
}


void Bunny::updatePosition(const float& deltaTime)
{
	Point2 position = get_position();

	position.x += m_vx * deltaTime;
	position.y += m_vy * deltaTime;

	m_vy += m_ay * deltaTime;

	if (position.x > 800)
	{
		m_vx *= -1;
		position.x = 800;
	}
	if (position.x < 0)
	{
		m_vx = Math::abs(m_vx);
		position.x = 0;
	}

	if (position.y > 600)
	{
		m_vy = -0.85 * m_vy;
		position.y = 600;
		if (Math::randf() > 0.5)
		{
			m_vy = -(Math::rand() % 1100 + 50);
		}
	}
	if (position.y < 0)
	{
		m_vy = 0;
		position.y = 0;
	}

	set_position(position);
}

void Bunny::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("updateBunny", "deltaTime"), &Bunny::updatePosition);
}

void Bunny::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_PROCESS:
			{
				updatePosition(get_process_delta_time());
			}
			break;

		case NOTIFICATION_DRAW:
			{
			//print_line("Drawing");
			}
			break;

		case NOTIFICATION_ENTER_TREE:
			{
			if( !get_tree()->is_editor_hint() )
				set_process(true);
			//print_line("Enter Tree");
			}
			break;

		case NOTIFICATION_EXIT_TREE:
			{
			if (!get_tree()->is_editor_hint())
				set_process(false);
			//print_line("Exit Tree");
			}
			break;

		case NOTIFICATION_READY:
			{
			//print_line("Ready");
			
			
			}
			break;

		case NOTIFICATION_ENTER_CANVAS:
			{
				//print_line("Enter Canvas");

			}
			break;

		case NOTIFICATION_EXIT_CANVAS:
			{
				//print_line("Exit Canvas");
			}
			break;
		case NOTIFICATION_POSTINITIALIZE:
			{
				//print_line("Post initialize");
			}
			break;
	}
}

//-------------------
BunnyController::BunnyController()
{
}



void BunnyController::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_texture", "texture:Texture"), &BunnyController::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture:Texture"), &BunnyController::get_texture);
	
	ClassDB::bind_method(D_METHOD("get_starting_bunnies:int"), &BunnyController::get_starting_bunnies);
	ClassDB::bind_method(D_METHOD("set_starting_bunnies", "int"), &BunnyController::set_starting_bunnies);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_texture", "get_texture");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "startingBunnies"), "set_starting_bunnies", "get_starting_bunnies");

}

void BunnyController::_notification(int p_what) {

	switch (p_what) {
	case NOTIFICATION_PROCESS:
	{
	}
	break;

	case NOTIFICATION_DRAW:
	{
	}
	break;

	case NOTIFICATION_ENTER_TREE:
	{
	}
	break;

	case NOTIFICATION_EXIT_TREE:
	{
	}
	break;

	case NOTIFICATION_READY:
	{
		if (!get_tree()->is_editor_hint())
			spawnBunnies(startingBunnies);
	}
	break;

	case NOTIFICATION_ENTER_CANVAS:
	{
	}
	break;

	case NOTIFICATION_EXIT_CANVAS:
	{
	}
	break;
	case NOTIFICATION_POSTINITIALIZE:
	{
	}
	break;
	}
}

void BunnyController::spawnBunnies( int p_amount )
{
	print_line("Spawn bunnies! " + p_amount );

	for( int i = 0; i < p_amount; i++)
	{
		Bunny* b = memnew(Bunny());
		b->set_texture(bunnyTexture);
		add_child(b);

		totalBunnies++;
	}
}

void BunnyController::set_texture(const Ref<Texture> &p_texture) 
{
	if (p_texture == bunnyTexture)
		return;

	bunnyTexture = p_texture;
	update();
}

Ref<Texture> BunnyController::get_texture() const 
{
	return bunnyTexture;
}

int BunnyController::get_starting_bunnies() const
{
	return startingBunnies;
}

void BunnyController::set_starting_bunnies(const int& p_starting)
{
	startingBunnies = p_starting;
	update();
}
