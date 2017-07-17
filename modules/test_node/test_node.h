#pragma once
/* test_node.h */
#ifndef TEST_NODE_H
#define TEST_NODE_H


#include "scene/2d/node_2d.h"
#include "object.h"
#include "math_2d.h"
#include "math.h"
#include "scene/2d/sprite.h"
#include "engine.h"


class TestNode : public Node2D {
	//needed to bind the class into the engine
	GDCLASS(TestNode, Node2D);

	NodePath path;

protected:
	static void _bind_methods();

public:
	TestNode();

	void setPath(const NodePath& p_path);
	NodePath getPath() const;
};


class BunnyPosition : public  Object
{
	GDCLASS(BunnyPosition, Object)

protected:
	static void _bind_methods();

public:
	BunnyPosition();

	Vector2 getPosition() const;
	void setPosition( const Vector2& p_position);
	void updatePosition(const float& deltaTime);

private:
	int m_vx;
	int m_vy;
	int m_ay;
	Vector2 position;
};

class Bunny : public Sprite
{
	GDCLASS(Bunny, Sprite)

protected:
	static void _bind_methods();
	void _notification(int p_what);



public:
	Bunny();

	void updatePosition(const float& deltaTime);

private:
	int m_vx;
	int m_vy;
	int m_ay;
};

class BunnyController : public Node2D
{
	GDCLASS(BunnyController, Node2D);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	BunnyController();

	void spawnBunnies( int);

	Ref<Texture> get_texture() const;
	void set_texture(const Ref<Texture> &p_texture);

	int get_starting_bunnies() const;
	void set_starting_bunnies( const int& p_starting );

private:
	Ref<Texture> bunnyTexture;

	int totalBunnies = 0;
	int startingBunnies = 100;
};
#endif