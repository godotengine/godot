extends CharacterBody2D
uses BadTrait

trait BadTrait:
    extends CollisionObject2D
    func move_and_collide()
