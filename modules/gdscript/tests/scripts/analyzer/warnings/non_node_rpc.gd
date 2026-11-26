extends Object

@rpc func test(): pass

class InnerNode extends Node:
    @rpc func test(): pass

class InnerNode2 extends InnerNode:
    @rpc func test(): pass

class ResourceClass extends Resource:
    @rpc func test(): pass

class InnerResource2 extends ResourceClass:
    @rpc func test(): pass

class InnerClass extends RefCounted:
    @rpc func test(): pass
