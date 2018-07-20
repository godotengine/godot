namespace Godot
{
    public partial class Node
    {
        public T GetNode<T>(NodePath path) where T : Godot.Node
        {
            return (T)GetNode(path);
        }
    }
}
