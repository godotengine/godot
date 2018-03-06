namespace Godot
{
    public partial class Node
    {
        public T GetNode<T>(NodePath path) where T : Godot.Node
        {
            return (T)GetNode(path);
        }

        public T GetNodeOrNull<T>(NodePath path) where T : Godot.Node
        {
            return GetNode(path) as T;
        }

        public T GetChild<T>(int idx) where T : Godot.Node
        {
            return (T)GetChild(idx);
        }

        public T GetChildOrNull<T>(int idx) where T : Godot.Node
        {
            return GetChild(idx) as T;
        }

        public T GetOwner<T>() where T : Godot.Node
        {
            return (T)GetOwner();
        }

        public T GetOwnerOrNull<T>() where T : Godot.Node
        {
            return GetOwner() as T;
        }

        public T GetParent<T>() where T : Godot.Node
        {
            return (T)GetParent();
        }

        public T GetParentOrNull<T>() where T : Godot.Node
        {
            return GetParent() as T;
        }
    }
}
