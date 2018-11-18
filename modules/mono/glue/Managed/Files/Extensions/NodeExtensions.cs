namespace Godot
{
    public partial class Node
    {
        public T GetNode<T>(NodePath path) where T : class
        {
            return (T)(object)GetNode(path);
        }

        public T GetNodeOrNull<T>(NodePath path) where T : class
        {
            return GetNode(path) as T;
        }

        public T GetChild<T>(int idx) where T : class
        {
            return (T)(object)GetChild(idx);
        }

        public T GetChildOrNull<T>(int idx) where T : class
        {
            return GetChild(idx) as T;
        }

        public T GetOwner<T>() where T : class
        {
            return (T)(object)GetOwner();
        }

        public T GetOwnerOrNull<T>() where T : class
        {
            return GetOwner() as T;
        }

        public T GetParent<T>() where T : class
        {
            return (T)(object)GetParent();
        }

        public T GetParentOrNull<T>() where T : class
        {
            return GetParent() as T;
        }
    }
}
