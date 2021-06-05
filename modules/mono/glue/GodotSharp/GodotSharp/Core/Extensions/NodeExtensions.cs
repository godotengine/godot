namespace Godot
{
    public partial class Node
    {
        public T GetNode<T>(NodePath path) where T : class => (T)(object)GetNode(path);

        public T GetNodeOrNull<T>(NodePath path) where T : class => GetNodeOrNull(path) as T;

        public T GetChild<T>(int idx) where T : class => (T)(object)GetChild(idx);

        public T GetChildOrNull<T>(int idx) where T : class => GetChild(idx) as T;

        public T GetOwner<T>() where T : class => (T)(object)Owner;

        public T GetOwnerOrNull<T>() where T : class => Owner as T;

        public T GetParent<T>() where T : class => (T)(object)GetParent();

        public T GetParentOrNull<T>() where T : class => GetParent() as T;
    }
}
