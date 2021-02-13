namespace Godot
{
    public static partial class ResourceLoader
    {
        public static T Load<T>(string path, string typeHint = null, CacheMode noCache = CacheMode.Reuse) where T : class
        {
            return (T)(object)Load(path, typeHint, noCache);
        }
    }
}
