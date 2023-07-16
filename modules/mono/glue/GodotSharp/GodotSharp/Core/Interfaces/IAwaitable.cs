namespace Godot
{
    /// <summary>
    /// An interface that requires a GetAwaiter() method to get a reference to the Awaiter.
    /// </summary>
    public interface IAwaitable
    {
        /// <summary>
        /// Gets a reference to the <see cref="IAwaiter"/>.
        /// </summary>
        IAwaiter GetAwaiter();
    }

    /// <summary>
    /// A templated interface that requires a GetAwaiter() method to get a reference to the Awaiter.
    /// </summary>
    /// <typeparam name="TResult">A reference to the result to be passed out.</typeparam>
    public interface IAwaitable<out TResult>
    {
        /// <summary>
        /// Gets a reference to the <see cref="IAwaiter{TResult}"/>.
        /// </summary>
        IAwaiter<TResult> GetAwaiter();
    }
}
