using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// An interface that requires a boolean for completion status and a method that gets the result of completion.
    /// </summary>
    public interface IAwaiter : INotifyCompletion
    {
        /// <summary>
        /// The completion status of this <see cref="IAwaiter"/>.
        /// </summary>
        public bool IsCompleted { get; }

        /// <summary>
        /// Gets the result of completion for this <see cref="IAwaiter"/>.
        /// </summary>
        public void GetResult();
    }

    /// <summary>
    /// A templated interface that requires a boolean for completion status and a method that gets the result of completion and returns it.
    /// </summary>
    /// <typeparam name="TResult">A reference to the result to be passed out.</typeparam>
    public interface IAwaiter<out TResult> : INotifyCompletion
    {
        /// <summary>
        /// The completion status of this <see cref="IAwaiter{TResult}"/>.
        /// </summary>
        public bool IsCompleted { get; }

        /// <summary>
        /// Gets the result of completion for this <see cref="IAwaiter{TResult}"/>.
        /// </summary>
        public TResult GetResult();
    }
}
