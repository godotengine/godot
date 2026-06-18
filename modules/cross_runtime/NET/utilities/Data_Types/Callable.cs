using System;
using System.Runtime.InteropServices;
using System.Diagnostics.CodeAnalysis;

#nullable enable
namespace Godot
{
	/// <summary>
	/// Stores the target instance id and method name.
	/// </summary>
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public readonly partial struct Callable
	{
		private readonly GodotObject _target;
		private readonly StringName _method;

		/// <summary>
		/// Object id that contains the method.
		/// </summary>
		public GodotObject Target => _target;

		/// <summary>
		/// Name of the method that will be called.
		/// </summary>
		public StringName Method => _method;

		/// <summary>
		/// Constructs a new <see cref="Callable"/> for the method called <paramref name="method"/>
		/// in the specified <paramref name="target"/>.
		/// </summary>
		/// <param name="target">Object that contains the method.</param>
		/// <param name="method">Name of the method that will be called.</param>
		public Callable(GodotObject target, StringName method)
		{
			_target = target;
			_method = method;
		}

	}
}
