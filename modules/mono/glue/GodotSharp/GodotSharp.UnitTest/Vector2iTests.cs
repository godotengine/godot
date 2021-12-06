using Godot;
using NUnit.Framework;

namespace GodotSharp.UnitTest
{
    public class Vector2iTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Vector2i_constructor()
        {
            var vector2i = new Vector2i(1, 2);
            Assert.AreEqual(1, vector2i.x);
            Assert.AreEqual(2, vector2i.y);
        }

        [Test]
        public void Vector2i_constructor_from_other()
        {
            var vector2i = new Vector2i(1, 2);
            var vector2iCopy = new Vector2i(vector2i);
            Assert.AreNotSame(vector2i, vector2iCopy);
            Assert.AreEqual(vector2i.x, vector2iCopy.x);
            Assert.AreEqual(vector2i.y, vector2iCopy.y);
        }

        [Test]
        public void Vector2i_constructor_from_Vector2()
        {
            var vector2i = new Vector2i(new Vector2(1.2f, 2.6f));
            Assert.AreEqual(1, vector2i.x);
            Assert.AreEqual(3, vector2i.y);
        }

        [Test]
        public void Vector2i_xy()
        {
            var vector2i = new Vector2i(1, 2);
            var vector2iCopy = vector2i.xy;
            Assert.AreNotSame(vector2i, vector2iCopy);
            Assert.AreEqual(vector2i.x, vector2iCopy.x);
            Assert.AreEqual(vector2i.y, vector2iCopy.y);
        }

        [Test]
        public void Vector2i_yx()
        {
            var vector2i = new Vector2i(1, 2);
            var vector2iCopy = vector2i.yx;
            Assert.AreNotSame(vector2i, vector2iCopy);
            Assert.AreEqual(vector2i.x, vector2iCopy.y);
            Assert.AreEqual(vector2i.y, vector2iCopy.x);
        }
    }
}
