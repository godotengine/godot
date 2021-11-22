using Godot;
using NUnit.Framework;

namespace GodotSharp.UnitTest
{
    public class Vector2Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Vector2_constructor()
        {
            var vector2 = new Vector2(1.1f, 2.2f);
            Assert.AreEqual(1.1, vector2.x);
            Assert.AreEqual(2.2, vector2.y);
        }

        [Test]
        public void Vector2_constructor_from_other()
        {
            var vector2 = new Vector2(1.1f, 2.2f);
            var vector2Copy = new Vector2(vector2);
            Assert.AreNotSame(vector2, vector2Copy);
            Assert.AreEqual(vector2.x, vector2Copy.x);
            Assert.AreEqual(vector2.y, vector2Copy.y);
        }

        [Test]
        public void Vector2_xy()
        {
            var vector2 = new Vector2(1.1f, 2.2f);
            var vector2Copy = vector2.xy;
            Assert.AreNotSame(vector2, vector2Copy);
            Assert.AreEqual(vector2.x, vector2Copy.x);
            Assert.AreEqual(vector2.y, vector2Copy.y);
        }

        [Test]
        public void Vector2_yx()
        {
            var vector2 = new Vector2(1.1f, 2.2f);
            var vector2Copy = vector2.yx;
            Assert.AreNotSame(vector2, vector2Copy);
            Assert.AreEqual(vector2.x, vector2Copy.y);
            Assert.AreEqual(vector2.y, vector2Copy.x);
        }
    }
}
