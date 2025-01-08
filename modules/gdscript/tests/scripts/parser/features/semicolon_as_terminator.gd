#GDTEST_OK

func test():
	a();
	b();
	c();
	d();
	e();

func a(): print("a");

func b(): print("b1"); print("b2")

func c(): print("c1"); print("c2");

func d():
	print("d1");
	print("d2")

func e():
	print("e1");
	print("e2");
