#pragma once

struct Base
{
	virtual int sync();
};

struct Test : public Base
{
	int sync() override;

	char buffer[20];
};
