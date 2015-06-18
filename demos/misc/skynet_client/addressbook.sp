.Person {
	name 0 : string
	id 1 : integer
	email 2 : string
	.PhoneNumber {
		number 0 : string
		type 1 : integer
	}
	phone 3 : *PhoneNumber
	cash 4 : real
}
.AddressBook {
	person 0 : *Person(id)
	others 1 : *Person
	element 2 : Person
}
