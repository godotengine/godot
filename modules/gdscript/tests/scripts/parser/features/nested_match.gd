func test():
	# 20 levels of nesting (and then some).
	var number = 1234
	match number:
		1234:
			print("1")
			match number:
				1234:
					print("2")
					match number:
						4321:
							print("Should not be printed")
						_:
							print("3")
							match number:
								1234:
									print("4")
									match number:
										_:
											print("5")
											match number:
												false:
													print("Should not be printed")
												true:
													print("Should not be printed")
												"hello":
													print("Should not be printed")
												1234:
													print("6")
													match number:
														_:
															print("7")
															match number:
																1234:
																	print("8")
																	match number:
																		_:
																			print("9")
																			match number:
																				1234:
																					print("10")
																					match number:
																						_:
																							print("11")
																							match number:
																								1234:
																									print("12")
																									match number:
																										_:
																											print("13")
																											match number:
																												1234:
																													print("14")
																													match number:
																														_:
																															print("15")
																															match number:
																																_:
																																	print("16")
																																	match number:
																																		1234:
																																			print("17")
																																			match number:
																																				_:
																																					print("18")
																																					match number:
																																						1234:
																																							print("19")
																																							match number:
																																								_:
																																									print("20")
																																									match number:
																																										[]:
																																											print("Should not be printed")
				_:
					print("Should not be printed")
		5678:
			print("Should not be printed either")
