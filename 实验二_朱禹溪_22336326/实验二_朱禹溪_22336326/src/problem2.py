from main import final_resolve
KB_str2 = "{(On(tony,mike),),(On(mike,john),),(Green(tony),),(~Green(john),),(~On(xx,yy),~Green(xx),Green(yy))}"
result =final_resolve(KB_str2)
