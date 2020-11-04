 function structB = add_fields(structA, structB)
    f = fieldnames(structA);
    for i = 1:length(f)
        structB.(f{i}) = structA.(f{i});
     end