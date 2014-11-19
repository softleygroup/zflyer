%% How to restruct a matrix
%

for i = 1:61
        Bz_Zurich(i,1) = decelmapaxrho(i,3);
        Br_Zurich(i,1) = decelmapaxrho(i,4);    
end



for j = 1:61
    for m = 1:(441-1)
        Bz_Zurich(j,m+1) = decelmapaxrho(m*61+j,3);
        Br_Zurich(j,m+1) = decelmapaxrho(m*61+j,4);
    end
end

dlmwrite('Bz_Zurich.txt', Bz_Zurich, 'delimiter', '\t', ...
    'precision', 16);
dlmwrite('Br_Zurich.txt', Br_Zurich, 'delimiter', '\t', ...
    'precision', 16);

