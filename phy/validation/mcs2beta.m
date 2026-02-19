function beta = mcs2beta(mcs)
% Initial MCS to EESM tuning parameter beta mapping
switch mcs
    case 0
        beta = 1;
    case 1
        beta = 2;
    case 2
        beta = 3;
    case 3
        beta = 5;
    case 4
        beta = 8;
    case 5
        beta = 34;
    case 6
        beta = 40;
    case 7
        beta = 49;
    case 8
        beta = 142;
    case 9
        beta = 197;
end
end