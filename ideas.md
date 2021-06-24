
use only 1 real value as Q(model)'s output instead of 4(long, short, ...)
1 means buy all, 0.5 means buy 0.5 repo;
-0.5 means sold 0.5 repo.
old method, return qvalue, and it's index means actions 
new method, return 1value means actions(long/short), this actions's value could
use another value, or 
