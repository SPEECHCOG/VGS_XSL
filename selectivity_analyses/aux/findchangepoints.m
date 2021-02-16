function i = findchangepoints(seq,direction)
% function i = findchangepoints(seq,direction)
%
% Finds indices [i] where value of [seq] changes between subsequent
% elements
%
% direction = 0 : detect all changes (default)
% direction > 0 : detect only positive changes
% direction < 0 : detect only negative changes


a = diff(seq);

if(nargin <2 || direction == 0)
    i = find(a ~= 0);
    
elseif(direction < 0)
    i = find(a < 0);
    
elseif(direction > 0)
    i = find(a > 0);
end