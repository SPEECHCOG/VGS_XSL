function list_out = cellunion(list1,list2)
% function list_out = cellunion(list1,list2)
%
% Performs an union of two cell array lists
%
% Example:
%
%   list1 = {'foo','bar','ping'};
%   list2 = {'foo','pong'};
%   list_out = cellintersect(list1,list2)
%   --> list_out = {'foo','bar','ping','pong'}



if(size(list1,2) > size(list1,1))
   transposed = 1; 
   list1 = list1';   
else
    transposed = 0;
end

for j = 1:length(list2)
   a = strcmp(list1,list2{j});
   if(sum(a) == 0)
      list1 = [list1;list2{j}]; 
   end
end

if(transposed == 1)
    list1 = list1';
end

list_out = list1;