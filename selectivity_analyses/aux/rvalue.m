function rval = rvalue(hit,os)
% This function returns R-value of segmentation results (hit and
% over-segmentation rate).
%
% Inputs:
%   hit         : hit rate (hits/total targets, 0-100 %)
%   os          : over segmentation rate hypos / total targets (0-100%)

rval = 1-(abs((-1.*os+hit-100)/sqrt(1+1))+sqrt((100-hit).^2+(abs(os)).^2))/200;



