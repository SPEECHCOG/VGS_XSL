function teekuva(filename,style,directory)
% Saves picture with filename to the default directory
%   Inputs:
%       filename    name of the picture file
%       style       0: black&white .eps, 1: colored .eps, 2: colored .jpeg
%       directory   save to other than default directory

if nargin <3 directory = []; end
if nargin<2 style = 1; end

if(isempty(directory))
    filename1 = ['/Users/rasaneno/Documents/kuvat/' filename];
    filename2 = ['/Users/rasaneno/Documents/kuvat/figs/' filename '.fig'];
else
    filename1 = [directory filename];
    filename2 = [directory filename '.fig'];
end


if(style == 1)
    set(gcf,'PaperPositionMode','auto')
    print(filename1,'-depsc','-loose');
    saveas(gcf,filename2, 'fig')
elseif(style == 2)
    set(gcf,'PaperPositionMode','auto')
    print(filename1,'-djpeg90','-loose');
    saveas(gcf,filename2, 'fig')
else
    set(gcf,'PaperPositionMode','auto')
    print(filename1,'-deps','-loose');
    saveas(gcf,filename2, 'fig')
end
