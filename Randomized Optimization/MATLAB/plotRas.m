
function [] = plotRas (cont)

if ( nargin == 0 )
    [x1,x2] = meshgrid ( -5:0.05:5, -5:0.05:5);
    Z = 20+x1.^2+x2.^2- 10*(cos(2*pi*x1)+cos(2*pi*x2));

    surf (x1,x2,Z, 'EdgeColor', 'none');
    shading interp;
    figure;
end;

[x1,x2] = meshgrid ( -2:0.05:2, -2:0.05:2);
Z = 20+x1.^2+x2.^2- 10*(cos(2*pi*x1)+cos(2*pi*x2));

[C,h] = contour(x1,x2,Z,1:4:40);

if ( nargin == 0 )
text_handle = clabel(C,h);
set(text_handle,'BackgroundColor',[1 1 .6], 'Edgecolor',[.7 .7 .7]);
end