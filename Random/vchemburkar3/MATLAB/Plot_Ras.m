
[x1,x2] = meshgrid ( -2:0.05:2, -2:0.05:2);
Ras =20+x1.^2+x2.^2- 10*(cos(2*pi*x1)+cos(2*pi*x2));
% subplot(1,2,2)
contour(x1,x2,Ras,1:4:40);
title('Contour Plot of Rastrigin Function')
% hold on
% for  i = 1:81
% plot(best_points(i,:),'+r');
% end


subplot(1,2,1)
surf (x1,x2,Ras, 'EdgeColor', 'none');
shading interp;
title('Surface Plot of Rastrigin Function') 