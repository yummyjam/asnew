clear all;clc
clear all;clc
I_gray1=imread('C:\Users\yummy\Desktop\实验图片\第二组-5\10-5.bmp');
%I_gray1=rgb2gray(I_gray1);
%滤波
% I_gray1 = medfilt2(I_gray1); 
w=fspecial('average');
I_gray1=imfilter(I_gray1,w,'replicate');
%figure;imshow(I_gray1);

B=strel('disk',3);
I_gray1=imclose(I_gray1,B); %腐蚀 
I_gray1=imopen(I_gray1,B);  %膨胀  

Img=imcrop(I_gray1,[1450,1350,100,100]);
level=graythresh(Img);  %%求二值化的阈值
bw_=im2bw(I_gray1,level);    %%二值化图像
[height,width]=size(I_gray1);

%bw=I_gray1;

aw=edge(bw_,'canny');
bw=bwmorph(aw,'thin',inf);
%figure;imshow(aw);
bw=imcrop(aw,[1450,1350,100,100]);
%求边缘坐标
bw=imfill(bw,'holes');
figure;imshow(bw);
[Bc,L]=bwboundaries(bw,'noholes');
x=Bc{1,1}(:,1);
y=Bc{1,1}(:,2);
%数组的行数 定义成N
Pointxy=[x,y];
save('xyData.mat','Pointxy');
%%
%直接计算法
para = funcEllipseFit_direct(Pointxy);
para=para';
p1=para/para(1,1);
A1_=p1(1,2);
B1_=p1(1,3);
C1_=p1(1,4);
D1_=p1(1,5);
E1_=p1(1,6);
[N,M]=size(x);
a1_= sqrt(abs(2*(A1_*C1_*D1_-B1_*C1_*C1_-D1_*D1_+4*B1_*E1_-A1_*A1_*E1_)/((A1_*A1_-4*B1_)*(B1_-sqrt(A1_*A1_+(1-B1_)*(1-B1_))+1))));
b1_ = sqrt(abs(2*(A1_*C1_*D1_-B1_*C1_*C1_-D1_*D1_+4*B1_*E1_-A1_*A1_*E1_)/((A1_*A1_-4*B1_)*(B1_+sqrt(A1_*A1_+(1-B1_)*(1-B1_))+1))));
Xc1_ = (2*B1_*C1_-A1_*D1_)/(A1_*A1_-4*B1_)+1350;
Yc1_ = (2*D1_-A1_*D1_)/(A1_*A1_-4*B1_)+1450;
theta1_ = atan(sqrt(abs((a1_*a1_-b1_*b1_*B1_)/(a1_*a1_*B1_-b1_*b1_))));
A_1 = [Yc1_,Xc1_,a1_,b1_,theta1_];

%%
%非线性拟合法
% [F,p]=funcEllipseFit_nlinfit(Pointxy); %F为函数 p为A、B、C、D、E、F的系数
% p1=p/p(1,1);
% A1=p1(1,2);
% B1=p1(1,3);
% C1=p1(1,4);
% D1=p1(1,5);
% E1=p1(1,6);
% [N,M]=size(x);
% a1 = sqrt(abs(2*(A1*C1*D1-B1*C1*C1-D1*D1+4*B1*E1-A1*A1*E1)/((A1*A1-4*B1)*(B1-sqrt(A1*A1+(1-B1)*(1-B1))+1))));
% b1 = sqrt(abs(2*(A1*C1*D1-B1*C1*C1-D1*D1+4*B1*E1-A1*A1*E1)/((A1*A1-4*B1)*(B1+sqrt(A1*A1+(1-B1)*(1-B1))+1))));
% Xc1 = (2*B1*C1-A1*D1)/(A1*A1-4*B1)+1550;
% Yc1 = (2*D1-A1*D1)/(A1*A1-4*B1)+1200;
% theta1 = atan(sqrt(abs((a1*a1-b1*b1*B1)/(a1*a1*B1-b1*b1))));
% A_1 = [Yc1,Xc1,a1,b1,theta1];

%%
%代数计算法
% sum_X2Y2=0;sum_X1Y3=0;sum_X2Y1=0;sum_X1Y2=0;
% sum_X1Y1=0;sum_Y4=0;sum_Y3=0;sum_Y2=0;sum_X2=0;
% sum_X1=0;sum_Y1=0;sum_X3Y1=0;sum_X3=0;
% 
% for i = 1:N
%     sum_X2Y2 = sum_X2Y2+x(i)*x(i)*y(i)*y(i);
%     sum_X1Y3 = sum_X1Y3+x(i)*y(i)*y(i)*y(i);
%     sum_X2Y1 = sum_X2Y1+x(i)*x(i)*y(i);
%     sum_X1Y2 = sum_X1Y2+x(i)*y(i)*y(i);
%     sum_X1Y1 = sum_X1Y1+x(i)*y(i);
%     sum_Y4 = sum_Y4+y(i)*y(i)*y(i)*y(i);
%     sum_Y3 = sum_Y3+y(i)*y(i)*y(i);
%     sum_Y2 = sum_Y2+y(i)*y(i);
%     sum_X2 = sum_X2+x(i)*x(i);
%     sum_X1 = sum_X1+x(i);
%     sum_Y1 = sum_Y1+y(i);
%     sum_X3Y1 = sum_X3Y1+x(i)*x(i)*x(i)*y(i);
%     sum_X3 = sum_X3+x(i)*x(i)*x(i);
% end
% 
% M1 = [sum_X2Y2,sum_X1Y3,sum_X2Y1,sum_X1Y2,sum_X1Y1;
%      sum_X1Y3,sum_Y4,sum_X1Y2,sum_Y3,sum_Y2;
%      sum_X2Y1,sum_X1Y2,sum_X2,sum_X1Y1,sum_X1;
%      sum_X1Y2,sum_Y3,sum_X1Y1,sum_Y2,sum_Y1;
%      sum_X1Y1,sum_Y2,sum_X1,sum_Y1,N
%     ];
% M2 = [sum_X3Y1;sum_X2Y2;sum_X3;sum_X2Y1;sum_X2];
% M3 = inv(M1);
% M4 = -M3*M2;
% A = M4(1);
% B = M4(2);
% C = M4(3);
% D = M4(4);
% E = M4(5);
% 
% % hold on
% % syms x1 y1;
% % h=ezplot(x1.^2+A*x1*y1+B*y1.^2+C*x1+D*y1+E==0);
% % set(h,'Color','red')
% % hold off;
% 
% a = sqrt(abs(2*(A*C*D-B*C*C-D*D+4*B*E-A*A*E)/((A*A-4*B)*(B-sqrt(A*A+(1-B)*(1-B))+1))));
% b = sqrt(abs(2*(A*C*D-B*C*C-D*D+4*B*E-A*A*E)/((A*A-4*B)*(B+sqrt(A*A+(1-B)*(1-B))+1))));
% Xc = (2*B*C-A*D)/(A*A-4*B);
% Yc = (2*D-A*D)/(A*A-4*B);
% theta = atan(sqrt(abs((a*a-b*b*B)/(a*a*B-b*b))));
% A_0 = [Yc,Xc,a,b,theta];
% 
% Y=Yc+1200;
% X=Xc+1550;
% A_=[Y,X];
% hold on
% plot( Y,X,'*');
% 
% 
