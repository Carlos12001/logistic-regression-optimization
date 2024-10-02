% Copyright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
%
% Linear regression as testbench for optimizer

clear all; close all;
set(0, 'DefaultAxesFontSize', 32); % Para el texto dentro de los ejes
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.25);

% Generate some artificial data:
x = 1.5*rand(30,1)-0.5; ## random data between -0.5 and 1
X = bsxfun(@power,x,0:2);
y = x.^2 + 0.05*rand(30,1); ## add noise

% Show the data
figure(1,"name","optimizer testbench");

% show the original data
plot(X(:,2),y,'x');
xlabel("x");
ylabel("y");
axis([-0.5,1,-0.1,1.1]);
grid on;
hold on;


figure(2,"name","Loss evolution");
hold on;

function h=linreg_hyp(theta,X)
  h=X*theta;
endfunction

function err=linreg_loss(theta,X,y)
  ## theta must be a column vector
  assert(columns(theta.value)==1)

  ## residuals
  r=y-linreg_hyp(theta,X);
  err=0.5*(r'*r); # OLS
endfunction

## Initial configuration for the optimizer
opt=optimizer("method","sgd",
              "minibatch",8,
              "maxiter",599,
              "alpha",0.005);

theta0=rand(columns(X),1)-0.5; ## Random starting point

## We want to predict the value at these points
px=bsxfun(@power,linspace(-0.5,1,100)',0:2);

# test all optimization methods
methods={"batch","sgd","momentum"};
% methods={"momentum"};
for m=1:numel(methods)
  method=methods{m};
  printf("Probando método '%s'.\n",method);
  msg=sprintf(";%s;",method); ## use method in legends

  try
    opt.configure("method",method); ## Just change the method
    [ts,errs]=opt.minimize(@linreg_loss,theta0,X,y);
    theta=ts{end};

    figure(1);
    py=linreg_hyp(theta,px);
    plot(px(:,2),py,msg,"linewidth",2);

    figure(2);
    plot(errs,msg,"linewidth",2);
  catch
    printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
           method,lasterror.message);
  end_try_catch
endfor

figure(2);
xlabel("Iteration");
ylabel("Loss");
grid on;
