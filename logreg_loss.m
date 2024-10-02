% Copyright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2024 <Su Copyright AQUÍ>

% Loss function used in logistic regression
function err=logreg_loss(theta,X,y)
  assert(rows(y)==rows(X))

  ## residuals
  r=y-logreg_hyp(theta,X);
  err=1/rows(y)*(r'*r); # OLS
endfunction
