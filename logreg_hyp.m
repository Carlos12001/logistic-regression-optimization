% Copyright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2024 <Su Copyright AQUÍ>

% Hypothesis function used in logistic regression
function h=logreg_hyp(theta,X)
  assert(columns(theta)==1)
  assert(columns(X)==rows(theta))
  h = 1 ./ (1 + exp(-(X*theta)));
endfunction
