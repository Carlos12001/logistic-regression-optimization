function [total_errors, percentage_error]= logreg_empirical_error(theta,X,y)
  assert(rows(y)==rows(X))
  h = round(logreg_hyp(theta,X));
  total_errors =  sum((y - h) != 0);;
  percentage_error = 100*total_errors/rows(y);
endfunction