% Copyright (C) 2022-2024 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% Copyright (C) 2024 <Su Copyright AQUÍ>

% Logistic regression testbench


% Obtener todas las comb posibles de 2 números de ese conjunto
clc; clear all; close all;
set(0, 'DefaultAxesFontSize', 32); % Make size labels bigger
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.25);
part_four = true;
part_seven = true;
part_extra_points = true;
part_ten = true;

fig_count = 1;

[Xtr,Ytr,Xte,Yte,names] = loadpenguindata("sex");

Xtr = [ ones(rows(Xtr),1) Xtr];
Xte = [ ones(rows(Xte),1) Xte];
% Xtr = matrix_design(Xtr,5);
% Xte = matrix_design(Xte,5);
%% We only need the first column of Y
%% Because we are doing binary classification
%% If one penguin is female, it is 1 otherwise 0
Ytr = Ytr(:,1);
Yte = Yte(:,1);
theta0=rand(columns(Xtr),1)-0.5; ## Random starting point



str_normalizer =  "normal";
%% Normalize input values
nx = normalizer(str_normalizer);
Xtr_normal = nx.fit_transform(Xtr);
Xte_normal = nx.transform(Xte);

## Initial configuration for the optimizer
## Use 10% of the data as minibatch
opt=optimizer("method","sgd",
              "mbmode","norep",
              "minibatch",floor(rows(Xtr_normal)*0.1),
              "maxiter",599,
              "alpha",0.05);

if part_four
  printf("\n\n##### PART FOUR #####\n\n");

  figure(fig_count,"name","Loss Evolution");
  hold on;

  # test all optimization methods
  methods={"sgd","momentum", "batch"};
  for m=1:numel(methods)
    method=methods{m};
    printf("Probando método '%s'.\n",method);
    msg=sprintf(";%s;",method); ## use method in legends

    try
      opt.configure("method",method); ## Just change the method
      [ts,errs]=opt.minimize(@logreg_loss,theta0,Xtr_normal,Ytr);
      theta=ts{end};
      display(theta);

      [total_errors, percentage_error]=logreg_empirical_error(theta,
                                                              Xtr_normal,Ytr);
      printf("Training error: %d / %d (%d %%)\n",  total_errors, rows(Ytr),
                                                percentage_error);
      [total_errors, percentage_error]=logreg_empirical_error(theta,
                                                              Xte_normal,Yte);
      printf("Test error: %d / %d (%d %%)\n",  total_errors, rows(Yte), 
                                            percentage_error);
      figure(fig_count);
      plot(errs,msg,"linewidth",4);
    catch
      printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
            method,lasterror.message);
    end_try_catch
  endfor

  figure(fig_count++);
  xlabel("Iteration");
  ylabel("Loss");
  grid on;
  hold off;
endif


if part_seven
  printf("\n\n##### PART SEVEN #####\n\n");
  ## Get all possible combinations
  comb = nchoosek(1:columns(Xtr_normal), 2);

  ## Initial configuration for the optimizer
  opt.configure("method","sgd");
  opt.configure("mbmode","norep");
  opt.configure("minibatch",floor(rows(Xtr_normal)*0.1));
  opt.configure("maxiter",599);
  opt.configure("alpha",0.05);

  ## Test all possible combinations
  list_loss = [];
  list_total_errors = [];
  list_percentage_error = [];
  list_theta = [];

  for i=1:rows(comb)
    %% Get the current combination of features
    Xtr_temp = Xtr_normal(:,comb(i,:));
    Xte_temp = Xte_normal(:,comb(i,:));
    theta0_temp = theta0(comb(i,:));

    printf("Combination: ");
    display(comb(i,:));

    try
      [ts,errs]=opt.minimize(@logreg_loss,theta0_temp,Xtr_temp,Ytr);
      theta=ts{end};

      [total_errors, percentage_error]=logreg_empirical_error(theta,
                                                              Xte_temp,Yte);
      list_loss = [list_loss errs(end)];
      list_total_errors = [list_total_errors total_errors];
      list_percentage_error = [list_percentage_error percentage_error];
      list_theta = [list_theta theta];
      printf("Test error: %d / %d (%d %%)\n",  total_errors, rows(Yte), 
                                            percentage_error);
    
    catch
      printf("\n### Error testing combination: \t");
      display(comb(i,:));
      printf("\t: ###\n %s\n\n", lasterror.message);
    end_try_catch
  endfor


  %% Find the features with least error
  min_percentage_error = min(list_percentage_error);
  index_min = find(list_percentage_error==min_percentage_error);
  index_min = index_min(:);
  assert(columns(index_min) == 1);

  labels_features = {"Bias", "Culmen Length (mm)", "Culmen Depth (mm)", ...
   "Flipper Length (mm)", "Body Mass (g)"};

  printf("\n\nBest combination are: \n\n");
  for i = 1:rows(index_min)
    printf("Combination: \n");
    display(comb(index_min(i),:));
    printf("[%s, %s]\n", labels_features{comb(index_min(i),1)},
           labels_features{comb(index_min(i),2)});
    printf("percentage error: %d %% \n###########\n\n", 
    list_percentage_error(index_min(i)));
  endfor

  %% Use the first combination with least error
  Xtr_temp = Xtr(:,comb(index_min(1),:));
  Xte_temp = Xtr(:,comb(index_min(1),:));
  best_theta = list_theta(:,index_min(1));

  %% Plot the features
  feature1_range = linspace(min(Xte_temp(:,1)), max(Xte_temp(:,1)), 100);
  feature2_range = linspace(min(Xte_temp(:,2)), max(Xte_temp(:,2)), 100);
  [feature1_grid, feature2_grid] = meshgrid(feature1_range, feature2_range);

  %% Create the design matrix for the grid
  X = [feature1_grid(:), feature2_grid(:)];

  %% Normalize the features
  %% Note: I can use the same normalizer nx but is the same cut the 
  %% features an then use a new normalizer and transform them. 
  %% It will generate the same results
  nx_temp = normalizer(str_normalizer);
  Xtr_temp_normal = nx_temp.fit_transform(Xtr_temp);
  Xte_temp_normal = nx_temp.transform(Xte_temp);
  X_normal = nx_temp.transform(X);

  %% Calculate the probabilities
  h = logreg_hyp(best_theta, X_normal);

  %% Reshape the probabilities to match the grid
  probabilities_grid = reshape(h, size(feature1_grid));

  %% Plot the surface
  figure(fig_count++, "name", "2 Most Important Feature Combination");
  surf(feature1_grid, feature2_grid, probabilities_grid);
  xlabel(sprintf('%s', labels_features{comb(index_min(1),1)}));
  ylabel(sprintf('%s', labels_features{comb(index_min(1),2)}));
  zlabel('p(y=1|x)');
  title(sprintf('Probability of Penguin being female'));
  colorbar;

  %% Plot the test points and decision boundary

  % Plotting test points differentiated by classes
  figure(fig_count++, "name", "Test Points and Decision Boundary"); 
  hold on;
  colors = {[0, 0, 1], [1, 0, 0]};
  labels = {'Male', 'Female'};
  for i = 0:1
      scatter(Xte(Yte == i, comb(index_min(1),1)), 
              Xte(Yte == i, comb(index_min(1),2)), 
              100,colors{i+1},'filled',
              'DisplayName', sprintf('%s',labels{i+1}));
  endfor

  % Calculating the decision boundary
  contour(  feature1_grid, feature2_grid, probabilities_grid, [0.5, 0.5], 
            'k', 'LineWidth', 2, "DisplayName", "Decision Boundary");
  xlabel(sprintf('%s', labels_features{comb(index_min(1),1)}));
  ylabel(sprintf('%s', labels_features{comb(index_min(1),2)}));
  title('Test Points and Decision Boundary');
  legend('show');
  hold off;


  if part_extra_points
    printf("\n\n##### PART EXTRA POINTS #####\n\n");
    %% Normalize the features
    %% Note: I can use the same normalizer nx but is the same cut the 
    %% features an then use a new normalizer and transform them. 
    %% It will generate the same results
    order = 12;
    Xtr_extended = matrix_design(Xtr_temp, order);
    X_extended = matrix_design(X, order);

    %% Normalize the features
    nx_extended = normalizer(str_normalizer);
    Xtr_extended_normal = nx_extended.fit_transform(Xtr_extended);
    X_extended_normal = nx_extended.transform(X_extended);
    theta0_extended = rand(columns(Xtr_extended),1)-0.5; ## Random starting point

    %% Minimize the loss function
    printf("Calculating the new extended theta polinomial order %d\n", order);
    [ts, ~] = opt.minimize(@logreg_loss, theta0_extended, Xtr_extended_normal, Ytr);
    extended_theta = ts{end};

    %% Calculate the probabilities
    h_extended = logreg_hyp(extended_theta, X_extended_normal);

    %% Reshape the probabilities to match the grid
    probabilities_grid_extended = reshape(h_extended, size(feature1_grid));

    %% Plot the surface
    figure(fig_count++, "name", sprintf("2 Most Important Feature Combination Polinomial Order %d", order));
    surf(feature1_grid, feature2_grid, probabilities_grid_extended);
    xlabel(sprintf('%s', labels_features{comb(index_min(1),1)}));
    ylabel(sprintf('%s', labels_features{comb(index_min(1),2)}));
    zlabel('p(y=1|x)');
    title(sprintf('Probability of Penguin being female Polinomial Order %d', 
              order));
    colorbar;

    %% Plot the test points and decision boundary

    % Plotting test points differentiated by classes
    figure(fig_count++, "name", ...
            sprintf("Test Points and Decision Boundary Polinomial Order %d", 
            order)); 
    hold on;
    colors = {[0, 0, 1], [1, 0, 0]};
    labels = {'Male', 'Female'};
    for i = 0:1
    scatter(Xte(Yte == i, comb(index_min(1),1)), 
            Xte(Yte == i, comb(index_min(1),2)), 
            100,colors{i+1},'filled',
            'DisplayName', sprintf('%s',labels{i+1}));
    endfor

    % Calculating the decision boundary
    contour(  feature1_grid, feature2_grid, probabilities_grid_extended, [0.5, 0.5], 
          'k', 'LineWidth', 2, "DisplayName", "Decision Boundary");
    xlabel(sprintf('%s', labels_features{comb(index_min(1),1)}));
    ylabel(sprintf('%s', labels_features{comb(index_min(1),2)}));
    title(sprintf('Test Points and Decision Boundary Polinomial Order %d',
                 order));
    legend('show');
    hold off;
  endif

endif


if part_ten
  printf("\n\n##### PART TEN #####\n\n");

 
  comb = nchoosek(1:columns(Xtr_normal), 3);

  ## Initial configuration for the optimizer
  opt.configure("method","sgd");
  opt.configure("mbmode","norep");
  opt.configure("minibatch",floor(rows(Xtr_normal)*0.1));
  opt.configure("maxiter",599);
  opt.configure("alpha",0.05);

  ## Test all possible combinations
  list_loss = [];
  list_total_errors = [];
  list_percentage_error = [];
  list_theta = [];

  for i=1:rows(comb)
    %% Get the current combination of features
    Xtr_temp = Xtr_normal(:,comb(i,:));
    Xte_temp = Xte_normal(:,comb(i,:));
    theta0_temp = theta0(comb(i,:));

    printf("Training Combination: ");
    display(comb(i,:));

    try
      [ts,errs]=opt.minimize(@logreg_loss,theta0_temp,Xtr_temp,Ytr);
      theta=ts{end};

      [total_errors, percentage_error]=logreg_empirical_error(theta,
                                                              Xte_temp,Yte);
      list_loss = [list_loss errs(end)];
      list_total_errors = [list_total_errors total_errors];
      list_percentage_error = [list_percentage_error percentage_error];
      list_theta = [list_theta theta];
    
    catch
      printf("\n### Error testing combination: \t");
      display(comb(i,:));
      printf("\t: ###\n %s\n\n", lasterror.message);
    end_try_catchfigure(fig_count,"name","Loss Evolution");
    hold on;
    end_try_catch
  endfor


  %% Find the features with least error
  min_percentage_error = min(list_percentage_error);
  index_min = find(list_percentage_error==min_percentage_error);
  index_min = index_min(:);
  assert(columns(index_min) == 1);

  labels_features = {"Bias", "Culmen Length (mm)", "Culmen Depth (mm)", ...
   "Flipper Length (mm)", "Body Mass (g)"};

  printf("\n\nThe most important feature combination are: \n\n");
  printf("Combination: \n");
  display(comb(index_min(1),:));
  printf( "[%s, %s, %s]\n", 
          labels_features{comb(index_min(1),1)},
          labels_features{comb(index_min(1),2)},
          labels_features{comb(index_min(1),3)});
  printf("percentage error: %d %% \n###########\n\n", 
  list_percentage_error(index_min(1)));

  %% Use the first combination with least error
  Xtr_temp = Xtr_normal(:, comb(index_min(1),:));
  theta0_temp = theta0(comb(index_min(1),:));


  figure(fig_count,"name","Loss Evolution 3 Most Important Features");
  hold on;

  # test all optimization methods
  methods={"sgd","momentum", "batch"};
  for m=1:numel(methods)
    method=methods{m};
    printf("Probando método '%s'.\n",method);
    msg=sprintf(";%s;",method); ## use method in legends

    try
      opt.configure("method",method); ## Just change the method
      [ts,~]=opt.minimize(@logreg_loss,theta0_temp,Xtr_temp,Ytr);
      theta=ts{end};
      
      % Take the components of theta
      t1 = cellfun(@(x) x(1), ts);
      t2 = cellfun(@(x) x(2), ts);
      t3 = cellfun(@(x) x(3), ts);

      figure(fig_count);
      plot3(t1, t2, t3, 'LineWidth', 2);
    catch
      printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
            method,lasterror.message);
    end_try_catch
  endfor

  figure(fig_count++);
  legend(methods);
  xlabel(sprintf('\\theta_{%d} (%s)',comb(index_min(1),1), ...
                                labels_features{comb(index_min(1),1)}));
  ylabel(sprintf('\\theta_{%d} (%s)',comb(index_min(1),2), ...
                                labels_features{comb(index_min(1),2)}));
  zlabel(sprintf('\\theta_{%d} (%s)',comb(index_min(1),3), ...
                                labels_features{comb(index_min(1),3)}));
  grid on;
  hold off;


endif
