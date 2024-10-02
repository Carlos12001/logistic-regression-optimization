function M = matrix_design(X, O=1)
  [n, variables] = size(X);
  
  % Generate all possible combinations
  combs = cell(1, variables);
  for i = 1:variables
    combs{i} = 0:O;
  end
  [combs{:}] = ndgrid(combs{:});
  for i = 1:variables
    combs{i} = combs{i}(:);
  end
  powers = [combs{:}];
  
  % Filter out the combinations greater than order O
  powers = powers(sum(powers, 2) <= O, :);
  
  % Make the matrix
  M = ones(n, size(powers, 1));
  for i = 1:size(powers, 1)
    for j = 1:variables
      M(:, i) = M(:, i) .* (X(:, j) .^ powers(i, j));
    end
  end
end
