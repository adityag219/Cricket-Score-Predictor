% Load cricket match data (replace 'cricket_data.csv' with  dataset)
data = readtable('cricket_data.csv');

% Preprocess data (select relevant features and target)
X = data(:, {'BattingTeam', 'BowlingTeam', 'Venue', 'Weather'});
y = data.Score;

% Convert categorical variables to dummy variables
X = table2array(X);
X = [X dummyvar(X(:,1))(:, 2:end) dummyvar(X(:,2))(:, 2:end) dummyvar(X(:,3))(:, 2:end) dummyvar(X(:,4))(:, 2:end)];

% Split data into training and testing sets
rng(42); % For reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test = X(test(cv), :);
y_test = y(test(cv));

% Train linear regression model
mdl = fitlm(X_train, y_train);

% Make predictions on test set
y_pred = predict(mdl, X_test);

% Evaluate model performance
mse = mean((y_pred - y_test).^2);
fprintf('Mean Squared Error: %.2f\n', mse);
