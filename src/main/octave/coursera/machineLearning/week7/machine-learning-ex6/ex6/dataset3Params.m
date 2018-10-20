function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_candidates = [0.01 0.03 0.1 0.3 1 3 10 30] ;
sigma_candidates = [0.01 0.03 0.1 0.3 1 3 10 30];

C = C_candidates(1);
sigma = sigma_candidates(1);

min_prediction_error = realmax;

for i = 1:length(C_candidates)
    for j = 1:length(sigma_candidates)
        C_try = C_candidates(i);
        sigma_try = sigma_candidates(j);
        model = svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        if prediction_error<min_prediction_error
            C = C_try;
            sigma = sigma_try;
            min_prediction_error = prediction_error;
        end
    end
end

% =========================================================================

end
