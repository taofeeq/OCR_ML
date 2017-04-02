function difference = check_finit_dif(cnet,num,Ip,d,order) 
%check_finit_dif calulate gradient or Hessian using finite differences
%
%  Syntax
%  
%    difference = check_finit_dif(cnet,num,Ip,d,order) 
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    num - number of parameter (weight or bias) in single-column weight
%    vector, which for gradient has to be calculated
%    Ip - cell array of input images
%    d - desired output
%   
%    
%   Output:
%    difference - gradient for a given parameter calculated using finite
%    difference
%   
%   Description:
%    This function is mostly used for debugging, because calculating
%    gradients such was is computationally expensive

%Epsilon determines the accuracy of this method
epsilon = 10^-8;        

        %Create an empty array
        dW = sparse(zeros(20691,1));
        %Set the given parameter to epsilon
        dW(num) = epsilon;
        %Apply this to cnet
        cnet_minus_e = adapt_dw(cnet,dW);
        cnet_plus_e = adapt_dw(cnet,-dW);
        %Simulate it with different error values
        e1 = sim(cnet_plus_e,Ip)-d;
        e2 = sim(cnet_minus_e,Ip)-d;
        %Calculate finite difference
        dEdWi = (mse(e1)-mse(e2))/(2*epsilon);
        difference = dEdWi;
   
   
end