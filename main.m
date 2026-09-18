clear,clc;
close all
addpath(genpath(pwd))

data_name = 'AwA';

if strcmpi(data_name, 'MSRC')
    load MSRC.mat
    fprintf('>>> Perform the BNTF-AGL model for the MSRC dataset!\n')
    opt.process              =              'rowProcess';
    anchorNum                =               28;
    lamb                     =               10;
    p                        =               0.5;
    theta                    =               0.1;
    q                        =               0.7;
    gamma                    =               0.1;
    sigma                    =               10;
elseif strcmpi(data_name, 'HW')
    load HW.mat
    fprintf('>>> Perform the BNTF-AGL model for the Handwritten dataset!\n')
    opt.process              =              'rowProcess';
    anchorNum                =               80;
    lamb                     =               10;
    p                        =               1;
    theta                    =               0.1;
    q                        =               0.4;
    gamma                    =               5;
    sigma                    =               10;
elseif strcmpi(data_name, 'AwA')
    load AwA.mat
    fprintf('>>> Perform the BNTF-AGL model for the AwA dataset!\n')
    opt.process              =              'rowProcess';
    anchorNum                =               100;
    lamb                     =               100;
    p                        =               1;
    theta                    =               1;
    q                        =               0.9;
    gamma                    =               0.1;
    sigma                    =               1;
end



ts = tic;
Y_pred = BNTF_AGL(X, Y, anchorNum, lamb, p, theta, q, gamma, sigma, opt);
time = toc(ts);
result = ClusteringMeasure1(Y, Y_pred);

fprintf('======================================================================================== \n')
fprintf('Method         ACC       NMI       Purity       PRE       REC       F_score       ARI \n')
fprintf('BNTF-AGL       %.3f     %.3f     %.3f        %.3f     %.3f     %.3f         %.3f \n',...
    result);
fprintf('======================================================================================== \n')
fprintf('Perform BNTF-AGL model cost %.3f seconds.\n', time)




%% Functions

function Y_pred = BNTF_AGL(X, Y, anchorNum, lamb, p, theta, q, gamma,sigma,opt)

solverM = 'sigmaNorm';
solverQ = 'L2q';
solverJ = 'sp';
X = dataProcessing(X,opt);
num_N = length(Y);
num_C = length(unique(Y));
num_M = anchorNum;
[~, S] = FastmultiCLR(X, num_C, num_M);

S = cat(3, S{:});  S_hat = fft(S, [], 3);
num_V = size(S, 3);
eta                   =                1.1;
rho1                  =                1e-3;
rho2                  =                1e-3;
rho3                  =                1e-3;
rho4                  =                1e-6;
rho5                  =                1e-6;
alpha                 =                0.999;
rho_max               =                1e1;
maxIter               =                500;
tol                   =                1e-3;

for v = 1:num_V
    H_hat(:, :, v) = eye(num_N, num_C);
    G_hat(:, :, v) = eye(num_M, num_C);
end
H      = ifft(H_hat, [], 3);              G     = ifft(G_hat, [], 3);
Q      = zeros(size(G));                  Q_hat = fft(Q, [], 3);
Y1     = zeros(size(G));                 Y1_hat = fft(Y1, [], 3);
M      = zeros(size(H));                  M_hat = fft(M, [], 3);
Y2     = zeros(size(H));                 Y2_hat = fft(Y2, [], 3);
J      = zeros(size(H));                  J_hat = fft(J, [], 3);
Y3     = zeros(size(H));                 Y3_hat = fft(Y3, [], 3);
Omega  = zeros(size(H));              Omega_hat = fft(Omega, [], 3);
Y4     = zeros(size(H));                 Y4_hat = fft(Y4, [], 3);
Lambda = zeros(size(G));             Lambda_hat = fft(Lambda, [], 3);
Y5     = zeros(size(G));                 Y5_hat = fft(Y5, [], 3);

for iter = 1 : maxIter
    R1_old = norm(G(:) - Q(:));             R2_old = norm(H(:) - M(:));             R3_old = norm(H(:) - J(:));         
    R4_old = norm(H(:) - Omega(:));         R5_old = norm(G(:) - Lambda(:));             
    [H, H_hat] = solvingH(S_hat, G_hat, M_hat, J_hat, Omega_hat, ...
        Y2_hat, Y3_hat, Y4_hat, rho2, rho3, rho4);

    [G, G_hat] = solvingG(S_hat, H_hat, Q_hat, Lambda_hat, Y1_hat, ...
        Y5_hat, rho1, rho5);
        
    if strcmpi(solverQ, 'L2q')
        term1Q = G + Y1 / rho1;
        term2Q = theta / rho1;
        [Q, Q_hat] = q2_norm(term1Q, term2Q, q);
    elseif strcmpi(solverQ, 'L21')
        term1Q = G + Y1 / rho1;
        term2Q = theta / rho1;
        [Q, Q_hat] = q2_norm(term1Q, term2Q, 1);
    elseif strcmpi(solverQ, 'sigmaNorm')
        [Q, Q_hat] = sigma_norm(G, Y1, Q, rho1, theta, sigma);
    end
    
    if strcmpi(solverM, 'L2q')
        term1M = H + Y2 / rho2;
        term2M = gamma / (2*rho2);
        [M, M_hat] = q2_norm(term1M, term2M, q);
    elseif strcmpi(solverM, 'L21')
        term1M = H + Y2 / rho2;
        term2M = gamma / (2 * rho2);
        [M, M_hat] = q2_norm(term1M, term2M, 1);
    elseif strcmpi(solverM, 'sigmaNorm')
        [M, M_hat] = sigma_norm(H, Y2, M, rho2, gamma, sigma);
    elseif strcmpi(solverM, 'Fnorm')
        [M, M_hat] = F_norm(H, Y2, rho2, gamma);
    end

    term1J = H + Y3 / rho3;
    term2J = lamb / rho3;
    if strcmpi(solverJ,'Sp')
        [J, J_hat] = solvingJ(term1J, term2J, p);
    else
        [J, J_hat] = TNN(term1J, term2J);
    end
    
    termO = H + Y4 / rho4;
    [Omega, Omega_hat] = solvingOmega(termO);
    
    termL = G + Y5 / rho5;
    [Lambda, Lambda_hat] = solvingLambda(termL);
    
    Y1 = Y1 + rho1 * (G - Q);              Y1_hat = fft(Y1, [], 3);
    Y2 = Y2 + rho2 * (H - M);              Y2_hat = fft(Y2, [], 3);
    Y3 = Y3 + rho3 * (H - J);              Y3_hat = fft(Y3, [], 3);
    Y4 = Y4 + rho4 * (H - Omega);          Y4_hat = fft(Y4, [], 3);
    Y5 = Y5 + rho5 * (G - Lambda);         Y5_hat = fft(Y5, [], 3);
    
    
    R1_new = norm(G(:) - Q(:));             R2_new = norm(H(:) - M(:));             R3_new = norm(H(:) - J(:));         
    R4_new = norm(H(:) - Omega(:));         R5_new = norm(G(:) - Lambda(:));
    
    if (R1_new > alpha * R1_old) || (rho1 < rho_max); rho1 = eta * rho1; end
    if (R2_new > alpha * R2_old) || (rho2 < rho_max); rho2 = eta * rho2; end
    if (R3_new > alpha * R3_old) || (rho3 < rho_max); rho3 = eta * rho3; end
    if (R4_new > alpha * R4_old) || (rho4 < rho_max); rho4 = eta * rho4; end
    if (R5_new > alpha * R5_old) || (rho5 < rho_max); rho5 = eta * rho5; end
    
    
    
    if max([max(G(:) - Q(:)), max(H(:) - M(:)),...
            max(H(:) - J(:)), max(H(:) - Omega(:)),...
            max(G(:) - Lambda(:))]) <= tol
        fprintf('The BNTF-AGL model reached the stopping criteria after %d iterations.\n', iter)
        break
    end
        
    
end

[~, Y_pred] = max(sum(H, 3) / num_V, [], 2);

end


function [J, J_hat] = solvingJ(term3, term4, p)
    Y_hat = fft(shiftdim(term3, 1), [], 3);
    for v = 1 : size(Y_hat, 3)
        [uu, ss, vv] = svd(Y_hat(:, :, v), 'econ');
        ss_ = Prox_lp(diag(ss), term4, p);
        Y_hat(:, :, v) = uu * diag(ss_) * vv';
    end
    J = shiftdim(ifft(Y_hat, [], 3), 2);
    clear Y_hat
    J_hat = fft(J, [], 3);
end

function [Lambda, Lambda_hat] = solvingLambda(termL)
    termL(termL <= 0) = 0;
    Lambda = termL;
    Lambda_hat = fft(Lambda, [], 3);
end

function [Omega, Omega_hat] = solvingOmega(termO)
    termO(termO <= 0) = 0;
    Omega = termO;
    Omega_hat = fft(Omega, [], 3);
end

function [Q, Q_hat] = q2_norm(term1, term2, q)
    W = term1;
    for v = 1 : size(W, 3)
        for i = 1 : size(W, 1)
            Wn = norm(W(i, :, v)) + eps;
            sigma = Prox_lp(Wn, term2, q);
            Q(i, :, v) = sigma * W(i, :, v) / Wn;
        end
    end
    clear W Wn
    Q_hat = fft(Q, [], 3);
end

function [M, M_hat] = sigma_norm(H, Y2, M, rho2, gamma, sigma)
    Lamb = H + Y2 / rho2;
    [num_N, ~, num_V] = size(H);
    for v = 1 : num_V
        for i = 1 : num_N
            Mn = norm(M(i, :, v)) + eps;
            di(i) = (1 + sigma) * ((Mn + 2 * sigma) / (2 * (Mn + sigma)^2));
        end
        d_vec = 1./(gamma.*di + rho2 * ones(1, num_N));
        M(:, :, v) = d_vec(:) .* (rho2 * Lamb(:, :, v));
    end
    clear D Mn Lamb
    M_hat = fft(M, [], 3);
end


function [G, G_hat] = solvingG(S_hat, H_hat, Q_hat, Lambda_hat, Y1_hat, Y5_hat, rho1, rho5)
    for v = 1:size(S_hat, 3)
        term=2 * S_hat(:, :, v)' * H_hat(:, :, v) + rho1 * Q_hat(:, :, v) +...
         rho5 * Lambda_hat(:, :, v) - Y1_hat(:, :, v) - Y5_hat(:, :, v);
        [uu, ~, vv] = svd(term, 'econ');
        G_hat(:, :, v) = uu * vv';
        clear uu vv term
    end
    G = ifft(G_hat, [], 3);    
    if min([rho1, rho5]) < 1e0; G(G < 0) = 0; G_hat = fft(G, [], 3);end
end

function [H, H_hat] = solvingH(S_hat, G_hat, M_hat, J_hat, Omega_hat, Y2_hat, Y3_hat, Y4_hat, rho2, rho3, rho4)
    for v = 1 : size(S_hat, 3)
        term = 2 * S_hat(:, :, v) * G_hat(:, :, v) + rho2 * M_hat(:, :, v) +...
         rho3 * J_hat(:, :, v) + rho4 * Omega_hat(:, :, v) - Y2_hat(:, :, v) -...
          Y3_hat(:, :, v) - Y4_hat(:, :, v);
        [uu, ~, vv] = svd(term, 'econ');
        H_hat(:, :, v) = uu * vv';
    end
    clear uu vv A
    H = ifft(H_hat, [], 3);
    if min([rho2, rho3, rho4]) < 1e0; H(H < 0) = 0; H_hat = fft(H, [], 3); end
    
end
