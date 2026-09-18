function X = dataProcessing(X,opt)
    if strcmp('maxProcess', opt.process)
        for v = 1:length(X); X{v} = double(X{v}./(max(X{v}(:)))); end
    elseif strcmp('rowProcess', opt.process)
        % fprintf('>>> Data processing: Row processing...\n')
        for v = 1:length(X)
            Xv = X{v};
            for row = 1:size(Xv,1); Xv(row,:) = double(Xv(row,:)./(norm(Xv(row,:),'fro'))); end
            X{v} = Xv;
        end
    elseif strcmp('colProcess', opt.process)
        fprintf('>>> Data processing: Column processing...\n')
        for v = 1:length(X)
            Xv = X{v};
            for col = 1:size(Xv,2); Xv(:,col) = double(Xv(:,col)./(norm(Xv(:,col),'fro'))); end
            X{v}=Xv;
        end
    else
        fprintf('>>> Data processing: None\n')
    end
    % fprintf('>>> Data processing completed! \n')
end
