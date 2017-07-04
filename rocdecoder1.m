cacified=groundTruth(:,:,1,1);
normC = cacified - min(cacified(:));
cacified=normC ./ max(normC(:));
cacified=reshape(cacified,[1,16384]);
cacified=single(cacified);

fibrotic=groundTruth(:,:,2,1);
normC = fibrotic - min(fibrotic(:));
fibrotic=normC ./ max(normC(:));
fibrotic=reshape(fibrotic,[1,16384]);
fibrotic=single(fibrotic);

lipidic=groundTruth(:,:,3,1);
lipidic=reshape(lipidic,[1,16384]);
normC = lipidic - min(lipidic(:));
lipidic=normC ./ max(normC(:));
lipidic=single(lipidic);

necrotic=groundTruth(:,:,4,1);
normC = necrotic - min(necrotic(:));
necrotic=normC ./ max(normC(:));
necrotic=reshape(necrotic,[1,16384]);
necrotic=single(necrotic);

calscore=results(:,:,1,1);
calscore=reshape(calscore,[1,16384]);
normC = calscore - min(calscore(:));
calscore=normC ./ max(normC(:));
calscore=single(calscore);

fibscore=results(:,:,2,1);
normC = calscore - min(calscore(:));
calscore=normC ./ max(normC(:));
fibscore=reshape(fibscore,[1,16384]);
fibscore=single(fibscore);

lipscore=results(:,:,3,1);
normC = lipscore - min(lipscore(:));
lipscore=normC ./ max(normC(:));
lipscore=reshape(lipscore,[1,16384]);
lipscore=single(lipscore);

necscore=results(:,:,4,1);
normC = necscore - min(necscore(:));
necscore=normC ./ max(normC(:));
necscore=reshape(necscore,[1,16384]);
necscore=single(necscore);

for y=2:412

    label1=groundTruth(:,:,1,y);
    normC = label1 - min(label1(:));
    label1=normC ./ max(normC(:));
    label1=reshape(label1,[1,16384]);
    label1=single(label1);
    cacified=horzcat(cacified,label1);
    
    label2=groundTruth(:,:,2,y);
    normC = label2 - min(label2(:));
    label2=normC ./ max(normC(:));
    label2=reshape(label2,[1,16384]);
    label2=single(label2);
    fibrotic=horzcat(fibrotic,label2);
    
    label3=groundTruth(:,:,3,y);
    normC = label3 - min(label3(:));
    label3=normC ./ max(normC(:));
    label3=reshape(label3,[1,16384]);
    label3=single(label3);
    lipidic=horzcat(lipidic,label3);
    
    label4=groundTruth(:,:,4,y);
    normC = label4 - min(label4(:));
    label4=normC ./ max(normC(:));
    label4=reshape(label4,[1,16384]);
    label4=single(label4);
    necrotic=horzcat(necrotic,label4);
    
    score1=results(:,:,1,y);
    normC = score1 - min(score1(:));
    score1=normC ./ max(normC(:));
    score1=reshape(score1,[1,16384]);
    score1=single(score1);
    calscore=horzcat(calscore,score1);
    
    score2=results(:,:,2,y);
    normC = score2 - min(score2(:));
    score2=normC ./ max(normC(:));
    score2=reshape(score2,[1,16384]);
    score2=single(score2);
    fibscore=horzcat(fibscore,score2);
    
    score3=results(:,:,3,y);
    normC = score3 - min(score3(:));
    score3=normC ./ max(normC(:));
    score3=reshape(score3,[1,16384]);
    score3=single(score3);
    lipscore=horzcat(lipscore,score3);
    
    score4=results(:,:,4,y);
    normC = score4 - min(score4(:));
    score4=normC ./ max(normC(:));
    score4=reshape(score4,[1,16384]);
    score4=single(score4);
    necscore=horzcat(necscore,score4);

end

sensitivity=zeros(4,20,'single');
specificity=zeros(4,20,'single');
Accuracy=zeros(4,20,'single');
tp=zeros(4,20,'single');
tn=zeros(4,20,'single');
fp=zeros(4,20,'single');
fn=zeros(4,20,'single');

n=6750208;

for cut=1:20
    cutval=cut*0.05;
    xcal=cacified >= cutval;
    ycal=calscore >= cutval;
    xfib=fibrotic >= cutval;
    yfib=fibscore >= cutval;
    xnec=necrotic >= cutval;
    ynec=necscore >= cutval;
    xlip=lipidic >= cutval;
    ylip=lipscore >=cutval;

    
    for i=1:n
        if xcal(i)==0 && ycal(i)==0
        tn(1,cut) = tn(1,cut)+1;
        elseif xcal(i)==1 && ycal(i)==1
        tp(1,cut) = tp(1,cut)+1;
        elseif xcal(i)==1 && ycal(i)==0
        fn(1,cut) = fn(1,cut)+1;
        elseif xcal(i)==0 && ycal(i)==1
        fp(1,cut) = fp(1,cut)+1;    
        end    
        
        if xfib(i)==0 && yfib(i)==0
        tn(2,cut) = tn(2,cut)+1;
        elseif xfib(i)==1 && yfib(i)==1
        tp(2,cut) = tp(2,cut)+1;
        elseif xfib(i)==1 && yfib(i)==0
        fn(2,cut) = fn(2,cut)+1;
        elseif xfib(i)==0 && yfib(i)==1
        fp(2,cut) = fp(2,cut)+1;    
        end    
        
        if xlip(i)==0 && ylip(i)==0
        tn(3,cut) = tn(3,cut)+1;
        elseif xlip(i)==1 && ylip(i)==1
        tp(3,cut) = tp(3,cut)+1;
        elseif xlip(i)==1 && ylip(i)==0
        fn(3,cut) = fn(3,cut)+1;
        elseif xlip(i)==0 && ylip(i)==1
        fp(3,cut) = fp(3,cut)+1;    
        end    
        
        if xnec(i)==0 && ynec(i)==0
        tn(4,cut) = tn(4,cut)+1;
        elseif xnec(i)==1 && ynec(i)==1
        tp(4,cut) = tp(4,cut)+1;
        elseif xnec(i)==1 && ynec(i)==0
        fn(4,cut) = fn(4,cut)+1;
        elseif xnec(i)==0 && ynec(i)==1
        fp(4,cut) = fp(4,cut)+1;    
        end    
    end
    
    for k=1:4
        sensitivity(k,cut)=tp(k,cut)/(tp(k,cut)+fn(k,cut));
        specificity(k,cut)=tn(k,cut)/(fp(k,cut)+tn(k,cut));
        
        Accuracy(k,cut)=(tp(k,cut) +tn(k,cut))/ (tp(k,cut)+fp(k,cut) +tn(k,cut) +fn(k,cut));
    end
end

%plot(1-specificity,sensitivity);
 