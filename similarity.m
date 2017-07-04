dice=zeros(4,412,'single');
jaccard_idx=zeros(4,412,'single');
jaccard_dis=zeros(4,412,'single');

a=sensitivity+specificity;
[cutvaloff,index]=max(a,[],2);
index=index*0.005;

 for i=1:412      
    k=i*16384;
    start=k-16383;
    
    label=cacified(start:k) >= index(1);
    score=calscore(start:k) >= index(1);     
    dice(1,i)=2*nnz(label&score)/(nnz(label)+nnz(score));
    inter=label & score;
    union=label |score;
    jaccard_idx(1,i)=sum(inter(:))/sum(union(:));
    jaccard_dis(1,i)=1-jaccard_idx(1,i);
    
    label=fibrotic(start:k) >= index(2);
    score=fibscore(start:k) >= index(2);     
    dice(2,i)=2*nnz(label&score)/(nnz(label)+nnz(score));
    inter=label & score;
    union=label |score;
    jaccard_idx(2,i)=sum(inter(:))/sum(union(:));
    jaccard_dis(2,i)=1-jaccard_idx(2,i);
    
    label=necrotic(start:k) >= index(3);
    score=necscore(start:k) >= index(3);     
    dice(3,i)=2*nnz(label&score)/(nnz(label)+nnz(score));
    inter=label & score;
    union=label |score;
    jaccard_idx(3,i)=sum(inter(:))/sum(union(:));
    jaccard_dis(3,i)=1-jaccard_idx(3,i);
    
    label=lipidic(start:k) >= index(4);
    score=lipscore(start:k) >= index(4);     
    dice(4,i)=2*nnz(label&score)/(nnz(label)+nnz(score));
    inter=label & score;
    union=label |score;
    jaccard_idx(4,i)=sum(inter(:))/sum(union(:));
    jaccard_dis(4,i)=1-jaccard_idx(4,i);
    
    
 end

 DICE=mean(dice,2);
 JACCARD=mean(jaccard_idx,2);
 
 