plot(specificity(1,:),sensitivity(1,:)); title('calcified mask ROC curve'); xlabel('False positive rate(Specificity)') ;ylabel('True postitive rate(Sensitivity)');
plot(specificity(2,:),sensitivity(2,:)); title('fibrotic mask ROC curve');xlabel('False positive rate(Specificity)') ;ylabel('True postitive rate(Sensitivity)');
plot(specificity(3,:),sensitivity(3,:)); title('necrotic mask ROC curve'); xlabel('False positive rate(Specificity)') ;ylabel('True postitive rate(Sensitivity)');
plot(specificity(4,:),sensitivity(4,:)); title('lipidic mask ROC curve'); xlabel('False positive rate(Specificity)') ;ylabel('True postitive rate(Sensitivity)');

plot(1-specificity(1,:),sensitivity(1,:)); title('calcified mask ROC curve'); xlabel('False positive rate(1-Specificity)') ;ylabel('True postitive rate(Sensitivity)');
plot(1-specificity(2,:),sensitivity(2,:));title('fibrotic mask ROC curve'); xlabel('False positive rate(1-Specificity)') ;ylabel('True postitive rate(Sensitivity)');
plot(1-specificity(3,:),sensitivity(3,:)); title('necrotic mask ROC curve'); xlabel('False positive rate(1-Specificity)') ;ylabel('True postitive rate(Sensitivity)');
plot(1-specificity(4,:),sensitivity(4,:)); title('lipidic mask ROC curve'); xlabel('False positive rate(1-Specificity)') ;ylabel('True postitive rate(Sensitivity)');