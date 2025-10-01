clc; clear;

%% === 1. DOSYA YOLLARI ===
trainFile = 'C:\ecg\ECG5000\ECG5000_TRAIN.txt';
testFile  = 'C:\ecg\ECG5000\ECG5000_TEST.txt';
modelSavePath = 'C:\ecg\ECG5000\ecg_model_SVM.mat';  % ← MODEL BURAYA KAYDEDİLECEK

%% === 2. VERİYİ YÜKLE ===
fprintf("📥 Veri yükleniyor...\n");
trainData = readmatrix(trainFile);
testData  = readmatrix(testFile);

X_raw = [trainData(:, 2:end); testData(:, 2:end)];
Y_raw = [trainData(:, 1);     testData(:, 1)];

% Binary sınıflandırma: 1 = Normal, 2 = Anormal
Y = Y_raw;
Y(Y ~= 1) = 2;

fprintf("✅ %d örnek yüklendi. Normal: %d | Anormal: %d\n", ...
    length(Y), sum(Y==1), sum(Y==2));

%% === 3. WAVELET ÖZELLİK ÇIKARIMI ===
fprintf("🌊 Wavelet ile özellik çıkarılıyor...\n");
waveletName = 'db4';
X_feat = [];

for i = 1:size(X_raw, 1)
    signal = X_raw(i,:);
    [c, l] = wavedec(signal, 4, waveletName);
    approx = appcoef(c, l, waveletName);
    [cd1, cd2, cd3, cd4] = detcoef(c, l, [1 2 3 4]);

    features = [
        mean(approx), std(approx), ...
        mean(cd1), std(cd1), ...
        mean(cd2), std(cd2), ...
        mean(cd3), std(cd3), ...
        mean(cd4), std(cd4)
    ];

    X_feat = [X_feat; features];
end

%% === 4. EĞİTİM VE TEST AYIRIMI ===
fprintf("📊 Eğitim ve test verisi ayrılıyor...\n");
cv = cvpartition(Y, 'HoldOut', 0.3);
XTrain = X_feat(training(cv), :); YTrain = Y(training(cv));
XTest  = X_feat(test(cv), :);    YTest  = Y(test(cv));

%% === 5. MODEL EĞİT (SVM) ===
fprintf("🤖 SVM modeli eğitiliyor...\n");
model = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear');
fprintf("✅ Eğitim tamamlandı.\n");

%% === 6. MODEL KAYDET ===
fprintf("💾 Model kaydediliyor: %s\n", modelSavePath);
save(modelSavePath, 'model');
fprintf("📁 Model başarıyla kaydedildi.\n");

%% === 7. TAHMİN & DOĞRULUK ===
YPred = predict(model, XTest);
accuracy = sum(YPred == YTest) / numel(YTest) * 100;
fprintf("🎯 Test doğruluğu: %.2f %%\n", accuracy);

%% === 8. CONFUSION MATRIX GRAFİĞİ ===
figure;
cm = confusionmat(YTest, YPred);
confusionchart(cm, {'Normal','Anormal'});
title(sprintf('ECG5000 SVM - Doğruluk: %.2f%%', accuracy));

%% === 9. ROC EĞRİSİ GRAFİĞİ ===
fprintf("📉 ROC eğrisi çiziliyor...\n");
[~, score] = predict(model, XTest);
[fpRate, tpRate, ~, auc] = perfcurve(YTest, score(:,2), 2);

figure;
plot(fpRate, tpRate, 'r-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('SVM ROC Eğrisi (AUC = %.3f)', auc));
grid on;
