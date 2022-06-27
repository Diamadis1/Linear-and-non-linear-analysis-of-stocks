%some code i guess


   % % % STOCK PLOTS

   figure()
   plot(BABA(:,:),'black');
   ylabel('Stock Price');
   xlabel('observations');
   title('Alibaba Group Holding Limited (BABA) ');
   set(gca,'FontName','Calibri','FontSize',10);
   axis tight
   
   figure()
   plot(ROKU(:,:),'black');
   ylabel('Stock Price');
   xlabel('observations');
   title('Roku, Inc. (ROKU)');
   set(gca,'FontName','Calibri','FontSize',10);
   axis tight
   
   figure()
   plot(AMD(:,:),'black');
   ylabel('Stock Price');
   xlabel('observations');
   title('Advanced Micro Devices, Inc. (AMD)');
   set(gca,'FontName','Calibri','FontSize',10);
   axis tight
   
   figure()
   plot(NVDA(:,:),'black');
   ylabel('Stock Price');
   xlabel('observations');
   title('NVIDIA Corporation (NVDA)');
   set(gca,'FontName','Calibri','FontSize',10);
   axis tight
   
   figure()
   plot(U(:,:),'black');
   ylabel('Stock Price');
   xlabel('observations');
   title('Unity Software Inc. (U)');
   set(gca,'FontName','Calibri','FontSize',10);
   axis tight

   % % LOGRETURNS PLOTS

   figure()
   LogReturnsBABA = diff(log(BABA));
   plot(LogReturnsBABA,'black');
   ylabel('LogPrices');
   xlabel('observations');
   title('LogReturnsBABA');
   axis tight
   set(gca,'FontName','Calibri','FontSize',10);
   
   figure()
   LogReturnsROKU = diff(log(ROKU));
   plot(LogReturnsROKU,'black');
   ylabel('LogPrices');
   xlabel('observations');
   title('LogReturnsROKU');
   axis tight
   set(gca,'FontName','Calibri','FontSize',10);

   figure()
   LogReturnsU = diff(log(U));
   plot(LogReturnsU,'black');
   ylabel('LogPrices');
   xlabel('observations');
   title('LogReturnsU');
   axis tight
   set(gca,'FontName','Calibri','FontSize',10);

   figure()
   LogReturnsNVDA = diff(log(NVDA));
   plot(LogReturnsNVDA,'black');
   ylabel('LogPrices');
   xlabel('observations');
   title('LogReturnsNVDA');
   axis tight
   set(gca,'FontName','Calibri','FontSize',10);

   figure()
   LogReturnsAMD = diff(log(AMD));
   plot(LogReturnsAMD,'black');
   ylabel('LogPrices');
   xlabel('observations');
   title('LogReturnsAMD');
   axis tight
   set(gca,'FontName','Calibri','FontSize',10);

   % %  AUTOCORRELATION PLOTS
   
   figure()
   autocorr(LogReturnsROKU,'NumLags',35,'NumSTD',2);
   title('Autocorrelation ROKU')
   
   figure()
   autocorr(LogReturnsBABA,'NumLags',35,'NumSTD',2);
   title('Autocorrelation BABA')

   figure()
   autocorr(LogReturnsU,'NumLags',35,'NumSTD',2);
   title('Autocorrelation U')

   figure()
   autocorr(LogReturnsNVDA,'NumLags',35,'NumSTD',2);
   title('Autocorrelation NVDA')

   figure()
   autocorr(LogReturnsAMD,'NumLags',35,'NumSTD',2);
   title('Autocorrelation AMD')

   % % PARTIAL CORRELATION PLOTS
   
   figure()
   parcorr(LogReturnsROKU,'NumLags',35,'NumSTD',2);
   title('Partial Correlation ROKU')
   
   figure()
   parcorr(LogReturnsBABA,'NumLags',35,'NumSTD',2);
   title('Partial Correlation BABA')

   figure()
   parcorr(LogReturnsU,'NumLags',35,'NumSTD',2);
   title('Partial Correlation U')

   figure()
   parcorr(LogReturnsNVDA,'NumLags',35,'NumSTD',2);
   title('Partial Correlation NVDA')

   figure()
   parcorr(LogReturnsAMD,'NumLags',35,'NumSTD',2);
   title('Partial Correlation AMD')

   % % % CROSS CORRELATION PLOTS

   %BABA
   figure()
   crosscorr(LogReturnsBABA,LogReturnsROKU)
   title('Cross Correlation (BABA,ROKU)')
   figure()
   crosscorr(LogReturnsBABA,LogReturnsU)
   title('Cross Correlation (BABA,U)')
   figure()
   crosscorr(LogReturnsBABA,LogReturnsNVDA)
   title('Cross Correlation (BABA,NVDA)')
   figure()
   crosscorr(LogReturnsBABA,LogReturnsAMD)
   title('Cross Correlation (BABA,AMD)')

   %AMD
   figure()
   crosscorr(LogReturnsAMD,LogReturnsROKU)
   title('Cross Correlation (AMD,ROKU)')
   figure()
   crosscorr(LogReturnsAMD,LogReturnsU)
   title('Cross Correlation (AMD,U)')
   figure()
   crosscorr(LogReturnsAMD,LogReturnsNVDA)
   title('Cross Correlation (AMD,NVDA)')
   figure()
   crosscorr(LogReturnsAMD,LogReturnsBABA)
   title('Cross Correlation (AMD,BABA)')

   %ROKU
   figure()
   crosscorr(LogReturnsROKU,LogReturnsAMD)
   title('Cross Correlation (ROKU,AMD)')
   figure()
   crosscorr(LogReturnsROKU,LogReturnsU)
   title('Cross Correlation (ROKU,U)')
   figure()
   crosscorr(LogReturnsROKU,LogReturnsNVDA)
   title('Cross Correlation (ROKU,NVDA)')
   figure()
   crosscorr(LogReturnsROKU,LogReturnsBABA)
   title('Cross Correlation (ROKU,BABA)')

   %NVDA
   figure()
   crosscorr(LogReturnsNVDA,LogReturnsAMD)
   title('Cross Correlation (NVDA,AMD)')
   figure()
   crosscorr(LogReturnsNVDA,LogReturnsU)
   title('Cross Correlation (NVDA,U)')
   figure()
   crosscorr(LogReturnsNVDA,LogReturnsROKU)
   title('Cross Correlation (NVDA,ROKU)')
   figure()
   crosscorr(LogReturnsNVDA,LogReturnsBABA)
   title('Cross Correlation (NVDA,BABA)')

   %U
   figure()
   crosscorr(LogReturnsU,LogReturnsAMD)
   title('Cross Correlation (U,AMD)')
   figure()
   crosscorr(LogReturnsU,LogReturnsNVDA)
   title('Cross Correlation (U,NVDA)')
   figure()
   crosscorr(LogReturnsU,LogReturnsROKU)
   title('Cross Correlation (U,ROKU)')
   figure()
   crosscorr(LogReturnsU,LogReturnsBABA)
   title('Cross Correlation (U,BABA)')


%   Recurrence plots

    [Net_ROKU] = visibilitynet (ROKU); Net_ROKU=double(Net_ROKU);
    figure;spy(Net_ROKU);
    title('ROKU RP')
    writetoPAJ(Net_ROKU,'Net_ROKU',0);
    
    [Net_AMD] = visibilitynet (AMD); Net_AMD=double(Net_AMD);
    figure;spy(Net_AMD);
    title('AMD RP')
    writetoPAJ(Net_AMD,'Net_AMD',0);
    
    [Net_NVDA] = visibilitynet (NVDA); Net_NVDA=double(Net_NVDA);
    figure;spy(Net_NVDA);
    title('NVDA RP')
    writetoPAJ(Net_NVDA,'Net_NVDA',0);
    
    [Net_BABA] = visibilitynet (BABA); Net_BABA=double(Net_BABA);
    figure;spy(Net_BABA);
    title('BABA RP')
    writetoPAJ(Net_BABA,'Net_BABA',0);
    
    [Net_U] = visibilitynet (U); Net_U=double(Net_U);
    figure;spy(Net_U);
    title('U RP')
    writetoPAJ(Net_U,'Net_U',0);

function [ output_args ] = clusteravra_met1_total(XV)

d = pdist(XV);  
Z = linkage(d,'average');
f = cophenet(Z,d)
		

xticklabels=({'AMD','BABA','NVDA','ROKU','U'});

[H,T,outperm]=dendrogram(Z, 0, 'Labels', xticklabels);
%dendrogram(Z, 0);
xtickangle(45)
end
%  Call in command window (clusteravra_met1_total(Stocksforpython'))
%  Import numeric matirx with all stocks as a variable first

xticklabels=({'AMD','BABA','NVDA','ROKU','U'});
cg= clustergram(Stocksforpython','Colormap',redbluecmap)
    set(cg,'RowLabels',xticklabels,'ColumnLabels',yticklabels,'FontSize',12)

t = array2table(Stocksforpython)
t2 = varfun(@(x) [mean(x); max(x); min(x); std(x)],t)
t1 = table2array(t2)
clusteravra_met1_total(t1')
