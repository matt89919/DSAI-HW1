# DSAI_HW1

## approch1 (F74082191)

### 想法
一開始其實是想使用svm來實作，但過程不太順利，剛好看到網路上有人做了股票趨勢分析是使用KNN來實作，加上之前有用過比較不陌生就來實作了。

基於最鄰近演算法的分類，本質上是對離散的資料標籤進行預測，實際上，最鄰近演算法也可以用於對連續的資料標籤進行預測，這種方法叫做基於最鄰近資料的迴歸，預測的值（即資料的標籤）是連續值，通過計算資料點最臨近資料點平均值而獲得預測值。

### data collection
Collect the '日期' from the open data, the time interval is between 2020/01/01 and 2021/04/30 
再從日期分出月份和星期幾
原因：因為發現隨著月份的不同，備轉容量的最大值最小值有週期性的變化（例如夏季普遍較高，冬季較低）
而星期幾是因為有可能碰到需要大量用電的產業那天恰好停工
原本還有取前一星期的備轉容量以及月均溫，但出來效果沒有很好。

### model
KNN(K Nearest Neighbors)屬於分類演算法的一種，處理原則也非常單純。一開始先儲存所有的變數資料，預測的作法為：計算所有資料點到x的距離，隨著遠離x的距離開始分類資料，最後再預測k值範圍內最主要的標籤。優點在於非常簡單，適用於任何類別，需要的參數少。缺點在於高預測費、不適用多維度資料及類別變數。


## approach2 (F74082060)

### 想法
備轉容量看起來好像有一個週期性的波動，選擇使用LSTM model來預測

### data collection
原先因為對ML或AI沒有什麼概念，大部分的時間都花在學習怎麼去使用這個model以及處理data上面，因此只使用了[備轉容量]來當作feature訓練model
後來對model較熟悉後嘗試加入月份與日期當作feature，結果卻沒有變好，test出來的RMSE反而大於一開始的做法，因此最後只保留一個feature

### model
LSTM有四種model，分別是一對多，多對一，一對一，多對多
一對多跟多對一感覺起來好像比較沒有顧慮到週期性，因此我只測試了多對一跟多對多

多對一使用前7天預測後1天，出來的結果誤差偏大，而且因為作業要預測15天就沒有繼續改良這個方法

多對多因為考量到輸出格式與data要求，選擇使用前15天預測後15天，沒想到結果竟然比之前的多對一結果好上不少，就採用了這個方案

### conclusion
寫完之後比較了我們兩個不同做法在測試資料上的誤差，感覺approch1是比較好的做法，因此選擇繳交approch1
感覺我對於ML方面的domain knowledge有點不足夠，因此做出來的結果偏奇怪，畫出來也沒辦法抓到整個資料的trend
也不知道該怎麼調整model的參數，可能要再花一些時間把基礎補好
