# Bivariate HAR($p$) model with COVID-19
 
[Modeling and forecasting the COVID-19 pandemic with heterogeneous autoregression approaches: South Korea (2021)](https://www.sciencedirect.com/science/article/pii/S2211379721007233?via%3Dihub)

국내 COVID-19 확진자 수를 예측하기 위한 모형을 제안한다.  
해당 모형에 대한 코드를 정리한다.  

## 목차
1. [모형 정의](#1-모형-정의)
2. [모의 실험](#2-모의-실험)
3. [실증분석 EDA](#3-실증분석-eda)
4. [실증분석 결과](#4-실증분석-결과)

## 1. 모형 정의

`Heterogeneous autoregressive (HAR)` 모형은 금융변동성의 장기기억성 특성을 분석하기 위해 Corsi (2009)에 의해 제안되었다.  
이후 Corsi는 Muller의 이질적 시장가설(1993)과 Andersen과 Bollerslev의 실현변동성(1998)을 이용하여 일별, 주별, 월별의 차이에 따른 
이질성을 반영하는 `HAR-RV (Realized volatility)`을 미래변동성 예측 모형으로 제안하였다(2009).

> 참고문헌  
-[Corsi, F. (2009), A Simple Approximate Long-Memory Model of Realized Volatility, Journal of Financial Econometrics, 7, 2.](https://doi.org/10.1093/jjfinec/nbp001)  
-[Muller, U. A., Dacorogna, M. M., Dave, R. D., Pictet, O. V., Olsen, R. B., Ward, J. R. (1993), Fractals and intrinsic time: a challenge to econometricians.](https://ssrn.com/abstract=5370)  
-[Andersen, T. G., Bollerslev, T. (1998), Answering the Skeptics: YES, Standard Volatility Models do Provide Accurate Forecasts, International Economic Review, 39, 885–905.](https://doi.org/10.2307/2527343)  

실현변동성 RV에 대해 HAR-RV 모형은 다음과 같다.

$$
RV_t = \alpha_0 + \alpha_d RV_{t-1} + \alpha_w RV_{t-5:t-1} + \alpha_m RV_{t-22:t-1} + \varepsilon_t
$$

여기서 $RV_{t-5:t-1}$와 $RV_{t-22:t-1}$는 과거의 주간, 월간 평균 실현변동성이며,  
모수 $\alpha_d, \alpha_w, \alpha_m$은 각각 일별, 주별, 월별에 대한 계수이다.  

이 모형은 고정된 시간 간격의 이동평균을 회귀변수로 한 HAR(3) 모형으로 AR(22) 모형과 같다.  
따라서 일간, 주간, 월간 대신에 이동평균 기간을 다양하게 고려해 볼 수 있다.  

> HAR 모형 코드

````python
"""
    Args:
        p : [1차 이동평균, 2차 이동평균, ..., p차 이동평균]
        alpha : [alpha_1, alpha_2, ..., alpha_p]
"""

def HAR_P(p,alpha,n): # p: 이동평균 차수
    X0=list(np.random.normal(0,1,p[-1])) # 초기 값 생성
    for t in range(p[-1],n):
        order,Y=[],[]
        for i in p:
            globals()['Xt_{}'.format(i)] = np.mean(X0[t-i:t])
            order.append(globals()['Xt_{}'.format(i)])
        for i in range(len(p)):
            Xt=alpha[i]*order[i] # RV_{t-1}
            Y.append(Xt)
        eps=np.random.normal(0,1) # 오차항
        Y=sum(Y)+eps
        X0.append(Y)
    return X0
````

변수 간 상관관계를 확인하기 위한 이변량 (Bivariate) HAR 모형은 다음과 같다.  

$$
\begin{matrix}
X_t = \alpha_{11} X_{t-1}^{(1)} + \cdots + \alpha_{1p} X_{t-1}^{(p)} + \beta_{11} Y_{t-1}^{(1)} + \cdots + \beta_{1q} Y_{t-1}^{(q)} + \varepsilon_{1,t}\\
Y_t = \alpha_{21} X_{t-1}^{(1)} + \cdots + \alpha_{2p} X_{t-1}^{(p)} + \beta_{21} Y_{t-1}^{(1)} + \cdots + \beta_{2q} Y_{t-1}^{(q)} + \varepsilon_{2,t}\\
\end{matrix}
$$

$X_{t-1}^{(i)}$과 $Y_{t-1}^{(i)}$는 차수가 i인 이동평균이며, 
정상성 조건은 $\sum\limits_{i=1}^p \alpha_{ji} + \sum\limits_{k=1}^q \beta_{jk} < 1$이다.  

> 이변량 HAR 모형 코드

```python
"""
    Args:
        p : [1차 이동평균, 2차 이동평균, ..., p차 이동평균]
        alpha : [alpha_11, alpha_12, ..., alpha_1p]
"""

def BiHAR_P(p,alpha,beta,n): # p: 이동평균 차수
    X0=list(np.random.normal(0,1,p[-1])) # 초기 값 생성
    Y0=list(np.random.normal(0,1,p[-1]))
    
    for t in range(p[-1],n):
        order1,order2,X,Y=[],[],[],[]
        for i in p:
            globals()['Xt_{}'.format(i)] = np.mean(X0[t-i:t])
            order1.append(globals()['Xt_{}'.format(i)])
            globals()['Yt_{}'.format(i)] = np.mean(Y0[t-i:t])
            order2.append(globals()['Yt_{}'.format(i)])
                
        for i in range(len(p)):
            Xt=alpha[0][i]*order1[i]+beta[0][i]*order2[i] # RV_{t-1}
            Yt=alpha[1][i]*order1[i]+beta[1][i]*order2[i]
            X.append(Xt)
            Y.append(Yt)
        eps=np.random.normal(0,1,2) # 오차항
        X=sum(X)+eps[0]
        Y=sum(Y)+eps[1]
        X0.append(X)
        Y0.append(Y)     
    return X0,Y0
```

이후 실제 데이터가 정상성을 만족하도록 전처리한 후 모수를 추정한다.

## 2. 모의 실험

모수 추정 방법은 `최소제곱법(Ordinary least squares; OLS)`을 이용한다.  
