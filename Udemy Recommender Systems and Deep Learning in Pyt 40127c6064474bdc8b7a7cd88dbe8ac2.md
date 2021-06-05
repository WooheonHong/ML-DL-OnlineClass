# Udemy: Recommender Systems and Deep Learning in Python

# Simple Recommendation Systems

- non-personalized recommenders: 개인적인 (비)선호도가 사용되지 않는다

- 아마존과 유튜브는 처음 방문했더라도 인구통계학적 정보를 통해 전체 모집단 추론을 하여 추천을 해준다.

- 사용자 한명이 5점으로 평가한 제품과 5만명이 평가해서 평균이 4.2점인 제품 중에서 전자가 더 좋지는 않다. 이 점수를 얼마나 신뢰할 수 있는지를 봐야한다 —> 베이지안 패러다임으로 확장: explorer exploit 딜레마라고 알려진 문제를 다룰 뿐만 아니라 신뢰성을 자동으로 설명한다
- 지도학습으로 얻어진 예측 스코어를 통해 추천에 사용할 수 있다.
- 마르코프 모델은 구글의 페이지 랭크 알고리즘에 사용된다

- Association Rule Analysis: Y가 없는 상태에서 데이터 속에 숨겨져 있는 패턴, 규칙을 찾아내는 비지도 학습

[[R 연관규칙(Association Rule)] 지지도(support), 신뢰도(confidence), 향상도(lift), IS측도, 교차지지도](https://rfriend.tistory.com/191?category=706118)

$$Lift = \frac{P(A, B)}{P(A)P(B)} = \frac{P(A|B)}{P(A)} = \frac{P(B|A)}{P(B)}$$

독립이면 lift = 1 이다. 

'지지도(Support)'가 높아서 전체 거래 건수 중에서 해당 rule이 포함된 거래건수가 많아야지만이 해당 rule을 가지고 마케팅전략을 수립해서 실전에 적용했을 때 높은 매출 증가를 기대할 수 있게 됩니다. 즉, 아무리 신뢰도(confidence)와 향상도(lift)가 높아도 지지도(support)가 빈약해서 전체 거래 중에 가뭄에 콩나듯이 나오는 거래유형의 rule이라면 사업부 현업은 아마 무시할 겁니다. 현업을 빼고 분석가만 참여한 연관규칙 분석이 위험하거나 아니면 실효성이 떨어질 수 있는 이유입니다. 그리고 지지도(support)가 매우 낮으면 몇 개 소수이 관측치의 치우침만으로도 신뢰도나 향상도가 크게 영향을 받게 되어 '우연'에 의한 규칙이 잘못 선별될 위험도 있습니다.

출처:

[https://rfriend.tistory.com/191?category=706118](https://rfriend.tistory.com/191?category=706118)

[R, Python 분석과 프로그래밍의 친구 (by R Friend)]

## Hacker 뉴스

$$score = \frac{(ups - downs - 1)^{0.8}}{(age + 2)^{gravity}} × penalty$$

- gravity = 1.8
- penalty = 비즈니스 규칙을 규현하기 위함(즉, penalize self-posts, "controversial" posts, + many more you'll see later...)
- age + 2는 score가 무한대로 가지 않게 하기 위함
- gravity = 1.8 > 0.8이므로 분모가 분자보다 빠르게 커진다 → age가 popularity를 앞지른다.

위의 순위 공식을 링크와 코멘트에 둘 다 사용한다(Reddit)은 아님. 

[How Hacker News ranking really works: scoring, controversy, and penalties](http://www.righto.com/2013/11/how-hacker-news-ranking-really-works.html)

### How ranking works

페이지를 방문할 때 마다 순위가 매겨지는 것이라고 생각할 수 있다. 하지만 효율성을 위해,  30초마다 50개의 스토리 중 하나를 랜덤으로 선택하고 순위를 다시 매긴다. 투표를 받지 못한다면, 수 분 동안 잘못 순위가 매겨질 수 있다. 게다가, 페이지는 90초 마다 숨겨질 수 있다.

### Raw scores and the #1 spot on a typical day

![Udemy%20Recommender%20Systems%20and%20Deep%20Learning%20in%20Pyt%2040127c6064474bdc8b7a7cd88dbe8ac2/Untitled.png](Udemy%20Recommender%20Systems%20and%20Deep%20Learning%20in%20Pyt%2040127c6064474bdc8b7a7cd88dbe8ac2/Untitled.png)

위 이미지는 11월 11일 하루 동안의 top 60 HN기사의 raw 스코어(패널티 제외)를 보여준다. 각 선은 하나의 기사에 대응한다. 빨간 선은 HN의 top 기사를 나타낸다. 패널티가 없기 때문에 top raw score를 가진 기사가 top 기사가 아님에 주의하자.

초록색 삼각형은 controversy 패널티(너무 많은 코멘트가 존재)가 적용됐다. 파란색 삼각형은  "망각" 패널티. milder 패널티는 여기서 보여지진 않는다. 

## Reddit

- 레딧과 해커뉴스의 차이는 다운로드 방식이다. 해커 뉴스에서는 다운로드 하기 전에 특정 포인트를 가질 필요가 있는 반면, 레딧은 모두 같은 다운로드 파워를 가진다.

$$score = sign(ups - downs) × log\{max(1, |ups - downs|)\} + \frac{age}{45000}$$

- 첫 번째 주요 차이점은 항들이 additive라는 것이다.

첫번째 항

- net vote이 양수인지 음수인지 따라 첫번째 항의 부호가 정해진다. max는 log(0)을 피하기 위함.

두번째 항 

- 2005년 12월 8일부터 초를 잰다.
- 새로운 링크 → 더 높은 점수(선형적으로) 즉, 시간이 흐름에 따라 해커 뉴스는 스코어가 감소하지만 레딧은 점점 커진다. 그래서 상대적인 스코어가 중요하다.

결국 사이트가 커지면 정치와 돈 ..

## Problems with Average Rating & Explore vs. Exploit

5점 척도로 하면 regression, 좋아요/싫어요로 한다면 분류 모델을 생각해볼 수 있다. 

score를 단순히 평균낼 수 있다. 하지만 문제점이 있는데 평가한 사람 수를 고려하지 않는다는 점이다. 사람 수가 적으면 신뢰구간이 넓고 많으면 좁다.

[Binomial proportion confidence interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)

정규근사의 개선버전인 wilson score interval이 가장 정확하고 robust하다.

해커 뉴스는 링크와 코멘트 둘 다 동일한 정렬 알고리즘을 사용하지만 레딧은 코멘트에 wilson interval를 사용한다. 

### 평균 문제 해결

smoothing(dampening)

$$r = \frac{\sum^n_{i = 1} X_i + \lambda {\mu_0}}{N + \lambda}$$

NLP에서 단어 확률, 페이지 순위에서 볼 수 있다. $\mu_0$를 전체 평균이나 중간 점수를 사용한다. 

### Explore-Exploit Dilemma

A/B 테스트와 강화학습 관점에서 논의됨.  

슬롯 머신을 너무 적게 돌리면 추정치가 좋지 않아서 신뢰구간이 넓다. 전통적인 통계 검정을 쓸 수는 있지만 문제가 있다. 좋은 추정치를 얻기 위해선 많이 돌려야 하지만 그렇게 되면 최적이 아닌 것을 돌리는 데에 더 많은 시간을 쏟아야만 한다: explore-exploit dilemma

smoothed average가 한 가지 해결 방법이 될 수 있다. 이 문제를 베이지안으로 해결해보자. 

online learning 

사후 분포에서 샘플을 뽑는 것이 핵심 아이디어이다. 이 코스의 다른 모든 기술은 deterministic ranking을 사용하지만, 여기서는 아니다. 자동적으로 explore와 exploit 사이의 벨런스를 맞춘다. 사후분포로부터 샘플링 하는 것은 우리가 모은 데이터를 설명하기 때문에 똑똑한 랜덤이다. 

```python
# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit(object):
  def __init__(self, p):
    self.p = p
    self.a = 1
    self.b = 1

  def pull(self):
    return np.random.random() < self.p

  def sample(self):
    return np.random.beta(self.a, self.b)

  def update(self, x):
    self.a += x
    self.b += 1 - x

def plot(bandits, trial):
  x = np.linspace(0, 1, 200)
  for b in bandits:
    y = beta.pdf(x, b.a, b.b)
    plt.plot(x, y, label="real p: %.4f" % b.p)
  plt.title("Bandit distributions after %s trials" % trial)
  plt.legend()
  plt.show()

def experiment():
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
  for i in range(NUM_TRIALS):

    # take a sample from each bandit
    bestb = None
    maxsample = -1
    allsamples = [] # let's collect these just to print for debugging
    for b in bandits:
      sample = b.sample()
      allsamples.append("%.4f" % sample)
      if sample > maxsample:
        maxsample = sample
        bestb = b
    if i in sample_points:
      print("current samples: %s" % allsamples)
      plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bestb.pull()

    # update the distribution for the bandit whose arm we just pulled
    bestb.update(x)

if __name__ == "__main__":
  experiment()
```

plot을 보면 가장 큰 확률에서 신뢰구간이 좁은 것을 알 수 있는데 `if sample > maxsample`

때문에 가장 큰 샘플의 beta를 가지는 분포가 계속 뽑히기 때문에 발생한다. 

web server framework, data storage, database connector, distributed computing

## Page Rank

### Markov Models

NLP에서 bigram을 생각해보자. 

I 뒤에 나올 단어가 무엇일까?  P(love | I) = ? 

$$p(x_t | x_{t-1}, x_{t-2},...,x_1) = p(x_t|x_{t-1}) $$

**Transition Probability Matrix**

A(i, j) 는 상태 i에서 j로 가는 확률 $A(i, j) = p(x_t = j | x_{t - 1} = i)$

여기서 당연히 다음은 성립 $\sum^M_{j = 1} A(i, j) = \sum^M_{j = 1}p(x_t = j | x_{t - 1} = i) = 1$

"the quick brown fox jumps over the lazy dog?"

$p(the)p(quick|the)p(brown|quick)...$

$p(x_1,...,x_T) = p(x_1)\prod^T_{i = 2}p(x_t | x_{t - 1})$

위와 같은 방법은 문제가 있다. train set에 없는 bigram을 포함하는 문장은 어떡할까? 

—> Add-1 Smoothing 

$p(x_t= j|x_{t - 1} = i) = \frac{count(i\rightarrow j) + 1}{count(i) + V}$

여기서 각 상태는 단어이기 때문에 V =  M(number of state)로 정의한다. p(and | and)가 절대 나타나지 않더라도 양수의 확률을 가진다. 

beta 사후 평균을 떠올리면 

$$E(\pi) = \frac{\alpha'}{\alpha' + \beta'} = \frac{\alpha + \sum^N_{i = 1}X_i}{\alpha + \beta + N}$$

여기서는 단 2개의 가능한 결과가 있었지만 여기서는 V개의 가능한 결과가 있다. 

사전분포로 beta(1, 1)보다 디리클레 분포(Dirichlet(1))를 사용하는 것이 좀 더 정확하다. 

**Add-epsilon smoothing**

$$p(x_t= j|x_{t - 1} = i) = \frac{count(i\rightarrow j) + \epsilon}{count(i) + \epsilon V}$$

**State Distribution**

$\pi_t = \text{state probability distribution at time t}$

만약 2개의 상태가 "sunny"와 "rainy"라면 $\pi(t)$는 크기 2의 벡터이다. 관습적으로 $\pi(t)$는 행 벡터이다.

**Future State Distribution**

$$p(x_{t + 1} = j) = \pi_{t + 1}(j)$$

베이즈 룰을 쓰면 위와 같이 보일 수 있다. 

$$\pi_{t + 1}(j) = \sum^M_{i = 1}A(i, j)\pi_t(i) \\ \pi_{t + 1} = \pi_t A$$

$$\pi_{t + k} = \pi_tA^k$$

$$\pi_\infin = lim_{t \rightarrow \infin}\pi_0 A^t \\ \pi_\infin = \pi_\infin A$$

위 사실을 통해서 고윳값을 떠올릴 수 있다. 고윳값은 (고유)벡터의 방향을 바꾸지 않고 크기만 변화시킨다. 

### PageRank

인터넷의 모든 페이지는 마르코프 모델의 상태에 있다. 

$$p(x_t = j | x_{t - 1} = i) = 1/n(i) \text{ if i links to j}, n(i) = \text{links on page i 0 otherwise}$$

인터넷의 수십업개의 페이지가 있어서 대부분 전이 확률은 0이다. smoothing을 더하자. 

G = 0.85A + 0.15U, U(i, j) = 1/M 

**Limiting Distribution**

G의 극한 분포를 찾아라. 이 확률들은 인터넷의 각 페이지에 대한 각각의 페이지랭크이다.

극한 분포: 무한 번 곱한 후에도 같음 

정상 분포: G를 곱한 후에 변하지 않음

**Perron-Frobenius theorem**

만약 G가 마르코프 행렬이고 이 행렬의 모든 성분이 양수이면 정상 분포와 극한 분포는 동일하다. 

smoothing으로 양수를 만들 수 있다.

- 예전에는 같은 링크를 수백번 클릭했다: 한번 클릭하는 것과 동일하게
- 페이지를 뒤로 가게 하는 수많은 더미 웹사이트를 만듬: SEO기술 다음 페이지로 가는 링크의 파워는 감소 (뭔소리야)

['쉽게 설명한' 구글의 페이지 랭크 알고리즘](https://sungmooncho.com/2012/08/26/pagerank/)

[PageRank](http://www.secmem.org/blog/2019/07/21/pagerank/)

## Evaluating a Ranking

# Collaborative Filtering

[협업 필터링 (Collaborative filtering) > 도리의 디지털라이프](http://blog.skby.net/%ED%98%91%EC%97%85-%ED%95%84%ED%84%B0%EB%A7%81-collaborative-filtering/)

지금까지 공부한 알고리즘은 각 유저에 대해 각각의 아이템에 스코어를 줌으로써 아이템에 랭크를 매긴다. 

s(j): j번째 아이템의 스코어 , j = 1,...M M은 아이템

개인화 되지는 않았다. 개별 아이템을 보는 동일한 사람

$$s(j) = \frac{sum_{i \in \Omega_j}r_{ij}}{|\Omega_j|} \\ \Omega_j = \text{set of all users who rated item j} \\ r_{ij} = \text{rating user i gave item j}$$

아이템에만 의존한다.

$$s(i, j) = \frac{sum_{i' \in \Omega_j}r_{i'j}}{|\Omega_j|} \\ i = 1,...,N, N = \text{number of users}$$

유저 i와 아이템 j에 모두 의존한다. i가 아니라 i'인 이유는 i'가 인덱스이기 때문이다. 

$$r_{ij} = \text{rating user i gave item j} \\ i = 1...N, j = 1...M \\ R_{N×M} = \text{user - item ratings matrix of size N × M}$$

user-item 행렬은 NLP에서 term(word)-document matrix라고 생각하면 된다. X(t, d) = # of times term t appears in document d 

- r(i, j)가 대부분 존재 하지 않는 것이 좋다. 행렬이 꽉 차 있으면 수학적으로는 좋지만 비즈니스적으로는 좋지 않다. 추천할 필요가 없기 때문이다. 그래서 행렬은 반드시 sparse 해야한다.
- $s(i, j) = \hat r(i, j) = \text{guess what user i might rate item j}$

**Regression**

- $MSE = \frac{1}{|\Omega|}\sum_{i, j\in\Omega}(r_{ij} - \hat r_{ij})^2 \\ \Omega = \text{set of pairs(i, j) where user i has rated item j}$

### User-User Collaborative Filtering

직관적으로 크게 상관되어있는 유저를 통해 추천을 할 수 있다. 하지만 모든 사람의 영화의 평점을 동일하게 다루기 때문에 문제가 된다. Bob의 s(i, j)는 boc이 캐롤에 일치하지 않더라도 캐롤 엘리스의 점수와 carol의 점수에 의존한다.

일치하지 않은 유저는 가중치를 작게 주고 일치한 유저는 가중치를 크게 준다.

$$s(i, j) = \frac{\sum_{i' \in \Omega_j}w_{ii'}r_{i'j}}{\sum_{i' \in \Omega_j}w_{ii'}}$$

 단순 평균을 냈을 때 또 다른 문제점은 rating의 해석이 각자 다를 수 있다. 다시 말해서, 사람 마다 점수를 주는 방식이 다를 수 있다. 

점수의 절대치 보다는 얼마나 유저의 평균에서 벗어나 잇는지를 신경쓰자. 

$$dev(i, j) = r(i, j) - \bar r_i, \text{for a known rating}$$

$$\hat dev(i, j) = \frac{1}{|\Omega_j|}\sum_{i' \in \Omega_j}r(i', j) - \bar r_i' \text{~~~~~~~~~~~~for a prediction from known ratings} $$

$$s(i,j) = \bar r_i + \frac{\sum_{i' \in \Omega_j}r(i', j) - \bar r_{i'}}{|\Omega_j|} = \bar r_i + \hat dev(i, j)$$

deviation과 weight를 조합하면 

$$s(i, j) = \bar r_i + \frac{\sum_{i' \in \Omega_j}w_{ii'}\{r_{i'j} - \bar r _{i'}\}} {\sum_{i' \in \Omega_j}|w_{ii'}|}$$

가중치를 계산하기위해 피어슨 상관계수를 생각할 수 있지만 행렬이 sparse하다는 문제가 있다. 

$$w_{ii'} = \frac{\sum_{j \in \Psi_{ii'}}(r_{ij} - \bar r_i)(r_{i'j} - \bar r_{i'})}{\sqrt {\sum_{j \in \Psi_{ii'}}(r_{ij} - \bar r_i)^2} \sqrt {\sum_{j \in \Psi_{ii'}}(r_{i'j} - \bar r_{i'})^2}} \\ \Psi_i = \text{set of movies that user i has rated} \\ \Psi_{ii'} = \text{set of movies both user i and i\'~~~have rated} \\ \Psi_{ii'} = \Psi_i \cap \Psi_{i'}$$

코사인 유사도의 확률변수를 중심화시키면 피어슨 상관계수가 된다.

**문제점**

두 유저가 어떠한 공통된 영화도 가지고 있지 않다면 상관계수를 계산할 수 없다. 이러한 유저들을 무시할 수는 있다. 실제로 계산하기 전에 영화가 공통적으로 가져야만하는 수에 대한 하한을 설정한다. 보통 하한은 5로 정한다. 다시 말해서, 두 명의 유저가 적어도 5영화를 공통으로 가지고 있지 않다면 계산하지 않는다. 

실무에서는 영화 j에 대한 평점을 모든 유저에 대해서 합하지 않는다. 오래걸리기 때문이다. 가중치를 사전에 계산해서 재사용함으로써 시간을 줄인다. 모든 유저에 대해서 합하는 대신에, 가장 큰 가중치를 취한다. 모든 유저에 대해 미리 계산할 때 K nearest neighbor를 사용한다(k = 25~50). 모든 유저를 포함하지 않는 것에 대한 한 가지 단점은 유용한 데이터를 버린다는 것이다. 상관계수에 기반하여 K 개의 가장 가까운 것을 선택한다. 

유저 한명에 대한 추천을 한다고 하자. s(i, j)를 계산하는데 O(MN) 정렬하는데 O(Mlog(M))이 필요하다. 그래서 총 O(MN) + O(Mlog(M))시간이 걸린다. 이것은 이론적인것이고 실제로는 user-user 가중치를 미리 계산하고 나와 비슷한 몇몇의 유저에 대해서만 계산한다. top가 K라면 O(MK). 자료구조의 크기를 제한한다면 log(L)로 만들 수 있고 이 때 L은 취하려고 하는 아이템의 수이다. 그러면 정렬에 대한 시간 복잡도는 O(Mlog(L))

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pandas as pd

# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('../large_files/movielens-20m-dataset/rating.csv')

# note:
# user ids are ordered sequentially from 1..138493
# with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!

# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# add them to the data frame
# takes awhile
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('../large_files/movielens-20m-dataset/edited_rating.csv', index=False)
```

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
from collections import Counter

# load in the data
# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('../large_files/movielens-20m-dataset/edited_rating.csv')
print("original dataframe size:", len(df))

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

# number of users and movies we would like to keep
n = 10000
m = 2000

user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [m for m, c in movie_ids_count.most_common(m)]

# make a copy, otherwise ids won't be overwritten
df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

# need to remake user ids and movie ids since they are no longer sequential
new_user_id_map = {}
i = 0
for old in user_ids:
  new_user_id_map[old] = i
  i += 1
print("i:", i)

new_movie_id_map = {}
j = 0
for old in movie_ids:
  new_movie_id_map[old] = j
  j += 1
print("j:", j)

print("Setting new ids")
df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)
# df_small.drop(columns=['userId', 'movie_idx'])
# df_small.rename(index=str, columns={'new_userId': 'userId', 'new_movie_idx': 'movie_idx'})
print("max user id:", df_small.userId.max())
print("max movie id:", df_small.movie_idx.max())

print("small dataframe size:", len(df_small))
df_small.to_csv('../large_files/movielens-20m-dataset/small_rating.csv', index=False)
```

overwrite를 위해서는 pd.df.copy()로 deep copy를 해줘야 한다는 점을 잊지 말자. deep copy를 하지 않으면 새로운 데이터프레임에서 작업하더라도 원래의 데이터프레임에서의 포인터만을 얻을 뿐이다. 

위 코드는 아래와 같다. 

```sql
select * from dataframe where used_id in user_ids and movie_id in movie_ids
```

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# load in the data
# https://www.kaggle.com/grouplens/movielens-20m-dataset
df = pd.read_csv('large_files/movielens-20m-dataset/small_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# a dictionary to tell us which users have rated which movies
user2movie = {}
# a dicationary to tell us which movies have been rated by which users
movie2user = {}
# a dictionary to look up ratings
usermovie2rating = {}
print("Calling: update_user2movie_and_movie2user")
count = 0
def update_user2movie_and_movie2user(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/cutoff))

  i = int(row.userId)
  j = int(row.movie_idx)
  if i not in user2movie:
    user2movie[i] = [j]
  else:
    user2movie[i].append(j)

  if j not in movie2user:
    movie2user[j] = [i]
  else:
    movie2user[j].append(i)

  usermovie2rating[(i,j)] = row.rating
df_train.apply(update_user2movie_and_movie2user, axis=1)

# test ratings dictionary
usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test")
count = 0
def update_usermovie2rating_test(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/len(df_test)))

  i = int(row.userId)
  j = int(row.movie_idx)
  usermovie2rating_test[(i,j)] = row.rating
df_test.apply(update_usermovie2rating_test, axis=1)

# note: these are not really JSONs
with open('user2movie.json', 'wb') as f:
  pickle.dump(user2movie, f)

with open('movie2user.json', 'wb') as f:
  pickle.dump(movie2user, f)

with open('usermovie2rating.json', 'wb') as f:
  pickle.dump(usermovie2rating, f)

with open('usermovie2rating_test.json', 'wb') as f:
  pickle.dump(usermovie2rating_test, f)
```

판다스 데이터프레임이 SQL 테이블과 비슷하다. SQL이 lookup을 빠르게 해주지만 판다스는 어떨까?  딕셔너리를 사용하겠다.  배열을 사용하면 O(MN)이 걸리지만 딕셔너리를 사용하면 O($|\Omega|)$ (여기서 오메가는 전체 평가수) 만큼 걸리므로 딕셔너리가 빠르다. 

json은 반드시 key가 문자열이여야 하므로 pickle을 이용해 딕셔너리 객체 자체를 저장하였다. pickle은 파이썬의 어떤 객체도 저장하고 로드할 수 있다. 

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

# load in the data
df = pd.read_csv('large_files/movielens-20m-dataset/edited_rating.csv')
# df = pd.read_csv('../large_files/movielens-20m-dataset/small_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

A = lil_matrix((N, M))
print("Calling: update_train")
count = 0
def update_train(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/cutoff))

  i = int(row.userId)
  j = int(row.movie_idx)
  A[i,j] = row.rating
df_train.apply(update_train, axis=1)

# mask, to tell us which entries exist and which do not
A = A.tocsr()
mask = (A > 0)
save_npz("Atrain.npz", A)

# test ratings dictionary
A_test = lil_matrix((N, M))
print("Calling: update_test")
count = 0
def update_test(row):
  global count
  count += 1
  if count % 100000 == 0:
    print("processed: %.3f" % (float(count)/len(df_test)))

  i = int(row.userId)
  j = int(row.movie_idx)
  A_test[i,j] = row.rating
df_test.apply(update_test, axis=1)
A_test = A_test.tocsr()
mask_test = (A_test > 0)
save_npz("Atest.npz", A_test)
```

[Scipy sparse matrix handling](https://lovit.github.io/nlp/machine%20learning/2018/04/09/sparse_mtarix_handling/)

[scipy.sparse.lil_matrix - SciPy v1.5.1 Reference Guide](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html)

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict

with open('user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

# to find the user similarities, you have to do O(N^2 * M) calculations!
# in the "real-world" you'd want to parallelize this
# note: we really only have to do half the calculations, since w_ij is symmetric
K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use
for i in range(N):
  # find the 25 closest users to user i
  movies_i = user2movie[i]
  movies_i_set = set(movies_i)

  # calculate avg and deviation
  ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    # don't include yourself
    if j != i:
      movies_j = user2movie[j]
      movies_j_set = set(movies_j)
      common_movies = (movies_i_set & movies_j_set) # intersection
      if len(common_movies) > limit:
        # calculate avg and deviation
        ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:  
          del sl[-1] # 비효율적인거 같다.

  # store the neighbors
  neighbors.append(sl)

  # print out useful things
  if i % 1 == 0:
    print(i)

# using neighbors, calculate train and test MSE

def predict(i, m):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have rated the same movie
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction

train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usermovie2rating_test.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)

# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))
```

[Sorted Containers](http://www.grantjenks.com/docs/sortedcontainers/)

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict

with open('user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

# to find the user similarities, you have to do O(N^2 * M) calculations!
# in the "real-world" you'd want to parallelize this
# note: we really only have to do half the calculations, since w_ij is symmetric
K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use
for i in range(N):
  # find the 25 closest users to user i
  movies_i = user2movie[i]
  movies_i_set = set(movies_i)

  # calculate avg and deviation
  ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    # don't include yourself
    if j != i:
      movies_j = user2movie[j]
      movies_j_set = set(movies_j)
      common_movies = (movies_i_set & movies_j_set) # intersection
      if len(common_movies) > limit:
        # calculate avg and deviation
        ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:  
          del sl[-1] # 비효율적인거 같다.

  # store the neighbors
  neighbors.append(sl)

  # print out useful things
  if i % 1 == 0:
    print(i)

# using neighbors, calculate train and test MSE

def predict(i, m):
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += -neg_w * deviations[j][m] # 여기서 영화 m이 없어서 keyerror 발생할 수 있음
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have rated the same movie
      # don't want to do dictionary lookup twice
      # so just throw exception
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction

train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usermovie2rating_test.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)

# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))
```

## Item-Item Collaborative Filtering

User-User collaborative filtering이 user-item matrix에서 비슷한 행(유저) 벡터를 기준으로 추천을 한다면  Item-Item collaborative filtering은 비슷한 열(item) 벡터를 기준으로 추천을 한다. 

$$w_{jj'} = \frac{\sum_{i \in \Omega_{jj'}}(r_{ij} - \bar r_j)(r_{ij'} - \bar r_{j'})}{\sqrt {\sum_{i \in \Omega_{jj'}}(r_{ij} - \bar r_j)^2} \sqrt {\sum_{i \in \Omega_{jj'}}(r_{ij'} - \bar r_{j'})^2}} \\ \Omega_j = \text{users who rated item j} \\ \Omega_{jj'} = \text{users who rated item j and item j\'~~~have rated} \\ \bar r_i = \text{average rating for item j}$$

user i 가 j'를 좋아한다고 가정해보자. j와 j'가 아주 비슷해서 w_jj' 가 아주 크다. 그래서 s(i, j)는 item j에 대한 총 예측에 큰 양의 값을 기여한다. 

행과 열을 바꿔서 user-user cf와 item-item cf를 생각할 수 있다. 두 방법은 수학적으로 동일하지만, 여전히 차이는 존재한다. 2개의 아이템을 비교할 때가 2명의 유저를 비교할 때 보다 더 많은 데이터를 필요로 한다. item-based cf가 더 많은 데이터를 학습시키기 때문에 더 정확하다. 

 item-item cf는 $O(M^2N)$인 반면, user-user cf는 $O(N^2M)$이다. N >> M이므로 item-based cf가 더 빠르다. 

item-based cf도 이웃을 20으로 제한해서 계산해보자.

$$\hat d(i, j) = \sum_{i' \in \Omega_j}w_{ii'}d(i', j)$$

위 식에서 input feature를 d(i', j)라고 생각하고 output prediction을 $\hat d(i, j)$라고 생각하면 선형 회귀와 같다. 

실제로 사용할때는, 너무 정확하면 문제가 될 수 있다. 추천의 다양성이 떨어지기 때문이다. 그래서 item-based 보다 user-based를 사용하는 것이 나을 수도 있다. 

### Cold Start Problem

데이터가 적으면 상관계수를 계산할 수 없지만 아예 없다면 어떡할까? prior 설정해서 평균을 계산한다. —> 베이지안 방법

### Not necessarily movies/ ratings

다른것들도 가능. 

# Matrix Factorization and Deep Learning

## Matrix Factorization

전체 user-item 행렬은 메모리에 올릴 수 없었다. 

$$\hat R = WU^T$$

### **Interpretation**

K = 5라고 가정하자. 그리고 

- Action/adventure
- Comedy
- Romance
- Horror
- Animation

$w_i(1) = \text{how much user i likes action} \\ w_i(2) = \text{how much user i likes comedy, etc} \\ u_j(1) = \text{how much movie j contains action} \\ u_j(2) = \text{how much movie j likes comedy, etc}$

$$w_i^Tu_j = ||w_i||||u_j||cos\theta \propto sim(i, j)
$$

각 피쳐는 latent feature이고, k는 latent dimensionality이다. 

예를 들어, "왜 유저 i가 파워 레인져를 좋아할까"→ 숨겨진 이유: 유저 i가 액션을 좋아하고 파워 레인저가 액션이기 때문 

피쳐의 의미를 검사하기 전에는 알 수 없다. 비지도 학습이므로 의미를 부여해야함. 

Truncated SVD로 차원 축소 가능 

### **Loss**

 

$$J = \sum_{i, j\in\Psi_i}(r_{ij} - \hat r_{ij})^2 = \sum_{i, j\in\Psi_i}(r_{ij} - w_i^T u_{j})^2$$

위 식을 w에 관하여 풀어보자.

w_i에 관하여 미분하면

$$\frac{\partial{J}}{\partial w_i} = 2\sum_{i, j\in\Psi}(r_{ij} - w_i^T u_{j})(-u_j) = 0
$$

$$\sum_{j\in\Psi_i}(w_i^T u_{j})u_j = \sum_{j\in\Psi_i} r_{ij}u_j $$

내적은 교환법칙이 성립하므로 

$$\sum_{j\in\Psi_i}(u_j^T w_i)u_j = \sum_{j\in\Psi_i} r_{ij}u_j $$

스칼라 × 벡터 = 벡터 × 스칼라 

$$\sum_{j\in\Psi_i}u_j(u_j^T w_i) = \sum_{j\in\Psi_i} r_{ij}u_j $$

합이 i에 영향을 받지 않으므로 

$$(\sum_{j\in\Psi_i}u_ju_j^T) w_i = \sum_{j\in\Psi_i} r_{ij}u_j \\ w_i = (\sum_{j\in\Psi_i}u_ju_j^T) ^{-1}\sum_{j\in\Psi_i} r_{ij}u_j $$

이번에는 U에 관하여 풀어보자.

U_j에 관하여 미분해서 쭈욱 풀어보면 

$$u_j = (\sum_{j\in\Omega_j}w_iw_i^T) ^{-1}\sum_{j\in\Omega_j} r_{ij}u_j $$

방정식은 하나인데 2개의 변수를 가져서 문제가 된다고 생각할 수 있다. 하지만 단지 하나의 전역 방정식이 아니라 모델의 특성이다. W와 U는 업데이트 해야할 파라미터이다. W와 U를 랜덤으로 초기화한 다음 루프 내에서 두 파라미터를 업데이트한다. 머신러닝에서 이 알고리즘을 alternating least square라고 한다. 각 단계에서 극솟값으로 근접한다는 것이 증명되어 있다. W와 U의 업데이트 순서는 상관이 없다. 

### Expanding Our Model

협업 필터링에서 user의 movie rating의 bias문제를 해결하기 위해 deviation을 사용했었다. 마찬가지로, bias term을 추가시킬 수 있다.  matrix factorization에서는 3개의 bias term을 추가시킨다. 선형 회귀에서는 하나의 bias term(intercep)을 추가 시켰다는 것을 기억하자.

$$\hat r_{ij} = w_i^Tu_j + b_i + c_j + \mu \\ b_i = \text{user bias} \\ c_j = \text{movie bias} \\ \mu = \text{global average} $$

**movie bias**

영화 아바타를 생각해보자. latent feature로 Sci-Fi, Aliens, Humans fighting aliens를 가지고 있다고 하자. battlefield earth라는 영화도 이러한 특성을 가지고 있다. 하지만 평가가 매우 안좋다. 따라서 movie-specific bias는 필요하다. 

**Training**

$$J = \sum_{i, j \in \Omega}(r_{ij} - \hat r_{ij})^2 \\ \hat r_{ij} = w_i^Tu_j + b_i + c_j + \mu$$

w와 u는 위에서 보여준것처럼 미분하면 된다. b_i는 조금 다른 점이 있다. Sigma 내에 여전히 b_i가 있기 때문에 b_i를 Sigma 밖으로 꺼낼 때 유의해야한다. 

$$\frac{\partial{J}}{\partial b_i}  = 2\sum_{j \in \Psi_i}(r_{ij} - w_i^Tu_j - b_i - c_j - \mu)(-1) = 0 \\ b_i = \frac{1}{|\Psi_i|}\sum_{j \in \Psi_i}(r_{ij} - w_i^Tu_j - c_j - \mu)$$

c_j도 마찬가지이다.

$$\frac{\partial{J}}{\partial c_j}  = 2\sum_{i \in \Omega_j}(r_{ij} - w_i^Tu_j - b_i - c_j - \mu)(-1) = 0 \\ c_j = \frac{1}{|\Omega_j|}\sum_{i \in \Omega_j}(r_{ij} - w_i^Tu_j - b_i - \mu)$$

### Regularization

선형 회귀에서

Model: $\hat y = w^Tx$

Objective: $J = \sum^N_{i = 1}(y_i - \hat y_i)^2 + \lambda||w||_2^2$

Solution: $w = (\lambda I + X^TX)^{-1}X^Ty$

$$J = \sum_{i, j \in \Omega}(r_{ij} - \hat r_{ij})^2 + \lambda(||W||_F^2 + ||U||_F^2 + ||b||_F^2 + ||c||_F^2)$$

여기서 $||*||_F$  는 Frobenius norm이다. 

Frobenius norm은 다음과 같다.

![Udemy%20Recommender%20Systems%20and%20Deep%20Learning%20in%20Pyt%2040127c6064474bdc8b7a7cd88dbe8ac2/Untitled%201.png](Udemy%20Recommender%20Systems%20and%20Deep%20Learning%20in%20Pyt%2040127c6064474bdc8b7a7cd88dbe8ac2/Untitled%201.png)

$$||W||_F^2 = \sum^N_{i = 1}\sum^K_{k = 1}|w_{ik}|^2 = \sum^N_{i = 1}||w_i||_2^2 = \sum^N_{i = 1}w_i^Tw_i$$

미분해서 w_i에 관해 나타내면 

$$w_i = (\sum_{j \in \Psi_i}u_ju_j^T + \lambda I )^{-1}\sum_{j \in \Psi_i}(r_{ij} - b_i - c_j - \mu)u_j$$

u_j도 마찬가지이다.

b_i와 c_j를 푸는 것에 이전처럼 유의하자.

$$b_i \Big\{ (\sum_{j \in \Psi_i} 1+ \lambda \Big\} = \sum_{j \in \Psi_i}(r_{ij} - w_i^Tu_j - c_j - \mu)$$

$$b_i = \frac{1}{|\Psi_i| + \lambda}\sum_{j \in \Psi_i}(r_{ij} - w_i^Tu_j - c_j - \mu)$$

```python
N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

# initialize variables
K = 10 # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu

def get_loss(d):
  # d: (user_id, movie_id) -> rating
  N = float(len(d))
  sse = 0
  for k, r in d.items():
    i, j = k
    p = W[i].dot(U[j]) + b[i] + c[j] + mu
    sse += (p - r)*(p - r)
  return sse / N

# train the parameters
epochs = 25
reg =20. # regularization penalty
train_losses = []
test_losses = []
for epoch in range(epochs):
  print("epoch:", epoch)
  epoch_start = datetime.now()
  # perform updates

  # update W and b
  t0 = datetime.now()
  for i in range(N):
    # for W
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for b
    bi = 0
    for j in user2movie[i]:
      r = usermovie2rating[(i,j)]
      matrix += np.outer(U[j], U[j])
      vector += (r - b[i] - c[j] - mu)*U[j]
      bi += (r - W[i].dot(U[j]) - c[j] - mu)

    # set the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + reg)

    if i % (N//10) == 0:
      print("i:", i, "N:", N)
  print("updated W and b:", datetime.now() - t0)

  # update U and c
  t0 = datetime.now()
  for j in range(M):
    # for U
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for c
    cj = 0
    try:
      for i in movie2user[j]:
        r = usermovie2rating[(i,j)]
        matrix += np.outer(W[i], W[i])
        vector += (r - b[i] - c[j] - mu)*W[i]
        cj += (r - W[i].dot(U[j]) - b[i] - mu)

      # set the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie2user[j]) + reg)

      if j % (M//10) == 0:
        print("j:", j, "M:", M)
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  print("updated U and c:", datetime.now() - t0)
  print("epoch duration:", datetime.now() - epoch_start)

  # store train loss
  t0 = datetime.now()
  train_losses.append(get_loss(usermovie2rating))

  # store test loss
  test_losses.append(get_loss(usermovie2rating_test))
  print("calculate cost:", datetime.now() - t0)
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])

print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
```

### Vectorized

앞에서는 W와 U를 구하기 위해 벡터의 외적을 계산한 후, 루프를 통하여 합하였다. 여기서는 바로 행렬의 내적을 하였다. 기존의 usermovie2rating 행렬이 성분 하나가 하나의 유저와 하나의 영화의 rating을 의미하기 때문에 유저 하나에 대해 여러 영화와 대응하는 rating들을 뽑아 낼 수 있어야 벡터화가 가능하다. 따라서 새로운 객체 user2movierating를 만ㄷ을어준다. 

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy

# load in the data
import os
if not os.path.exists('user2movie.json') or \
   not os.path.exists('movie2user.json') or \
   not os.path.exists('usermovie2rating.json') or \
   not os.path.exists('usermovie2rating_test.json'):
   import preprocess2dict

with open('user2movie.json', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.json', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.json', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.json', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

# convert user2movie and movie2user to include ratings
print("converting...")
user2movierating = {}
for i, movies in user2movie.items():
  r = np.array([usermovie2rating[(i,j)] for j in movies])
  user2movierating[i] = (movies, r)
movie2userrating = {}
for j, users in movie2user.items():
  r = np.array([usermovie2rating[(i,j)] for i in users])
  movie2userrating[j] = (users, r)

# create a movie2user for test set, since we need it for loss
movie2userrating_test = {}
for (i, j), r in usermovie2rating_test.items():
  if j not in movie2userrating_test:
    movie2userrating_test[j] = [[i], [r]]
  else:
    movie2userrating_test[j][0].append(i)
    movie2userrating_test[j][1].append(r)
for j, (users, r) in movie2userrating_test.items():
  movie2userrating_test[j][1] = np.array(r)
print("conversion done")

# initialize variables
K = 10 # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

def get_loss(m2u):
  # d: movie_id -> (user_ids, ratings)
  N = 0.
  sse = 0
  for j, (u_ids, r) in m2u.items():
    p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu
    delta = p - r
    sse += delta.dot(delta)
    N += len(r)
  return sse / N

# train the parameters
epochs = 25
reg = 20. # regularization penalty
train_losses = []
test_losses = []
for epoch in range(epochs):
  print("epoch:", epoch)
  epoch_start = datetime.now()
  # perform updates

  # update W and b
  t0 = datetime.now()
  for i in range(N):
    m_ids, r = user2movierating[i]
    matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
    vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids])
    bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum()

    # set the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + reg)

    if i % (N//10) == 0:
      print("i:", i, "N:", N)
  print("updated W and b:", datetime.now() - t0)

  # update U and c
  t0 = datetime.now()
  for j in range(M):
    try:
      u_ids, r = movie2userrating[j]
      matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K) * reg
      vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])
      cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()

      # set the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie2user[j]) + reg)

      if j % (M//10) == 0:
        print("j:", j, "M:", M)
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  print("updated U and c:", datetime.now() - t0)
  print("epoch duration:", datetime.now() - epoch_start)

  # store train loss
  t0 = datetime.now()
  train_losses.append(get_loss(movie2userrating))

  # store test loss
  test_losses.append(get_loss(movie2userrating_test))
  print("calculate cost:", datetime.now() - t0)
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])

print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
```

### SVD

MF와 SVD의 관계는 어떻게 될까? 

$$\hat x_{ij} = \sum_k u_{ik}s_{kk}v_{kj} = \sum_k(u_{ik}s_{kk})v_{kj} = \sum_k u'_{ik}v_{kj} = u_i'^Tv_j$$

따라서 동일하다.

N > M이라고 가정하면, rank(X) = M 

$X^TX \text{and} XX^T$를 찾는다. $X^TX$는 X가 중심화되어 있다면 공분산 행렬에 비례한다. $X^TX, XX^T$는 같은 고윳값을 가진다. U와 V는 orthonormal임을 기억하자. 

X가 결측치를 가지고 있으면 SVD는 작동하지 않는다.

**SVD++**

netflix contest 에서 simon Funk는 FunkSVD를 도입하였는데 이것은 stochastic gradient descent를 사용한다. 

$$min_{p, q, b}\sum_{u, i}(r_{ui} - q_i^T(p_u + |N(u)|^{-1/2}\sum_{j\in N(u)}y_i)  - b_u - b_i - \mu)^2 + \lambda(||P||_F^2 + ||Q||_F^2 + ||b_u||_F^2 + ||b_i||_F^2)$$

$|N(u)|^{-1/2}\sum_{j\in N(u)}y_i$는 오직 명시적(explicit)인 것의 효과를 포함하는 p(u)에 반대되는 "암시적(implicit)"정보의 효과이다. 평가된 아이템을 유저가 좋아할 가능성이 임의의 평가되지 않은 아이템보다 더 높다. 이 항에 대한 흥미로운 다른 점은 유저 factor를 포함하지 않는다는 것이다. 이것은 새로운 유저에 대해 좋은 유저 팩터를 떠올리기에 충분한 데이터가 없을 때 긍정적인 영향을 준다.

SVD++는 Asymmetric 의 간소화된 버젼이다. 넷플릭스 데이터셋에서 SVD++가 더 좋았던 이유는 데이터셋이 오직 명시적 rating을 가졌고 암시적 데이터를 추론되었기 때문이다. 

## Probabilistic Matrix Factorization

구글드라이브에 있는 논문 참고.

### Bayesian Matrix Factorization

$$E(r_{ij}|R) = \int r_{ij}p(r_{ij}|R)dr_{ij} \\ E(r_{ij}|R) = \int r_{ij}\underbrace {p(r_{ij}|W, U)}_{\text{Our original Gaussian}} \underbrace{p(W, U|R)}_{\text{Our posterior}}dWdUdr_{ij} $$

여기서 

$$\int r_{ij}p(r_{ij} | W, U)dr_{ij} = E(r_{ij} | W, U) = w_i^T u_j \\ r_{ij} \sim N(w_i^Tu_j, \sigma^2)$$

이기 때문에 

$$\begin{aligned} E(r_{ij}|R) &= \int w_i^Tu_j p(W, U | R)dWdU \\ &= E(w_i^Tu_j | R) \end{aligned} $$

기댓값은 다음과 같이 근사시킬 수 있다.

$$E(w_i^Tu_j | R) \approx \frac{1}{T}\sum^T_{i = 1}w_i^{(t)T}u_j^{(t)}$$

MCMC의 gibbs sampling을 사용한다. 

![Udemy%20Recommender%20Systems%20and%20Deep%20Learning%20in%20Pyt%2040127c6064474bdc8b7a7cd88dbe8ac2/Untitled%202.png](Udemy%20Recommender%20Systems%20and%20Deep%20Learning%20in%20Pyt%2040127c6064474bdc8b7a7cd88dbe8ac2/Untitled%202.png)

딥마인드의 A.Mnih가 쓴 베이지안 pmf 논문을 참고하자. 

word embedding은 모든 단어를 대응하는 피쳐 벡터로 매핑하는 것을 말한다. 뉴럴 넷은 인풋으로 숫자를 받는다. 하지만 단어는 범주형 객체이기 때문에 embedding이 필요하다. 

bias term은 Embedding(N, 1)을 하면 되는데 왜그렇지..

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# load in the data
df = pd.read_csv('../large_files/movielens-20m-dataset/edited_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10 # latent dimensionality
mu = df_train.rating.mean()
epochs = 15
reg = 0. # regularization penalty

# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(u) # (N, 1, K)
m_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(m) # (N, 1, K)

# subsubmodel = Model([u, m], [u_embedding, m_embedding])
# user_ids = df_train.userId.values[0:5]
# movie_ids = df_train.movie_idx.values[0:5]
# print("user_ids.shape", user_ids.shape)
# p = subsubmodel.predict([user_ids, movie_ids])
# print("p[0].shape:", p[0].shape)
# print("p[1].shape:", p[1].shape)
# exit()

u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)

# submodel = Model([u, m], x)
# user_ids = df_train.userId.values[0:5]
# movie_ids = df_train.movie_idx.values[0:5]
# p = submodel.predict([user_ids, movie_ids])
# print("p.shape:", p.shape)
# exit()

x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (N, 1)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9), # SGD가 Adam보다 더 좋았다고 한다. 
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)

# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()
```

## Neural Network

MF는 선형이였다. 딥러닝으로 구현하게 되면 하이퍼파라미터는 어떻게 튜닝할 것인가?

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# load in the data
df = pd.read_csv('../large_files/movielens-20m-dataset/edited_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10 # latent dimensionality
mu = df_train.rating.mean()
epochs = 15
# reg = 0.0001 # regularization penalty

# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u) # (N, 1, K)
m_embedding = Embedding(M, K)(m) # (N, 1, K)
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
x = Concatenate()([u_embedding, m_embedding]) # (N, 2K)

# the neural network
x = Dense(400)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(100)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
x = Dense(1)(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)

# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()
```

## Residual Learning

 

```python
# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# load in the data
df = pd.read_csv('../large_files/movielens-20m-dataset/edited_rating.csv')

N = df.userId.max() + 1 # number of users
M = df.movie_idx.max() + 1 # number of movies

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10 # latent dimensionality
mu = df_train.rating.mean()
epochs = 15
reg = 0. # regularization penalty

# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u) # (N, 1, K)
m_embedding = Embedding(M, K)(m) # (N, 1, K)

##### main branch
u_bias = Embedding(N, 1)(u) # (N, 1, 1)
m_bias = Embedding(M, 1)(m) # (N, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)
x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (N, 1)

##### side branch
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
y = Concatenate()([u_embedding, m_embedding]) # (N, 2K)
y = Dense(400)(y)
y = Activation('elu')(y)
# y = Dropout(0.5)(y)
y = Dense(1)(y)

##### merge
x = Add()([x, y])

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)

r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)

# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()
```

## Residual Learning