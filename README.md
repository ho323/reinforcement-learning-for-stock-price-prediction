# Reinforcement Learning Trader
강화학습을 이용한 주식 트레이더 AI

---
## Development Environment
#### OS
- Mac
#### 개발 언어 
- Python 3
#### 라이브러리
- Tensorflow 2.5
- Keras 2.5
- Numpy 1.19.5
- Pandas 1.3.2
- Matplotlib 3.4.3

---
## Modules
### agent.py  
- 투자 행동 수행
- 투자금과 보유 주식 관리
### data_manager.py  
- 차트 데이터와 학습 데이터 생성
- 자질 벡터 정의
- 데이터 전처리
### enviroment.py
- 투자할 종목의 차트 데이터를 관리
### learners.py
- 다양한 강화학습 방식을 수행
### main.py
- 다양한 조건으로 강화학습을 수행할 수 있게 프로그램 인자를 구성 
- 입력받은 인자에 따라 학습기 클래스를 이용해 강화학습을 수행
- 학습한 신경망들을 저장
### networks.py
- 가치 신경망과 정책 신경망을 사용하기 위한 신경망(DNN, LSTM, CNN)
### settings.py
- 경로 설정
- 로케일 설정
### utils.py
- 시간 문자열 생성 함수
- 시그모이드(sigmoid) 함수
### visualizer.py
- 신경망을 학습하는 과정에서 보유 주식수, 가치 신경망 출력, 정책 신경망 출력, 정책 신경망 출력, 투자 행동, 포트폴리오 가치 등을 시간에 따라 연속으로 보여주기 위한 시각화 기능을 담당

---
## Reference
파이썬과 케라스를 이용한 딥러닝/강화학습 주식투자 (개정판)을 참고했습니다.  
