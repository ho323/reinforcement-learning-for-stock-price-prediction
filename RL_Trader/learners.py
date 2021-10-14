import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from tqdm import tqdm
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer


'''
속성
stock_code: 강화학습 대상 주식 종목 코드
chart_data: 주식 종목의 차트 데이터
environment: 강화학습 환경 객체
agent: 강화학습 에이전트 객체
training_data: 학습 데이터
value_network: 가치 신경망
policy_network: 정책 신경망

함수
init_value_network(): 가치 신경망 생성 함수
init_policy_network(): 정책 신경망 생성 함수
build_sample(): 환경 객체에서 샘플을 휙득하는 함수
get_batch(): 배치 학습 데이터 생성 함수
update_network(): 가치 신경망 및 정책 신경망 학습 함수
fit(): 가치 신경망 및 정책 신경망 학습 요청 함수
visualize(): 에포크 정보 가시화 함수
run(): 강화학습 수행 함수
save_models(): 가치 신경망 및 정책 신경망 저장 함수
'''

# DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner 등의 클래스가 상속하는 상위 클래스
# 강화학습에 필요한 환경, 에이전트, 신경망 인스턴스들과 학습 데이터를 속성으로 가집니다.
class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta  # 클래스의 메타클래스 정의
    lock = threading.Lock()  # 해당 쓰레드만 공유 데이터에 접근할 수 있는 기능

    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_unit=1, max_trading_unit=2, 
                net='dnn', num_steps=1, lr=0.001, 
                discount_factor=0.9, num_epoches=100,
                balance=10000000, start_epsilon=1,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, balance,
                    min_trading_unit=min_trading_unit,
                    max_trading_unit=max_trading_unit)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리 (강화학습 과정에서 발생하는 각종 데이터를 쌓아두기 위함)
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []  # 포트폴리오 가치
        self.memory_num_stocks = []  # 보유 주식 수
        self.memory_exp_idx = []  # 탐험 위치
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.  # 에포크 동안 학습에서 발생한 손실
        self.itr_cnt = 0  # 수익 발생 횟수
        self.exploration_cnt = 0  # 탐험 횟수
        self.batch_size = 0
        self.learning_cnt = 0  # 학습 횟수
        # 로그 등 출력 경로
        self.output_path = output_path  # 로그, 가시화, 학습모델 등 저장 경로

    # net에 지정된 신경망 종류에 맞게 가치 신경망 생성 
    # 손익률을 회귀분석하는 모델이라 activation은 선형 함수, 손실 함수로 MSE 설정
    def init_value_network(self, shared_network=None, 
            activation='linear', loss='mse'):
        # 이 클래스들은 Network 클래스를 상속하므로 Network 클래스의 함수를 모두 갖고 있음
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        # 경로에 모델이 존재하면 불러옴
        if self.reuse_models and os.path.exists(self.value_network_path):
                self.value_network.load_model(model_path=self.value_network_path)

    # loss에 binary_crossenrtopy인데 output_dim이 3일 경우?
    def init_policy_network(self, shared_network=None, 
            activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network, 
                activation=activation, loss=loss)
        # 경로에 모델이 존재하면 불러옴
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    # 에포크마다 새로 데이터가 쌓이는 변수들을 초기화
    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()  # 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽게 함
        # 학습 데이터에 다음 인덱스 데이터가 존재하면
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()  # TRAINING_DATA_V3 47개
            self.sample.extend(self.agent.get_states())  # 3개
            return self.sample  # 총 50개의 값으로 리스트 구성
        return None

    # 학습 데이터 생성
    @abc.abstractmethod  # 추상 메소드
    def get_batch(self):
        pass  # 하위 클래스들은 이 함수를 구현해야 함

    # 신경망 학습
    def update_networks(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()  # 학습 데이터 생성
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    # 배치 학습 데이터의 크기를 정하고 update_networks() 함수를 호출합니다.
    def fit(self):
        # 배치 학습 데이터 생성 및 신경망 갱신
        _loss = self.update_networks()
        if _loss is not None:
            self.loss += abs(_loss)  # 위 함수로 반환받은 loss값을 loss에 더해주어 에포크 동안의 총 학습 손실을 갖게 됨.
            self.learning_cnt += 1  # 학습 횟수 (나중에 loss를 학습 횟수로 나누어 에포크의 학습 손실)
            self.memory_learning_idx.append(self.training_data_idx)  # 학습 위치 저장

    # 하나의 에포크 관련 정보를 가시화 
    # 대상: 에이전트의 행동, 보유 주식 수, 가치 신경망 출력, 정책 신경망 출력, 포트폴리오 가치, 탐험 위치, 학습 위치 등
    def visualize(self, epoch_str, num_epoches, epsilon):
        # LSTM, CNN 신경망의 경우 환경의 일봉 수보다 (num_steps -1) 만큼 부족하기 때문에 첫 부분에 의미 없는 값을 채워줍니다. 
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv
        
        # PNG 파일로 가시화해서 저장
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir, 
            'epoch_summary_{}.png'.format(epoch_str))
        )

    # ReinforcementLearner 클래스의 핵심 함수
    def run(self, learning=True):
        # 제목
        info = (
            "[{code}] RL:{rl} Net:{net} LR:{lr} "
            "DF:{discount_factor} TU:[{min_trading_unit},{max_trading_unit}]"
        ).format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=self.discount_factor,
            min_trading_unit=self.agent.min_trading_unit, 
            max_trading_unit=self.agent.max_trading_unit,
        )
        with self.lock:
            logging.info(info)  # 계획대로 작동하고 있음을 알림

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)  # 폴더 생성
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))  # 이미 있으면 폴더에 있는 파일들 삭제

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0  # 수익이 발생한 에폭 수 저장

        # 학습 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)  # 양방향에서 데이터를 처리할 수 있는 queue형 자료구조
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = 10 / (epoch + 10) if epoch < self.num_epoches - 1 else 0
                self.agent.reset_exploration()  # exploration_base를 새로 정합니다.
            else:
                epsilon = self.start_epsilon
                self.agent.reset_exploration(alpha=0)

            for i in tqdm(range(len(self.training_data))):
                # 샘플 생성
                next_sample = self.build_sample()  # 환경 객체에서 샘플을 휙득하는 함수
                if next_sample is None:
                    break  # 마지막까지 데이터를 다 읽은 것이므로 반복문 종료

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상 획득
                reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            # 에포크 종료 후 학습
            if learning:
                self.fit()  # 가치 신경망 및 정책 신경망 학습 함수

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(self.num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code, epoch_str, self.num_epoches, epsilon, 
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell, 
                    self.agent.num_hold, self.agent.num_stocks, 
                    self.agent.portfolio_value, self.learning_cnt, 
                    self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            if self.num_epoches == 1 or (epoch + 1) % 10 == 0:
                self.visualize(epoch_str, self.num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time, 
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self, balance=10000000):
        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)
        
        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))
            
            # 신경망에 의한 행동 결정
            action, confidence, _ = self.agent.decide_action(pred_value, pred_policy, 0)
            
            result.append((action, confidence))

        return result


class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            value_max_next = value.max()
        return x, y_value, None


class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_policy[i, action] = 1 if r > 0 else 0
        return x, None, y_policy


class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, 
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps, 
                input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, action] = r + self.discount_factor * value_max_next
            y_policy[i, action] = 1 if r > 0 else 0
            value_max_next = value.max()
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, action] = r + self.discount_factor * value_max_next
            advantage = y_value[i, action] - y_value[i].mean()
            y_policy[i, action] = 1 if advantage > 0 else 0
            value_max_next = value.max()
        return x, y_value, y_policy


class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None, 
        list_chart_data=None, list_training_data=None,
        list_min_trading_unit=None, list_max_trading_unit=None, 
        value_network_path=None, policy_network_path=None,
        **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps, 
            input_dim=self.num_features)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data, 
            min_trading_unit, max_trading_unit) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_unit, list_max_trading_unit
            ):
            learner = A2CLearner(*args, 
                stock_code=stock_code, chart_data=chart_data, 
                training_data=training_data,
                min_trading_unit=min_trading_unit, 
                max_trading_unit=max_trading_unit, 
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(self, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={
                'num_epoches': self.num_epoches, 'balance': self.agent.balance,
                'discount_factor': self.discount_factor, 
                'start_epsilon': self.start_epsilon,
                'learning': learning
            }))
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()