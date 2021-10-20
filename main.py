import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+', default=['030200', '000270', '005380'])  # 강화학습의 환경이 될 주식의 종목 코드입니다. A3C이 경우 여러 개의 종목 코드를 입력합니다.
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3'], default='v1')  # RL Trader이 버전을 명시합니다. 기본값으로 v3을 사용합니다.
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default='a3c')  # 강화학습 방식을 설정합니다. dqn, pg, ac, a2c, a3c, monkey 중에서 하나를 정합니다.
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')  # 가치 신경망과 정책 신경망에서 사용할 신경망 유형을 선택합니다. dnn, lstm, cnn, monkey 중에서 하나를 정합니다. 
    parser.add_argument('--num_steps', type=int, default=1)  # lstm과 cnn에서 사용할 Step 크기를 정합니다. 이 크기만큼 자질 벡터의 크기가 확장됩니다.
    parser.add_argument('--lr', type=float, default=0.001)  # 학습 속도를 정합니다. 0.01, 0.001 등으로 정할 수 있습니다.
    parser.add_argument('--discount_factor', type=float, default=0.9)  # 할인율을 정합니다. 0.9, 0.8 등으로 정할 수 있습니다.
    parser.add_argument('--start_epsilon', type=float, default=1)  # 시작 탐험율을 정합니다. 에포크가 수행되면서 탐험률은 감소합니다. 1. 0.5 등으로 정할 수 있습니다.
    parser.add_argument('--balance', type=int, default=100000000)  # 주식투자 시뮬레이션을 위한 초기 자본금을 설정합니다. 
    parser.add_argument('--num_epoches', type=int, default=100)  # 수행할 에포크 수를 지정합니다. 100, 1000 등으로 정할 수 있습니다. 
    parser.add_argument('--backend', choices=['tensorflow', 'plaidml'], default='tensorflow')  # Keras으 백엔드로 사용할 프레임워크를 설정합니다. tensorflow와 plaidml을 선택할 수 있습니다.
    parser.add_argument('--output_name', default=utils.get_time_str())  # 로그, 가시화 파일, 신경망 모델 등의 출력 파일을 저장할 폴더의 이름
    parser.add_argument('--value_network_name', default='value_network_model')  # 가치 신경망 모델 파일명
    parser.add_argument('--policy_network_name', default='policy_network_model')  # 정책 신경망 모델 파일명
    parser.add_argument('--reuse_models', action='store_false')  # 신경망 모델 재사용 유무 store_true or store_false
    parser.add_argument('--learning', action='store_false')  # 강화학습 유무
    parser.add_argument('--start_date', default='20170101')  # 차트 데이터 및 학습 데이터 시작 날짜
    parser.add_argument('--end_date', default='20191230')  # 차트 데이터 및 학습 데이터 끝 날짜
    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, 
        'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)
        
    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(output_path, '{}_{}_{}_value.h5'.format(args.output_name, args.rl_method, args.net))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(output_path, '{}_{}_{}_policy.h5'.format(args.output_name, args.rl_method, args.net))

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        print(stock_code)
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)
        
        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int((args.balance/10) / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(args.balance / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': args.num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': args.start_epsilon,
            'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_unit': min_trading_unit, 
                'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                args.net = args.rl_method
                args.num_epoches = 1
                args.discount_factor = None
                args.start_epsilon = 1
                args.learning = False
                learner = ReinforcementLearner(**common_params)
            if learner is not None:
                learner.run(learning=args.learning)
                learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_unit': list_min_trading_unit, 
            'list_max_trading_unit': list_max_trading_unit,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
        learner.run(learning=args.learning)
        learner.save_models()