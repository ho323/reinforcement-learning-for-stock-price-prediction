class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 (일반적으로 0.015%)
    TRADING_TAX = 0.0025  # 거래세 (실제 0.25%)
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수