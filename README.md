LSTM 기반 클래식 멜로디 생성기

LSTM 모델을 사용하여 클래식 피아노 음악 데이터를 학습하고, 새로운 멜로디를 생성한다. 

----------
목표: 시계열 데이터의 특성을 이해하고, 딥러닝 모델을 통해 창의적인 결과물 생성

핵심 기술: TensorFlow, Keras, LSTM, Music21

데이터셋: Classical Piano MIDI Dataset

----------
모델 구조

장기 의존성 학습에 유리한 LSTM을 사용했다.

----------
실행 방법
1. ‘requirements.txt’ 설치
2. MIDI 데이터셋 준비
3. ‘train.py’를 통해 데이터셋 학습
4. ‘generate.py’를 통해 결과물 생성
