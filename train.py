# train.py
import os
from network import create_model
from data_loader import get_notes, prepare_sequences
import config
from tensorflow.keras.callbacks import ModelCheckpoint

def train():
    """모델 학습을 실행하는 메인 함수"""

    notes = get_notes()
    network_input, network_output, n_vocab = prepare_sequences(notes)
    
    model = create_model(n_vocab, (network_input.shape[1], network_input.shape[2]))
    
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    filepath = config.CHECKPOINT_DIR + "/weights-improvement-{epoch:02d}-{loss:.4f}-best.keras"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=1, save_best_only=True, mode='min'
    )
    callbacks_list = [checkpoint]
    
    print("Training Started...")
    model.fit(network_input, network_output, 
              epochs=config.EPOCHS, 
              batch_size=config.BATCH_SIZE, 
              callbacks=callbacks_list)

if __name__ == '__main__':
    train()