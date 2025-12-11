# data_loader.py
import music21
import glob
import numpy as np
import pickle 
from tensorflow.keras.utils import to_categorical
import config

def get_notes():
    """data 폴더 내의 모든 midi 파일을 읽어 음표(Note/Chord) 리스트를 반환합니다."""
    notes = []
    
    for file in glob.glob(config.DATA_PATH):
        try:
            midi = music21.converter.parse(file)
            
            notes_to_parse = None
            try: 
                s2 = music21.instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() 
            except: 
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, music21.note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, music21.chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        
        except Exception as e:
            print(f"Error parsing file {file}: {e}")
            continue
            
    return notes

def prepare_sequences(notes):
    """음표 리스트를 모델 학습용 입력(X)과 출력(Y)으로 변환합니다."""

    note_to_int = dict((note, number) for number, note in enumerate(sorted(list(set(notes)))))
    n_vocab = len(note_to_int)
    
    with open(config.MAPPING_FILE, 'wb') as filepath:
        pickle.dump(note_to_int, filepath)
        
    network_input = []
    network_output = []
    sequence_length = config.SEQUENCE_LENGTH
    
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length] 
        sequence_out = notes[i + sequence_length]  
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab) 

    network_output = to_categorical(network_output, num_classes=n_vocab)
    
    print(f"Total sequences (학습 샘플 수): {n_patterns}")
    print(f"Total unique notes (N_VOCAB): {n_vocab}")
    
    return network_input, network_output, n_vocab