# generate.py
import config
from network import create_model
from data_loader import get_notes
import numpy as np
import music21
import pickle
import random
import os

def generate_music():
    """학습된 가중치를 로드하여 새로운 멜로디를 생성하고 MIDI로 저장합니다."""
    
    notes = get_notes()
    note_to_int = pickle.load(open(config.MAPPING_FILE, 'rb'))
    int_to_note = dict((number, note) for note, number in note_to_int.items())
    n_vocab = len(note_to_int)

    model = create_model(n_vocab, (config.SEQUENCE_LENGTH, 1))
    
    weights_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.keras')]
    if not weights_files:
        print("Error: No weight files found. Please run train.py first.")
        return
        
    latest_weights = sorted(weights_files)[-1]
    full_weights_path = os.path.join(config.CHECKPOINT_DIR, latest_weights)
    model.load_weights(full_weights_path)
    print(f"Loaded weights from: {full_weights_path}")

    start = np.random.randint(0, len(notes) - config.SEQUENCE_LENGTH)
    pattern = notes[start:start + config.SEQUENCE_LENGTH]
    
    prediction_output = []
    
    for note_index in range(500): 
        prediction_input = [note_to_int[note] for note in pattern]
        prediction_input = np.reshape(prediction_input, (1, config.SEQUENCE_LENGTH, 1))
        prediction_input = prediction_input / float(n_vocab)
        
        prediction = model.predict(prediction_input, verbose=0)
        
        index = np.argmax(prediction)
        result = int_to_note[index]
        
        prediction_output.append(result)
        
        pattern.append(result)
        pattern = pattern[1:len(pattern)]

    offset = 0
    output_notes = []

    for pattern_str in prediction_output:
        if ('.' in pattern_str) or pattern_str.isdigit():
            notes_in_chord = pattern_str.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = music21.instrument.Piano()
                notes.append(new_note)
            new_chord = music21.chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = music21.note.Note(pattern_str)
            new_note.offset = offset
            new_note.storedInstrument = music21.instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_melody.mid')
    print("Generation complete! Saved as output_melody.mid")

if __name__ == '__main__':
    generate_music()