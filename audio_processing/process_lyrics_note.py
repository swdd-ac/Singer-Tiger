import json
from collections import Counter

def convert_time_to_seconds(time_str):
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_milliseconds = float(parts[2])
    total_seconds = hours * 3600 + minutes * 60 + seconds_milliseconds
    return total_seconds

def find_most_frequent_note(note_path, text_path):
    with open(note_path, 'r') as f1, open(text_path, 'r') as f2:
        notes_data = json.load(f1)
        text_data = json.load(f2)

    output_json = []

    for text_item in text_data:
        start_time_seconds = convert_time_to_seconds(text_item['start_time'])
        end_time_seconds = convert_time_to_seconds(text_item['end_time'])
        text = text_item['text']
        notes_in_range = []

        for note_item in notes_data:
            if start_time_seconds <= note_item['time'] < end_time_seconds:
                notes_in_range.append(note_item['note'])

        note_counts = Counter(notes_in_range)
        most_common_note = None
        if note_counts:
            most_common_note = note_counts.most_common(1)[0][0]

        output_json.append({"歌詞": text, "音符": most_common_note})

    return json.dumps(output_json, ensure_ascii=False, indent=2)

