import json
import mne

event_codes = {
    'resting_EO': 1,
    'resting_EC': 2,
    'white_noise': 3,
    'EO/class_1_A': 111,
    'EO/class_1_B': 112,
    'EO/class_2_A': 121,
    'EO/class_2_B': 122,
    'EO/class_3_A': 131,
    'EO/class_3_B': 132,
    'EO/class_4_A': 141,
    'EO/class_4_B': 142,
    'EC/class_1_A': 211,
    'EC/class_1_B': 212,
    'EC/class_2_A': 221,
    'EC/class_2_B': 222,
    'EC/class_3_A': 231,
    'EC/class_3_B': 232,
    'EC/class_4_A': 241,
    'EC/class_4_B': 242,

}


def generate_em_mne_events(file, trials, part_one=True, part_two=False):
    raw_file = open(file)
    raw_json = json.load(raw_file)
    status_data = raw_json['recording']['statusData']
    i = 0
    t_idx = []
    while i < len(status_data):
        if status_data[i] == 1.0:
            # print("Second: " + str(i/250) + " - Index: " + str(i))
            t_idx.append(i)
            i += 250
        else:
            i += 1
    events = []
    if part_one:
        events.append([t_idx[1], 0, event_codes['resting_EO']])
        events.append([t_idx[2], 0, event_codes['resting_EC']])
        events.append([t_idx[3], 0, event_codes['white_noise']])
        events.append([t_idx[4], 0, event_codes[trials[0]]])
        events.append([t_idx[5], 0, event_codes['white_noise']])
        events.append([t_idx[6], 0, event_codes[trials[1]]])
        events.append([t_idx[7], 0, event_codes['white_noise']])
        events.append([t_idx[8], 0, event_codes[trials[2]]])
        events.append([t_idx[9], 0, event_codes['white_noise']])
        events.append([t_idx[10], 0, event_codes[trials[3]]])
        events.append([t_idx[11], 0, event_codes['white_noise']])
        events.append([t_idx[12], 0, event_codes[trials[4]]])
        events.append([t_idx[13], 0, event_codes['white_noise']])
        events.append([t_idx[14], 0, event_codes[trials[5]]])
        events.append([t_idx[15], 0, event_codes['white_noise']])
        events.append([t_idx[16], 0, event_codes[trials[6]]])
        events.append([t_idx[17], 0, event_codes['white_noise']])
        events.append([t_idx[18], 0, event_codes[trials[7]]])
    if part_two:
        events.append([t_idx[1], 0, event_codes['white_noise']])
        events.append([t_idx[2], 0, event_codes[trials[0]]])
        events.append([t_idx[3], 0, event_codes['white_noise']])
        events.append([t_idx[4], 0, event_codes[trials[1]]])
        events.append([t_idx[5], 0, event_codes['white_noise']])
        events.append([t_idx[6], 0, event_codes[trials[2]]])
        events.append([t_idx[7], 0, event_codes['white_noise']])
        events.append([t_idx[8], 0, event_codes[trials[3]]])
        events.append([t_idx[9], 0, event_codes['white_noise']])
        events.append([t_idx[10], 0, event_codes[trials[4]]])
        events.append([t_idx[11], 0, event_codes['white_noise']])
        events.append([t_idx[12], 0, event_codes[trials[5]]])
        events.append([t_idx[13], 0, event_codes['white_noise']])
        events.append([t_idx[14], 0, event_codes[trials[6]]])
        events.append([t_idx[15], 0, event_codes['white_noise']])
        events.append([t_idx[16], 0, event_codes[trials[7]]])
        events.append([t_idx[17], 0, event_codes['resting_EO']])
        events.append([t_idx[18], 0, event_codes['resting_EC']])
    evefile = file.split(".json")[0] + ".eve"
    mne.write_events(evefile, events)
    return events


def add_em_events_melo_raw(file, events):
    raw_file = open(file)
    raw_json = json.load(raw_file)
    raw_json["events"] = events

    # We save the changes
    with open(file + '.json', 'w', encoding='utf-8') as f:
        json.dump(raw_json, f, indent=4)
        print("Successfully saved events into ", file)


def parse_playlist(playlist_raw, condition):
    playlist = []
    con_a = ''
    con_b = ''
    for elem in playlist_raw:
        song_a = elem[0]
        song_b = elem[1]
        song_a = song_a.split('res\\playlist\\')[1].split('.ogg')[0]
        song_b = song_b.split('res\\playlist\\')[1].split('.ogg')[0]
        if condition == '1EOEC':
            con_a = 'EO/'
            con_b = 'EC/'
        elif condition == '2ECEO':
            con_a = 'EC/'
            con_b = 'EO/'
        playlist.append(con_a + song_a)
        playlist.append(con_b + song_b)
    return playlist
