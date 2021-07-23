import json
import mne
import glob
import numpy as np

eeg_folder = 'eeg_raw'
metadata_folder = 'metadata'
physio_folder = 'physio_raw'

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

dictionary_utf8 = [
    ["a", "01100001"],
    ["b", "01100010"],
    ["c", "01100011"],
    ["d", "01100100"],
    ["e", "01100101"],
    ["f", "01100110"],
    ["g", "01100111"],
    ["h", "01101000"],
    ["n", "01101110"],
    ["o", "01101111"],
    ["s", "01110011"],
    ["z", "01111010"]
]

trigger_codes_decoder = {
    's': 'start_experiment',
    'o': 'resting_state_eo',
    'z': 'resting_state_ec',
    'n': 'white_noise',
    'a': 'trial_a_1',
    'b': 'trial_b_1',
    'c': 'trial_a_2',
    'd': 'trial_b_2',
    'e': 'trial_a_3',
    'f': 'trial_b_3',
    'g': 'trial_a_4',
    'h': 'trial_b_4'
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
    raw_file.close()
    return events


def add_em_events_melo_raw(file, events):
    raw_file = open(file)
    raw_json = json.load(raw_file)
    raw_json["events"] = events

    # We save the changes
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(raw_json, f, indent=4) # WARNING: indent makes it pretty but consumes way more space on disk!!
        print("Successfully saved events into ", file)
    raw_file.close()


def parse_playlist(playlist_raw, group):
    playlist = []
    con_a = ''
    con_b = ''
    for elem in playlist_raw:
        song_a = elem[0]
        song_b = elem[1]
        song_a = song_a.split('res\\playlist\\')[1].split('.ogg')[0]
        song_b = song_b.split('res\\playlist\\')[1].split('.ogg')[0]
        if group == '1EOEC':
            con_a = 'EO/'
            con_b = 'EC/'
        elif group == '2ECEO':
            con_a = 'EC/'
            con_b = 'EO/'
        playlist.append(con_a + song_a)
        playlist.append(con_b + song_b)
    return playlist


def generate_participants_events(group_path, group_folders, group):
    for participant_id in group_folders:
        # Find and open all the .json files containing EEG data
        file_p1_path = glob.glob(group_path + '/' + participant_id + '/' + eeg_folder + '/p1/' + '*.json')[0]
        file_p2_path = glob.glob(group_path + '/' + participant_id + '/' + eeg_folder + '/p2/' + '*.json')[0]
        file_metadata_path = glob.glob(group_path + '/' + participant_id + '/' + metadata_folder + '/' + '*.json')[0]
        file_p1 = open(file_p1_path)
        file_p2 = open(file_p2_path)
        file_metadata = open(file_metadata_path)

        # Load all the physiological data as numpy arrays
        # bvp = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'BVP.csv')
        # eda = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'EDA.csv')
        # hr = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'HR.csv')
        # temp = np.genfromtxt(group_path + '/' + participant_id + '/' + physio_folder + '/' + 'TEMP.csv')

        # Load all the EEG data as dictionaries
        data_p1 = json.load(file_p1)
        data_p2 = json.load(file_p2)
        metadata = json.load(file_metadata)

        # Parse the metadata
        playlist_p1_raw = metadata['playlist_p1']['val']
        playlist_p2_raw = metadata['playlist_p2']['val']
        p1_timestamp = metadata['playlist_p1']['timestamp']
        p2_timestamp = metadata['playlist_p2']['timestamp']
        playlist_p1 = parse_playlist(playlist_p1_raw, group)
        playlist_p2 = parse_playlist(playlist_p2_raw, group)
        print(participant_id, "...loaded!")

        events_p1 = generate_em_mne_events(file_p1_path, playlist_p1, part_one=True, part_two=False)
        events_p2 = generate_em_mne_events(file_p2_path, playlist_p2, part_one=False, part_two=True)
        add_em_events_melo_raw(file_p1_path, events_p1)
        add_em_events_melo_raw(file_p2_path, events_p2)
        print(participant_id, "...generated events!")

        file_p1.close()
        file_p2.close()
        file_metadata.close()

        # participant['group'] = group
        # participant['data_p1'] = data_p1
        # participant['data_p2'] = data_p2
        # participant['playlist_p1'] = playlist_p1
        # participant['playlist_p2'] = playlist_p2
        # participant['p1_timestamp'] = p1_timestamp
        # participant['p2_timestamp'] = p2_timestamp
        # participants[participant_id] = participant
